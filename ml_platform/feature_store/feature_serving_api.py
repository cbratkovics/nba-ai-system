"""
Real-time Feature Serving API with caching and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import redis.asyncio as redis
import json
import hashlib
from prometheus_client import make_asgi_app, Counter, Histogram
import logging
import aiohttp
from circuitbreaker import circuit
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Metrics
api_requests = Counter('feature_api_requests_total', 'Total API requests', ['endpoint', 'status'])
api_latency = Histogram('feature_api_latency_seconds', 'API latency')
cache_hits = Counter('feature_cache_hits_total', 'Cache hits', ['feature'])
cache_misses = Counter('feature_cache_misses_total', 'Cache misses', ['feature'])

app = FastAPI(title="NBA Feature Store API", version="1.0.0")

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeatureRequest(BaseModel):
    """Feature request model"""
    entity_id: str = Field(..., description="Entity ID (e.g., player_id)")
    features: List[str] = Field(..., description="List of feature names to retrieve")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context for computation")
    point_in_time: Optional[datetime] = Field(None, description="Historical point-in-time request")


class FeatureResponse(BaseModel):
    """Feature response model"""
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime
    cache_status: Dict[str, str]
    computation_time_ms: float


class BatchFeatureRequest(BaseModel):
    """Batch feature request"""
    entity_ids: List[str] = Field(..., max_items=1000)
    features: List[str] = Field(..., max_items=50)
    context: Optional[Dict[str, Any]] = Field(default={})


class FeatureServingAPI:
    """High-performance feature serving with caching"""
    
    def __init__(self):
        self.redis_pool = None
        self.feature_store = None
        self.cache_ttl = 300  # 5 minutes
        
    async def startup(self):
        """Initialize connections"""
        self.redis_pool = redis.ConnectionPool.from_url(
            "redis://localhost:6379",
            max_connections=100,
            decode_responses=True
        )
        self.redis = redis.Redis(connection_pool=self.redis_pool)
        
        # Initialize feature store client
        from feature_store import FeatureStore
        self.feature_store = FeatureStore({
            'redis_host': 'localhost',
            'redis_port': 6379,
            's3_bucket': 'nba-features',
            'postgres_url': 'postgresql://user:pass@localhost/features',
            'kafka_brokers': 'localhost:9092'
        })
    
    async def shutdown(self):
        """Cleanup connections"""
        if self.redis_pool:
            await self.redis_pool.disconnect()
    
    def _get_cache_key(self, entity_id: str, feature_name: str, 
                      context: Dict[str, Any] = None) -> str:
        """Generate cache key with context"""
        context_str = json.dumps(context or {}, sort_keys=True)
        key_parts = [entity_id, feature_name, context_str]
        key_hash = hashlib.md5(":".join(key_parts).encode()).hexdigest()
        return f"feature:{feature_name}:{entity_id}:{key_hash}"
    
    @circuit(failure_threshold=5, recovery_timeout=30)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def get_feature_with_fallback(self, entity_id: str, feature_name: str,
                                      context: Dict[str, Any]) -> Any:
        """Get feature with circuit breaker and retry logic"""
        # Try cache first
        cache_key = self._get_cache_key(entity_id, feature_name, context)
        
        try:
            cached_value = await self.redis.get(cache_key)
            if cached_value:
                cache_hits.labels(feature=feature_name).inc()
                return json.loads(cached_value), "hit"
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        
        cache_misses.labels(feature=feature_name).inc()
        
        # Compute feature
        try:
            result = await self.feature_store.compute_features(
                [entity_id],
                [feature_name],
                context
            )
            
            if not result.empty:
                value = result.iloc[0]['value']
                
                # Cache asynchronously
                asyncio.create_task(self._cache_feature(cache_key, value))
                
                return value, "computed"
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            
            # Try fallback to default value
            default_value = await self._get_default_value(feature_name)
            if default_value is not None:
                return default_value, "default"
            
            raise
    
    async def _cache_feature(self, key: str, value: Any) -> None:
        """Cache feature value asynchronously"""
        try:
            await self.redis.setex(key, self.cache_ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Failed to cache feature: {e}")
    
    async def serve_features(self, request: FeatureRequest) -> FeatureResponse:
        """Serve features with monitoring"""
        start_time = datetime.utcnow()
        
        features = {}
        cache_status = {}
        
        # Parallel feature retrieval
        tasks = []
        for feature_name in request.features:
            task = self.get_feature_with_fallback(
                request.entity_id,
                feature_name,
                request.context
            )
            tasks.append((feature_name, task))
        
        # Gather results
        for feature_name, task in tasks:
            try:
                value, status = await task
                features[feature_name] = value
                cache_status[feature_name] = status
            except Exception as e:
                logger.error(f"Failed to get feature {feature_name}: {e}")
                features[feature_name] = None
                cache_status[feature_name] = "error"
        
        # Calculate response time
        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return FeatureResponse(
            entity_id=request.entity_id,
            features=features,
            timestamp=datetime.utcnow(),
            cache_status=cache_status,
            computation_time_ms=computation_time
        )
    
    async def serve_batch_features(self, request: BatchFeatureRequest) -> List[FeatureResponse]:
        """Serve features for multiple entities"""
        # Create tasks for all entity-feature combinations
        tasks = []
        
        for entity_id in request.entity_ids:
            single_request = FeatureRequest(
                entity_id=entity_id,
                features=request.features,
                context=request.context
            )
            tasks.append(self.serve_features(single_request))
        
        # Execute in parallel with concurrency limit
        semaphore = asyncio.Semaphore(50)  # Limit concurrent requests
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_task(task) for task in tasks])
        
        return results
    
    async def _get_default_value(self, feature_name: str) -> Any:
        """Get default value for feature"""
        # Feature-specific defaults
        defaults = {
            'player_hot_streak': False,
            'injury_risk': 0.0,
            'projected_minutes': 0.0,
            'rest_days': 0
        }
        
        return defaults.get(feature_name)


# Initialize API
serving_api = FeatureServingAPI()

@app.on_event("startup")
async def startup():
    await serving_api.startup()

@app.on_event("shutdown")
async def shutdown():
    await serving_api.shutdown()


@app.post("/features", response_model=FeatureResponse)
async def get_features(request: FeatureRequest):
    """Get features for a single entity"""
    with api_latency.time():
        try:
            response = await serving_api.serve_features(request)
            api_requests.labels(endpoint="/features", status="success").inc()
            return response
        except Exception as e:
            api_requests.labels(endpoint="/features", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/batch", response_model=List[FeatureResponse])
async def get_batch_features(request: BatchFeatureRequest):
    """Get features for multiple entities"""
    with api_latency.time():
        try:
            responses = await serving_api.serve_batch_features(request)
            api_requests.labels(endpoint="/features/batch", status="success").inc()
            return responses
        except Exception as e:
            api_requests.labels(endpoint="/features/batch", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/catalog")
async def get_feature_catalog(search: Optional[str] = None, tag: Optional[str] = None):
    """Get searchable feature catalog"""
    try:
        catalog = serving_api.feature_store.create_feature_catalog()
        
        # Filter by search term
        if search:
            search_lower = search.lower()
            matching_features = set()
            
            for term, features in catalog['search_index'].items():
                if search_lower in term:
                    matching_features.update(features)
            
            catalog['features'] = [
                f for f in catalog['features'] 
                if f['name'] in matching_features
            ]
        
        # Filter by tag
        if tag and tag in catalog['categories']:
            tag_features = catalog['categories'][tag]
            catalog['features'] = [
                f for f in catalog['features']
                if f['name'] in tag_features
            ]
        
        api_requests.labels(endpoint="/features/catalog", status="success").inc()
        return catalog
    except Exception as e:
        api_requests.labels(endpoint="/features/catalog", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/{feature_name}/drift")
async def get_feature_drift(feature_name: str):
    """Get drift metrics for a feature"""
    try:
        drift_metrics = serving_api.feature_store.monitor_drift(feature_name)
        api_requests.labels(endpoint="/features/drift", status="success").inc()
        return {
            'feature_name': feature_name,
            'drift_metrics': drift_metrics,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        api_requests.labels(endpoint="/features/drift", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/{feature_name}/backfill")
async def backfill_feature(
    feature_name: str,
    start_date: datetime,
    end_date: datetime,
    background_tasks: BackgroundTasks
):
    """Trigger feature backfill"""
    try:
        # Run backfill in background
        background_tasks.add_task(
            serving_api.feature_store.backfill_features,
            feature_name,
            start_date,
            end_date
        )
        
        api_requests.labels(endpoint="/features/backfill", status="success").inc()
        return {
            'status': 'backfill_started',
            'feature_name': feature_name,
            'start_date': start_date,
            'end_date': end_date
        }
    except Exception as e:
        api_requests.labels(endpoint="/features/backfill", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        await serving_api.redis.ping()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow(),
            'version': '1.0.0'
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)