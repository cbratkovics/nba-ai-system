"""
Production Feature Store Implementation for NBA Analytics Platform
Supports versioning, lineage tracking, and real-time serving
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from redis import Redis
from confluent_kafka import Producer, Consumer
import dask.dataframe as dd
import pyarrow.parquet as pq
import pyarrow as pa
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import mlflow
from prometheus_client import Counter, Histogram, Gauge
import logging
from concurrent.futures import ThreadPoolExecutor
import boto3

# Metrics
feature_compute_time = Histogram('feature_compute_seconds', 'Time to compute features')
feature_serve_time = Histogram('feature_serve_seconds', 'Time to serve features')
feature_drift_gauge = Gauge('feature_drift_score', 'Feature drift score', ['feature_name'])
feature_requests = Counter('feature_requests_total', 'Total feature requests', ['feature_name'])

Base = declarative_base()
logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Feature metadata and computation logic"""
    name: str
    version: str
    description: str
    compute_function: str
    dependencies: List[str]
    ttl_seconds: int
    data_type: str
    statistics: Dict[str, float]
    created_at: datetime
    updated_at: datetime
    tags: List[str]


@dataclass
class FeatureValue:
    """Individual feature value with metadata"""
    entity_id: str
    feature_name: str
    value: Any
    timestamp: datetime
    version: str
    
    
class FeatureRegistry(Base):
    """SQLAlchemy model for feature registry"""
    __tablename__ = 'feature_registry'
    
    name = Column(String, primary_key=True)
    version = Column(String, primary_key=True)
    definition = Column(JSON)
    lineage = Column(JSON)
    statistics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeatureStore:
    """Production-grade feature store with versioning and real-time serving"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize storage backends
        self.redis_client = Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True,
            connection_pool_kwargs={'max_connections': 100}
        )
        
        # S3 for offline storage
        self.s3_client = boto3.client('s3')
        self.bucket_name = config['s3_bucket']
        
        # PostgreSQL for metadata
        self.engine = create_engine(config['postgres_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Kafka for streaming
        self.kafka_producer = Producer({
            'bootstrap.servers': config['kafka_brokers'],
            'compression.type': 'snappy',
            'batch.size': 16384,
            'linger.ms': 10
        })
        
        # Dask for distributed compute
        self.dask_client = dd.from_pandas(pd.DataFrame(), npartitions=1)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=20)
        
    def register_feature(self, feature_def: FeatureDefinition) -> None:
        """Register a new feature with versioning"""
        with self.Session() as session:
            # Check if feature exists
            existing = session.query(FeatureRegistry).filter_by(
                name=feature_def.name,
                version=feature_def.version
            ).first()
            
            if existing:
                raise ValueError(f"Feature {feature_def.name} v{feature_def.version} already exists")
            
            # Create lineage tracking
            lineage = {
                'dependencies': feature_def.dependencies,
                'compute_function': feature_def.compute_function,
                'created_by': 'system',
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Register feature
            registry_entry = FeatureRegistry(
                name=feature_def.name,
                version=feature_def.version,
                definition=asdict(feature_def),
                lineage=lineage,
                statistics=feature_def.statistics
            )
            
            session.add(registry_entry)
            session.commit()
            
            # Log to MLflow
            mlflow.log_params({
                f"feature_{feature_def.name}_version": feature_def.version,
                f"feature_{feature_def.name}_type": feature_def.data_type
            })
            
            logger.info(f"Registered feature {feature_def.name} v{feature_def.version}")
    
    async def compute_features(self, entity_ids: List[str], feature_names: List[str],
                             compute_context: Dict[str, Any]) -> pd.DataFrame:
        """Compute features with Spark/Dask for scalability"""
        feature_compute_time.time()
        
        results = []
        
        for feature_name in feature_names:
            # Get feature definition
            feature_def = self._get_feature_definition(feature_name)
            
            # Check cache first
            cached_values = await self._get_cached_features(entity_ids, feature_name)
            
            # Identify entities needing computation
            missing_entities = [eid for eid in entity_ids if eid not in cached_values]
            
            if missing_entities:
                # Compute missing features
                computed_values = await self._compute_feature_batch(
                    missing_entities,
                    feature_def,
                    compute_context
                )
                
                # Cache computed values
                await self._cache_features(computed_values, feature_def)
                
                # Merge with cached
                cached_values.update(computed_values)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([
                {
                    'entity_id': eid,
                    'feature_name': feature_name,
                    'value': cached_values.get(eid),
                    'timestamp': datetime.utcnow()
                }
                for eid in entity_ids
            ])
            
            results.append(feature_df)
        
        # Combine all features
        result_df = pd.concat(results, ignore_index=True)
        
        # Track metrics
        for feature_name in feature_names:
            feature_requests.labels(feature_name=feature_name).inc(len(entity_ids))
        
        return result_df
    
    async def _compute_feature_batch(self, entity_ids: List[str], 
                                   feature_def: FeatureDefinition,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features using Dask for distributed processing"""
        # Load data for entities
        data = await self._load_entity_data(entity_ids)
        
        # Convert to Dask DataFrame for distributed compute
        ddf = dd.from_pandas(data, npartitions=max(1, len(data) // 1000))
        
        # Parse and execute compute function
        compute_func = self._parse_compute_function(feature_def.compute_function)
        
        # Apply computation
        result = ddf.map_partitions(
            lambda df: df.apply(lambda row: compute_func(row, context), axis=1)
        ).compute()
        
        # Convert to dict
        return dict(zip(entity_ids, result.values))
    
    async def serve_features(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Real-time feature serving API"""
        with feature_serve_time.time():
            entity_id = request['entity_id']
            feature_names = request['features']
            
            features = {}
            
            # Try online store first
            for feature_name in feature_names:
                cache_key = f"feature:{feature_name}:{entity_id}"
                cached_value = self.redis_client.get(cache_key)
                
                if cached_value:
                    features[feature_name] = json.loads(cached_value)
                else:
                    # Fallback to computation
                    computed = await self.compute_features(
                        [entity_id],
                        [feature_name],
                        request.get('context', {})
                    )
                    
                    if not computed.empty:
                        value = computed.iloc[0]['value']
                        features[feature_name] = value
                        
                        # Cache for future
                        self.redis_client.setex(
                            cache_key,
                            300,  # 5 minute TTL
                            json.dumps(value)
                        )
            
            return {
                'entity_id': entity_id,
                'features': features,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def backfill_features(self, feature_name: str, start_date: datetime, 
                         end_date: datetime) -> None:
        """Backfill historical features"""
        logger.info(f"Starting backfill for {feature_name} from {start_date} to {end_date}")
        
        # Get all entities
        entities = self._get_all_entities()
        
        # Process in daily batches
        current_date = start_date
        while current_date <= end_date:
            batch_context = {
                'date': current_date,
                'is_backfill': True
            }
            
            # Compute features for this date
            asyncio.run(self.compute_features(
                entities,
                [feature_name],
                batch_context
            ))
            
            # Store to offline storage
            self._store_offline_features(feature_name, current_date)
            
            current_date += timedelta(days=1)
            
        logger.info(f"Completed backfill for {feature_name}")
    
    def forward_fill_features(self, feature_name: str, entity_ids: List[str]) -> None:
        """Forward fill missing feature values"""
        # Get historical values
        historical = self._get_historical_features(feature_name, entity_ids)
        
        # Apply forward fill logic
        filled_values = historical.fillna(method='ffill')
        
        # Update storage
        self._update_feature_values(feature_name, filled_values)
    
    def monitor_drift(self, feature_name: str) -> Dict[str, float]:
        """Monitor feature drift using statistical tests"""
        # Get recent feature values
        recent_values = self._get_recent_feature_values(feature_name, hours=24)
        baseline_values = self._get_baseline_feature_values(feature_name)
        
        # Calculate drift metrics
        drift_metrics = {
            'psi': self._calculate_psi(baseline_values, recent_values),
            'kl_divergence': self._calculate_kl_divergence(baseline_values, recent_values),
            'wasserstein': self._calculate_wasserstein_distance(baseline_values, recent_values),
            'mean_shift': abs(recent_values.mean() - baseline_values.mean()),
            'std_ratio': recent_values.std() / baseline_values.std()
        }
        
        # Update Prometheus metrics
        feature_drift_gauge.labels(feature_name=feature_name).set(drift_metrics['psi'])
        
        # Alert if drift exceeds threshold
        if drift_metrics['psi'] > 0.1:
            self._send_drift_alert(feature_name, drift_metrics)
        
        return drift_metrics
    
    def create_feature_catalog(self) -> Dict[str, Any]:
        """Create searchable feature catalog"""
        with self.Session() as session:
            all_features = session.query(FeatureRegistry).all()
            
            catalog = {
                'features': [],
                'total_count': len(all_features),
                'categories': {},
                'search_index': {}
            }
            
            for feature in all_features:
                feature_info = {
                    'name': feature.name,
                    'version': feature.version,
                    'description': feature.definition.get('description'),
                    'tags': feature.definition.get('tags', []),
                    'statistics': feature.statistics,
                    'created_at': feature.created_at.isoformat(),
                    'dependencies': feature.lineage.get('dependencies', [])
                }
                
                catalog['features'].append(feature_info)
                
                # Build search index
                search_terms = [
                    feature.name.lower(),
                    feature.definition.get('description', '').lower()
                ] + [tag.lower() for tag in feature.definition.get('tags', [])]
                
                for term in search_terms:
                    if term not in catalog['search_index']:
                        catalog['search_index'][term] = []
                    catalog['search_index'][term].append(feature.name)
                
                # Categorize by tags
                for tag in feature.definition.get('tags', []):
                    if tag not in catalog['categories']:
                        catalog['categories'][tag] = []
                    catalog['categories'][tag].append(feature.name)
            
            return catalog
    
    def _calculate_psi(self, baseline: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index"""
        # Bin the data
        _, bin_edges = np.histogram(baseline, bins=10)
        baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
        current_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize
        baseline_hist = baseline_hist / len(baseline)
        current_hist = current_hist / len(current)
        
        # Calculate PSI
        psi = 0
        for i in range(len(baseline_hist)):
            if baseline_hist[i] > 0 and current_hist[i] > 0:
                psi += (current_hist[i] - baseline_hist[i]) * np.log(current_hist[i] / baseline_hist[i])
        
        return psi
    
    def _store_offline_features(self, feature_name: str, date: datetime) -> None:
        """Store features to S3 for offline access"""
        # Get computed features for date
        features_df = self._get_features_for_date(feature_name, date)
        
        # Convert to Parquet
        table = pa.Table.from_pandas(features_df)
        
        # S3 path with partitioning
        s3_path = f"features/{feature_name}/year={date.year}/month={date.month}/day={date.day}/data.parquet"
        
        # Write to S3
        pq.write_table(table, f"s3://{self.bucket_name}/{s3_path}")
        
        logger.info(f"Stored {len(features_df)} features to {s3_path}")


# Example usage
if __name__ == "__main__":
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        's3_bucket': 'nba-features',
        'postgres_url': 'postgresql://user:pass@localhost/features',
        'kafka_brokers': 'localhost:9092'
    }
    
    feature_store = FeatureStore(config)
    
    # Register a feature
    player_hot_streak = FeatureDefinition(
        name='player_hot_streak',
        version='1.0.0',
        description='Whether player is on a scoring hot streak',
        compute_function='lambda row, ctx: row["avg_points_last_5"] > row["season_avg_points"] * 1.2',
        dependencies=['avg_points_last_5', 'season_avg_points'],
        ttl_seconds=3600,
        data_type='boolean',
        statistics={'true_rate': 0.23},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        tags=['player', 'performance', 'streak']
    )
    
    feature_store.register_feature(player_hot_streak)