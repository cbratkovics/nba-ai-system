"""
Production Model Serving Platform with TensorFlow Serving and Advanced Features
Includes blue-green deployments, canary testing, and GPU optimization
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, model_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, model_service_pb2_grpc
import grpc
import redis
import mlflow
from prometheus_client import Counter, Histogram, Gauge
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import torch
import tritonclient.grpc as grpcclient
from kubernetes import client, config
import yaml

# Metrics
model_predictions = Counter('model_predictions_total', 'Total predictions', ['model', 'version'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', ['model'])
model_errors = Counter('model_errors_total', 'Model errors', ['model', 'error_type'])
cache_performance = Counter('model_cache_performance', 'Cache hits/misses', ['status'])
active_model_version = Gauge('active_model_version', 'Currently active model version', ['model'])

logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_name: str
    version: int
    created_at: datetime
    metrics: Dict[str, float]
    status: str  # 'staging', 'production', 'retired'
    traffic_percentage: float
    tags: Dict[str, str]


@dataclass
class PredictionRequest:
    """Structured prediction request"""
    entity_id: str
    features: Dict[str, Any]
    model_name: str
    version: Optional[int] = None
    explain: bool = False


class ModelServingPlatform:
    """Enterprise-grade model serving with advanced deployment strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # TensorFlow Serving connection
        self.tf_serving_channel = grpc.insecure_channel(
            f"{config['tf_serving_host']}:{config['tf_serving_port']}"
        )
        self.prediction_stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.tf_serving_channel
        )
        self.model_stub = model_service_pb2_grpc.ModelServiceStub(
            self.tf_serving_channel
        )
        
        # Triton Inference Server for multi-framework support
        self.triton_client = grpcclient.InferenceServerClient(
            url=f"{config['triton_host']}:{config['triton_port']}"
        )
        
        # Redis for caching
        self.redis_client = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        
        # MLflow for model registry
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        
        # Kubernetes client for deployments
        config.load_incluster_config()
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Model version routing
        self.model_routes = {}
        self._load_model_routes()
    
    def _load_model_routes(self):
        """Load model routing configuration"""
        routes_config = self.redis_client.get("model_routes")
        if routes_config:
            self.model_routes = json.loads(routes_config)
        else:
            # Default routes
            self.model_routes = {
                "nba_points_predictor": {
                    "production": 3,
                    "canary": 4,
                    "traffic_split": {"production": 0.9, "canary": 0.1}
                }
            }
    
    async def deploy_model(self, model_name: str, version: int, 
                          strategy: DeploymentStrategy) -> Dict[str, Any]:
        """Deploy model with specified strategy"""
        logger.info(f"Deploying {model_name} v{version} with {strategy.value} strategy")
        
        if strategy == DeploymentStrategy.BLUE_GREEN:
            return await self._blue_green_deployment(model_name, version)
        elif strategy == DeploymentStrategy.CANARY:
            return await self._canary_deployment(model_name, version)
        elif strategy == DeploymentStrategy.SHADOW:
            return await self._shadow_deployment(model_name, version)
        else:
            raise ValueError(f"Unsupported deployment strategy: {strategy}")
    
    async def _blue_green_deployment(self, model_name: str, new_version: int) -> Dict[str, Any]:
        """Blue-green deployment with instant switchover"""
        deployment_id = f"{model_name}-deployment-{int(time.time())}"
        
        try:
            # 1. Deploy new version (green) alongside current (blue)
            green_deployment = self._create_model_deployment(
                model_name, new_version, f"{model_name}-green"
            )
            
            # 2. Wait for green deployment to be ready
            await self._wait_for_deployment_ready(f"{model_name}-green")
            
            # 3. Run smoke tests on green
            smoke_test_results = await self._run_smoke_tests(model_name, new_version)
            
            if not smoke_test_results['passed']:
                raise Exception(f"Smoke tests failed: {smoke_test_results['errors']}")
            
            # 4. Switch traffic to green
            self._update_service_selector(model_name, f"{model_name}-green")
            
            # 5. Monitor for 5 minutes
            await asyncio.sleep(300)
            metrics = await self._collect_deployment_metrics(model_name, new_version)
            
            if metrics['error_rate'] > 0.01:  # 1% error threshold
                # Rollback
                self._update_service_selector(model_name, f"{model_name}-blue")
                raise Exception(f"High error rate detected: {metrics['error_rate']}")
            
            # 6. Remove blue deployment
            self._delete_deployment(f"{model_name}-blue")
            
            # 7. Rename green to blue for next deployment
            self._rename_deployment(f"{model_name}-green", f"{model_name}-blue")
            
            return {
                'deployment_id': deployment_id,
                'status': 'success',
                'metrics': metrics,
                'strategy': 'blue_green'
            }
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            # Ensure we're on blue
            self._update_service_selector(model_name, f"{model_name}-blue")
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _canary_deployment(self, model_name: str, new_version: int,
                                initial_traffic: float = 0.1) -> Dict[str, Any]:
        """Canary deployment with gradual traffic shift"""
        deployment_id = f"{model_name}-canary-{int(time.time())}"
        
        try:
            # 1. Deploy canary version
            canary_deployment = self._create_model_deployment(
                model_name, new_version, f"{model_name}-canary"
            )
            
            await self._wait_for_deployment_ready(f"{model_name}-canary")
            
            # 2. Configure initial traffic split
            current_version = self.model_routes[model_name]['production']
            self._update_traffic_split(model_name, {
                'production': 1 - initial_traffic,
                'canary': initial_traffic
            })
            
            # 3. Gradual traffic increase
            traffic_steps = [0.1, 0.25, 0.5, 0.75, 1.0]
            
            for traffic_percentage in traffic_steps:
                # Update traffic split
                self._update_traffic_split(model_name, {
                    'production': 1 - traffic_percentage,
                    'canary': traffic_percentage
                })
                
                # Monitor for 10 minutes at each step
                await asyncio.sleep(600)
                
                # Check metrics
                canary_metrics = await self._collect_deployment_metrics(
                    model_name, new_version, deployment_type='canary'
                )
                prod_metrics = await self._collect_deployment_metrics(
                    model_name, current_version, deployment_type='production'
                )
                
                # Compare performance
                if self._should_rollback_canary(canary_metrics, prod_metrics):
                    logger.warning(f"Canary metrics degraded at {traffic_percentage}%, rolling back")
                    self._update_traffic_split(model_name, {
                        'production': 1.0,
                        'canary': 0.0
                    })
                    self._delete_deployment(f"{model_name}-canary")
                    return {
                        'deployment_id': deployment_id,
                        'status': 'rolled_back',
                        'stopped_at_traffic': traffic_percentage,
                        'reason': 'metrics_degradation'
                    }
            
            # 4. Finalize deployment
            self._finalize_canary_deployment(model_name, new_version)
            
            return {
                'deployment_id': deployment_id,
                'status': 'success',
                'final_metrics': canary_metrics
            }
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            # Rollback to 100% production
            self._update_traffic_split(model_name, {'production': 1.0, 'canary': 0.0})
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _shadow_deployment(self, model_name: str, new_version: int) -> Dict[str, Any]:
        """Shadow deployment for safe testing"""
        deployment_id = f"{model_name}-shadow-{int(time.time())}"
        
        # Deploy shadow version
        shadow_deployment = self._create_model_deployment(
            model_name, new_version, f"{model_name}-shadow"
        )
        
        # Configure shadow routing (duplicate traffic, don't return results)
        self.model_routes[model_name]['shadow'] = new_version
        self._save_model_routes()
        
        # Run for specified duration
        shadow_duration = timedelta(hours=24)
        start_time = datetime.utcnow()
        
        logger.info(f"Shadow deployment started for {model_name} v{new_version}")
        
        return {
            'deployment_id': deployment_id,
            'status': 'shadow_active',
            'start_time': start_time.isoformat(),
            'end_time': (start_time + shadow_duration).isoformat()
        }
    
    async def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Make prediction with smart routing and caching"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(request)
        cached_result = self._get_cached_prediction(cache_key)
        
        if cached_result:
            cache_performance.labels(status='hit').inc()
            return cached_result
        
        cache_performance.labels(status='miss').inc()
        
        # Determine which version to use
        version = self._route_request(request.model_name)
        
        try:
            # Prepare request based on serving framework
            if self._is_tensorflow_model(request.model_name):
                result = await self._predict_tensorflow(request, version)
            else:
                result = await self._predict_triton(request, version)
            
            # Shadow prediction if configured
            if 'shadow' in self.model_routes.get(request.model_name, {}):
                asyncio.create_task(self._shadow_predict(request))
            
            # Cache result
            self._cache_prediction(cache_key, result)
            
            # Track metrics
            prediction_latency.labels(model=request.model_name).observe(time.time() - start_time)
            model_predictions.labels(model=request.model_name, version=version).inc()
            
            return result
            
        except Exception as e:
            model_errors.labels(model=request.model_name, error_type=type(e).__name__).inc()
            logger.error(f"Prediction failed: {e}")
            
            # Try fallback model
            if self.config.get('enable_fallback'):
                return await self._fallback_predict(request)
            raise
    
    async def _predict_tensorflow(self, request: PredictionRequest, version: int) -> Dict[str, Any]:
        """TensorFlow Serving prediction"""
        # Convert features to tensor
        feature_tensor = self._features_to_tensor(request.features)
        
        # Create TF Serving request
        tf_request = predict_pb2.PredictRequest()
        tf_request.model_spec.name = request.model_name
        tf_request.model_spec.version.value = version
        tf_request.inputs['input'].CopyFrom(feature_tensor)
        
        # Make prediction
        result = self.prediction_stub.Predict(tf_request, timeout=10.0)
        
        # Parse response
        predictions = tf.make_ndarray(result.outputs['output'])
        
        response = {
            'entity_id': request.entity_id,
            'prediction': float(predictions[0]),
            'model_name': request.model_name,
            'model_version': version,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add explanation if requested
        if request.explain:
            explanation = await self._get_prediction_explanation(request, version)
            response['explanation'] = explanation
        
        return response
    
    async def _predict_triton(self, request: PredictionRequest, version: int) -> Dict[str, Any]:
        """Triton Inference Server prediction"""
        # Prepare input
        input_data = np.array([list(request.features.values())], dtype=np.float32)
        
        inputs = [grpcclient.InferInput('input', input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = [grpcclient.InferRequestedOutput('output')]
        
        # Make prediction
        response = self.triton_client.infer(
            model_name=request.model_name,
            model_version=str(version),
            inputs=inputs,
            outputs=outputs
        )
        
        prediction = response.as_numpy('output')[0]
        
        return {
            'entity_id': request.entity_id,
            'prediction': float(prediction),
            'model_name': request.model_name,
            'model_version': version,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def enable_gpu_optimization(self, model_name: str) -> None:
        """Enable GPU-specific optimizations"""
        # Configure TensorRT optimization
        config_update = {
            'optimization': {
                'gpu_execution_accelerator': {
                    'name': 'tensorrt',
                    'parameters': {
                        'precision_mode': 'FP16',
                        'max_workspace_size_bytes': '1073741824',
                        'minimum_segment_size': '3'
                    }
                }
            }
        }
        
        # Update model config
        self._update_model_config(model_name, config_update)
        
        # Enable dynamic batching
        batch_config = {
            'dynamic_batching': {
                'max_batch_size': 32,
                'batch_timeout_micros': 1000,
                'max_enqueued_batches': 10
            }
        }
        
        self._update_model_config(model_name, batch_config)
    
    def implement_request_batching(self, model_name: str, 
                                  batch_size: int = 32,
                                  timeout_ms: int = 10) -> None:
        """Configure intelligent request batching"""
        self.batching_config = {
            model_name: {
                'batch_size': batch_size,
                'timeout_ms': timeout_ms,
                'queue': asyncio.Queue(maxsize=1000),
                'processor_task': None
            }
        }
        
        # Start batch processor
        processor_task = asyncio.create_task(
            self._batch_processor(model_name)
        )
        self.batching_config[model_name]['processor_task'] = processor_task
    
    async def _batch_processor(self, model_name: str) -> None:
        """Process batched requests"""
        config = self.batching_config[model_name]
        pending_requests = []
        
        while True:
            try:
                # Collect requests up to batch size or timeout
                deadline = time.time() + (config['timeout_ms'] / 1000)
                
                while len(pending_requests) < config['batch_size'] and time.time() < deadline:
                    try:
                        timeout = deadline - time.time()
                        request = await asyncio.wait_for(
                            config['queue'].get(),
                            timeout=max(0, timeout)
                        )
                        pending_requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if pending_requests:
                    # Process batch
                    batch_result = await self._process_batch(model_name, pending_requests)
                    
                    # Return results to callers
                    for request, result in zip(pending_requests, batch_result):
                        request['future'].set_result(result)
                    
                    pending_requests = []
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                # Return errors to pending requests
                for request in pending_requests:
                    request['future'].set_exception(e)
                pending_requests = []
    
    def add_prediction_caching(self, ttl_seconds: int = 300) -> None:
        """Configure prediction caching layer"""
        self.cache_config = {
            'ttl': ttl_seconds,
            'max_size': 10000,
            'eviction_policy': 'lru'
        }
    
    def _get_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for prediction"""
        feature_str = json.dumps(request.features, sort_keys=True)
        key_parts = [
            request.model_name,
            str(request.version) if request.version else 'latest',
            request.entity_id,
            feature_str
        ]
        
        return hashlib.md5(':'.join(key_parts).encode()).hexdigest()
    
    def _should_rollback_canary(self, canary_metrics: Dict[str, float],
                               prod_metrics: Dict[str, float]) -> bool:
        """Determine if canary should be rolled back"""
        # Check error rate
        if canary_metrics['error_rate'] > prod_metrics['error_rate'] * 1.5:
            return True
        
        # Check latency
        if canary_metrics['p99_latency'] > prod_metrics['p99_latency'] * 1.2:
            return True
        
        # Check business metrics
        if canary_metrics.get('business_metric', 1.0) < prod_metrics.get('business_metric', 1.0) * 0.95:
            return True
        
        return False
    
    async def _collect_deployment_metrics(self, model_name: str, version: int,
                                        deployment_type: str = 'production') -> Dict[str, float]:
        """Collect comprehensive deployment metrics"""
        # Query Prometheus for metrics
        metrics = {
            'error_rate': self._query_error_rate(model_name, version),
            'p50_latency': self._query_latency_percentile(model_name, version, 0.5),
            'p99_latency': self._query_latency_percentile(model_name, version, 0.99),
            'throughput': self._query_throughput(model_name, version),
            'cpu_usage': self._query_resource_usage(model_name, 'cpu'),
            'memory_usage': self._query_resource_usage(model_name, 'memory'),
            'gpu_utilization': self._query_gpu_utilization(model_name)
        }
        
        # Add business metrics
        business_metrics = await self._collect_business_metrics(model_name, version)
        metrics.update(business_metrics)
        
        return metrics
    
    def add_model_explainability(self, model_name: str) -> None:
        """Add explainability API for model predictions"""
        self.explainers = {
            model_name: {
                'shap': self._init_shap_explainer(model_name),
                'lime': self._init_lime_explainer(model_name),
                'integrated_gradients': self._init_ig_explainer(model_name)
            }
        }
    
    async def _get_prediction_explanation(self, request: PredictionRequest,
                                        version: int) -> Dict[str, Any]:
        """Generate prediction explanation"""
        explainer = self.explainers[request.model_name]['shap']
        
        # Get SHAP values
        feature_array = np.array([list(request.features.values())])
        shap_values = explainer.shap_values(feature_array)
        
        # Create explanation
        feature_names = list(request.features.keys())
        explanation = {
            'method': 'shap',
            'feature_importance': dict(zip(feature_names, shap_values[0])),
            'base_value': float(explainer.expected_value),
            'prediction_impact': {
                name: float(value) for name, value in 
                zip(feature_names, shap_values[0])
            }
        }
        
        return explanation


# Kubernetes deployment helpers
def create_model_deployment_yaml(model_name: str, version: int, 
                               deployment_name: str) -> Dict[str, Any]:
    """Create Kubernetes deployment specification"""
    return {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': deployment_name,
            'labels': {
                'app': model_name,
                'version': str(version),
                'deployment': deployment_name
            }
        },
        'spec': {
            'replicas': 3,
            'selector': {
                'matchLabels': {
                    'app': model_name,
                    'deployment': deployment_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': model_name,
                        'version': str(version),
                        'deployment': deployment_name
                    }
                },
                'spec': {
                    'containers': [{
                        'name': 'model-server',
                        'image': f'tensorflow/serving:{version}',
                        'ports': [{'containerPort': 8501}],
                        'env': [
                            {'name': 'MODEL_NAME', 'value': model_name},
                            {'name': 'MODEL_VERSION', 'value': str(version)}
                        ],
                        'resources': {
                            'requests': {'cpu': '2', 'memory': '4Gi', 'nvidia.com/gpu': '1'},
                            'limits': {'cpu': '4', 'memory': '8Gi', 'nvidia.com/gpu': '1'}
                        },
                        'livenessProbe': {
                            'httpGet': {'path': '/v1/models/' + model_name, 'port': 8501},
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10
                        },
                        'readinessProbe': {
                            'httpGet': {'path': '/v1/models/' + model_name, 'port': 8501},
                            'initialDelaySeconds': 10,
                            'periodSeconds': 5
                        }
                    }]
                }
            }
        }
    }


if __name__ == "__main__":
    config = {
        'tf_serving_host': 'localhost',
        'tf_serving_port': 8501,
        'triton_host': 'localhost',
        'triton_port': 8001,
        'redis_host': 'localhost',
        'redis_port': 6379,
        'mlflow_tracking_uri': 'http://localhost:5000',
        'enable_fallback': True
    }
    
    platform = ModelServingPlatform(config)
    
    # Example deployment
    asyncio.run(platform.deploy_model(
        'nba_points_predictor',
        version=4,
        strategy=DeploymentStrategy.CANARY
    ))