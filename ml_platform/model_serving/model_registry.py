"""
Model Registry with Automatic Deployment and Version Management
Integrates with MLflow and supports automated promotion pipelines
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import boto3
from prometheus_client import Counter, Gauge
import logging
import yaml
from concurrent.futures import ThreadPoolExecutor
import hashlib
import requests

# Metrics
model_registrations = Counter('model_registrations_total', 'Total model registrations')
model_promotions = Counter('model_promotions_total', 'Model promotions', ['from_stage', 'to_stage'])
model_versions_gauge = Gauge('model_versions_active', 'Active model versions', ['model_name', 'stage'])
deployment_success = Counter('model_deployment_success_total', 'Successful deployments')
deployment_failures = Counter('model_deployment_failures_total', 'Failed deployments')

Base = declarative_base()
logger = logging.getLogger(__name__)


class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class PromotionStrategy(Enum):
    MANUAL = "manual"
    AUTOMATED_THRESHOLD = "automated_threshold"
    AUTOMATED_SCHEDULE = "automated_schedule"
    CHAMPION_CHALLENGER = "champion_challenger"


@dataclass
class ModelMetadata:
    """Extended model metadata"""
    model_name: str
    version: int
    stage: ModelStage
    metrics: Dict[str, float]
    tags: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: str
    dependencies: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    training_data_version: str
    feature_importance: Dict[str, float]
    
    
@dataclass
class PromotionCriteria:
    """Criteria for automated model promotion"""
    min_accuracy: float
    max_latency_ms: float
    min_test_coverage: float
    required_tests: List[str]
    comparison_metrics: List[str]
    improvement_threshold: float


class ModelRegistryDB(Base):
    """SQLAlchemy model for registry database"""
    __tablename__ = 'model_registry'
    
    id = Column(String, primary_key=True)
    model_name = Column(String)
    version = Column(Integer)
    stage = Column(String)
    metadata = Column(JSON)
    metrics = Column(JSON)
    deployment_config = Column(JSON)
    promotion_history = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelRegistry:
    """Production model registry with automated deployment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # MLflow client
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        self.mlflow_client = MlflowClient()
        
        # Database for extended metadata
        self.engine = create_engine(config['postgres_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # S3 for model artifacts
        self.s3_client = boto3.client('s3')
        self.bucket_name = config['model_bucket']
        
        # Deployment automation
        self.deployment_api = config['deployment_api_url']
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Promotion strategies
        self.promotion_strategies = {}
        self._load_promotion_strategies()
    
    def register_model(self, model: Any, metadata: ModelMetadata,
                      training_metrics: Dict[str, float]) -> str:
        """Register a new model version with comprehensive metadata"""
        model_registrations.inc()
        
        # Generate unique model ID
        model_id = self._generate_model_id(metadata)
        
        # Log to MLflow
        with mlflow.start_run() as run:
            # Log model
            signature = infer_signature(
                pd.DataFrame(metadata.input_schema),
                pd.DataFrame(metadata.output_schema)
            )
            
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                registered_model_name=metadata.model_name
            )
            
            # Log metrics
            for metric_name, metric_value in training_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log parameters
            mlflow.log_params({
                'model_version': metadata.version,
                'stage': metadata.stage.value,
                'created_by': metadata.created_by,
                'training_data_version': metadata.training_data_version
            })
            
            # Log tags
            mlflow.set_tags(metadata.tags)
            
            # Log feature importance
            if metadata.feature_importance:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                features = list(metadata.feature_importance.keys())
                importances = list(metadata.feature_importance.values())
                
                ax.barh(features, importances)
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                
                mlflow.log_figure(fig, "feature_importance.png")
                plt.close()
        
        # Store extended metadata in database
        with self.Session() as session:
            registry_entry = ModelRegistryDB(
                id=model_id,
                model_name=metadata.model_name,
                version=metadata.version,
                stage=metadata.stage.value,
                metadata=asdict(metadata),
                metrics=training_metrics,
                deployment_config={},
                promotion_history=[]
            )
            
            session.add(registry_entry)
            session.commit()
        
        # Update metrics
        model_versions_gauge.labels(
            model_name=metadata.model_name,
            stage=metadata.stage.value
        ).inc()
        
        logger.info(f"Registered model {metadata.model_name} v{metadata.version}")
        
        return model_id
    
    async def promote_model(self, model_name: str, from_version: int,
                          to_stage: ModelStage,
                          strategy: PromotionStrategy = PromotionStrategy.MANUAL) -> Dict[str, Any]:
        """Promote model to new stage with validation"""
        logger.info(f"Promoting {model_name} v{from_version} to {to_stage.value}")
        
        # Get current model metadata
        model_metadata = self._get_model_metadata(model_name, from_version)
        
        if strategy == PromotionStrategy.AUTOMATED_THRESHOLD:
            # Check if model meets promotion criteria
            criteria = self.promotion_strategies[model_name]
            validation_result = await self._validate_promotion(
                model_name, from_version, criteria
            )
            
            if not validation_result['passed']:
                return {
                    'status': 'rejected',
                    'reason': validation_result['reason'],
                    'metrics': validation_result['metrics']
                }
        
        elif strategy == PromotionStrategy.CHAMPION_CHALLENGER:
            # Compare with current champion
            comparison_result = await self._compare_with_champion(
                model_name, from_version
            )
            
            if not comparison_result['is_better']:
                return {
                    'status': 'rejected',
                    'reason': 'Did not outperform current champion',
                    'comparison': comparison_result
                }
        
        # Perform promotion
        try:
            # Update MLflow
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=from_version,
                stage=to_stage.value.capitalize()
            )
            
            # Update database
            with self.Session() as session:
                entry = session.query(ModelRegistryDB).filter_by(
                    model_name=model_name,
                    version=from_version
                ).first()
                
                if entry:
                    entry.stage = to_stage.value
                    entry.promotion_history = entry.promotion_history or []
                    entry.promotion_history.append({
                        'from_stage': model_metadata.stage.value,
                        'to_stage': to_stage.value,
                        'timestamp': datetime.utcnow().isoformat(),
                        'strategy': strategy.value
                    })
                    
                    session.commit()
            
            # Trigger deployment if promoting to production
            if to_stage == ModelStage.PRODUCTION:
                deployment_result = await self._trigger_deployment(
                    model_name, from_version
                )
                
                if not deployment_result['success']:
                    # Rollback promotion
                    self.mlflow_client.transition_model_version_stage(
                        name=model_name,
                        version=from_version,
                        stage=model_metadata.stage.value.capitalize()
                    )
                    
                    return {
                        'status': 'deployment_failed',
                        'reason': deployment_result['error']
                    }
            
            # Update metrics
            model_promotions.labels(
                from_stage=model_metadata.stage.value,
                to_stage=to_stage.value
            ).inc()
            
            return {
                'status': 'success',
                'model_name': model_name,
                'version': from_version,
                'new_stage': to_stage.value,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _validate_promotion(self, model_name: str, version: int,
                                criteria: PromotionCriteria) -> Dict[str, Any]:
        """Validate model meets promotion criteria"""
        # Get model metrics
        metrics = self._get_model_metrics(model_name, version)
        
        validation_results = {
            'passed': True,
            'checks': {},
            'metrics': metrics
        }
        
        # Check accuracy
        if metrics.get('accuracy', 0) < criteria.min_accuracy:
            validation_results['passed'] = False
            validation_results['checks']['accuracy'] = {
                'passed': False,
                'required': criteria.min_accuracy,
                'actual': metrics.get('accuracy', 0)
            }
        
        # Check latency
        if metrics.get('inference_latency_ms', float('inf')) > criteria.max_latency_ms:
            validation_results['passed'] = False
            validation_results['checks']['latency'] = {
                'passed': False,
                'required': criteria.max_latency_ms,
                'actual': metrics.get('inference_latency_ms')
            }
        
        # Run required tests
        test_results = await self._run_model_tests(model_name, version, criteria.required_tests)
        
        for test_name, test_result in test_results.items():
            if not test_result['passed']:
                validation_results['passed'] = False
                validation_results['checks'][test_name] = test_result
        
        if not validation_results['passed']:
            validation_results['reason'] = 'Failed validation checks'
        
        return validation_results
    
    async def _compare_with_champion(self, model_name: str, 
                                   challenger_version: int) -> Dict[str, Any]:
        """Compare challenger model with current champion"""
        # Get current production model
        champion_version = self._get_production_version(model_name)
        
        if not champion_version:
            # No current champion, challenger wins by default
            return {
                'is_better': True,
                'reason': 'No current champion'
            }
        
        # Load both models
        champion_metrics = self._get_model_metrics(model_name, champion_version)
        challenger_metrics = self._get_model_metrics(model_name, challenger_version)
        
        # Run A/B test
        ab_test_result = await self._run_ab_test(
            model_name,
            champion_version,
            challenger_version,
            duration_hours=24
        )
        
        comparison = {
            'is_better': False,
            'champion_version': champion_version,
            'challenger_version': challenger_version,
            'metrics_comparison': {},
            'ab_test_results': ab_test_result
        }
        
        # Compare key metrics
        improvement_count = 0
        total_metrics = 0
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in champion_metrics and metric in challenger_metrics:
                total_metrics += 1
                champion_value = champion_metrics[metric]
                challenger_value = challenger_metrics[metric]
                improvement = (challenger_value - champion_value) / champion_value
                
                comparison['metrics_comparison'][metric] = {
                    'champion': champion_value,
                    'challenger': challenger_value,
                    'improvement': improvement
                }
                
                if improvement > 0.01:  # 1% improvement threshold
                    improvement_count += 1
        
        # Check if challenger is better
        if improvement_count >= total_metrics * 0.6:  # 60% of metrics improved
            comparison['is_better'] = True
            comparison['reason'] = f'Improved {improvement_count}/{total_metrics} metrics'
        
        return comparison
    
    async def _trigger_deployment(self, model_name: str, version: int) -> Dict[str, Any]:
        """Trigger automated deployment"""
        deployment_config = self._get_deployment_config(model_name, version)
        
        try:
            # Call deployment API
            response = requests.post(
                f"{self.deployment_api}/deploy",
                json={
                    'model_name': model_name,
                    'version': version,
                    'config': deployment_config,
                    'strategy': 'blue_green'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                deployment_success.inc()
                return {
                    'success': True,
                    'deployment_id': response.json()['deployment_id']
                }
            else:
                deployment_failures.inc()
                return {
                    'success': False,
                    'error': response.text
                }
                
        except Exception as e:
            deployment_failures.inc()
            logger.error(f"Deployment failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_lineage(self, model_name: str, version: int) -> Dict[str, Any]:
        """Get complete model lineage and provenance"""
        with self.Session() as session:
            entry = session.query(ModelRegistryDB).filter_by(
                model_name=model_name,
                version=version
            ).first()
            
            if not entry:
                return {}
            
            # Get MLflow run info
            mlflow_model = self.mlflow_client.get_model_version(
                name=model_name,
                version=str(version)
            )
            
            run = self.mlflow_client.get_run(mlflow_model.run_id)
            
            lineage = {
                'model_name': model_name,
                'version': version,
                'created_at': entry.created_at.isoformat(),
                'created_by': entry.metadata.get('created_by'),
                'training_data': {
                    'version': entry.metadata.get('training_data_version'),
                    'source': run.data.params.get('data_source'),
                    'size': run.data.params.get('data_size')
                },
                'code': {
                    'git_commit': run.data.tags.get('mlflow.source.git.commit'),
                    'source_type': run.data.tags.get('mlflow.source.type'),
                    'source_name': run.data.tags.get('mlflow.source.name')
                },
                'dependencies': entry.metadata.get('dependencies', []),
                'metrics': entry.metrics,
                'promotion_history': entry.promotion_history or [],
                'experiments': {
                    'run_id': mlflow_model.run_id,
                    'experiment_id': run.info.experiment_id,
                    'artifact_uri': run.info.artifact_uri
                }
            }
            
            return lineage
    
    def search_models(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search models with advanced filtering"""
        with self.Session() as session:
            query = session.query(ModelRegistryDB)
            
            # Apply filters
            if 'model_name' in filters:
                query = query.filter(ModelRegistryDB.model_name.like(f"%{filters['model_name']}%"))
            
            if 'stage' in filters:
                query = query.filter(ModelRegistryDB.stage == filters['stage'])
            
            if 'min_accuracy' in filters:
                # This would require a more complex query with JSON operations
                pass
            
            results = []
            for entry in query.all():
                result = {
                    'id': entry.id,
                    'model_name': entry.model_name,
                    'version': entry.version,
                    'stage': entry.stage,
                    'metrics': entry.metrics,
                    'created_at': entry.created_at.isoformat(),
                    'tags': entry.metadata.get('tags', {})
                }
                
                results.append(result)
            
            return results
    
    async def schedule_automated_promotion(self, model_name: str,
                                         schedule: str,
                                         criteria: PromotionCriteria) -> None:
        """Schedule automated model promotion checks"""
        self.promotion_strategies[model_name] = criteria
        
        # Schedule periodic checks (using APScheduler or similar)
        # This is a simplified example
        while True:
            # Get latest staging model
            staging_models = self.search_models({
                'model_name': model_name,
                'stage': ModelStage.STAGING.value
            })
            
            if staging_models:
                latest_staging = max(staging_models, key=lambda x: x['version'])
                
                # Try to promote
                result = await self.promote_model(
                    model_name,
                    latest_staging['version'],
                    ModelStage.PRODUCTION,
                    PromotionStrategy.AUTOMATED_THRESHOLD
                )
                
                logger.info(f"Automated promotion result: {result}")
            
            # Wait for next check (parse schedule)
            await asyncio.sleep(3600)  # Hourly check
    
    def _generate_model_id(self, metadata: ModelMetadata) -> str:
        """Generate unique model ID"""
        id_parts = [
            metadata.model_name,
            str(metadata.version),
            metadata.created_at.isoformat()
        ]
        
        return hashlib.sha256(':'.join(id_parts).encode()).hexdigest()[:16]
    
    def _get_deployment_config(self, model_name: str, version: int) -> Dict[str, Any]:
        """Get deployment configuration for model"""
        # Default configuration
        config = {
            'replicas': 3,
            'cpu': '2',
            'memory': '4Gi',
            'gpu': '1' if 'deep' in model_name else '0',
            'autoscaling': {
                'enabled': True,
                'min_replicas': 2,
                'max_replicas': 10,
                'target_cpu': 70
            }
        }
        
        # Load custom config if exists
        with self.Session() as session:
            entry = session.query(ModelRegistryDB).filter_by(
                model_name=model_name,
                version=version
            ).first()
            
            if entry and entry.deployment_config:
                config.update(entry.deployment_config)
        
        return config


# Example usage
if __name__ == "__main__":
    config = {
        'mlflow_tracking_uri': 'http://localhost:5000',
        'postgres_url': 'postgresql://user:pass@localhost/models',
        'model_bucket': 'nba-models',
        'deployment_api_url': 'http://localhost:8080'
    }
    
    registry = ModelRegistry(config)
    
    # Register a model
    metadata = ModelMetadata(
        model_name='nba_points_predictor',
        version=5,
        stage=ModelStage.STAGING,
        metrics={'accuracy': 0.946, 'mae': 1.2},
        tags={'algorithm': 'random_forest', 'framework': 'sklearn'},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by='data_scientist_1',
        description='Random Forest model for NBA points prediction',
        dependencies=['scikit-learn==1.0.2', 'numpy==1.21.0'],
        input_schema={'type': 'object', 'properties': {}},
        output_schema={'type': 'number'},
        training_data_version='v2024.01.15',
        feature_importance={'rest_days': 0.25, 'avg_points_last_5': 0.35}
    )
    
    # Example model (placeholder)
    import joblib
    model = joblib.load('model.pkl')  # Load your actual model
    
    model_id = registry.register_model(
        model,
        metadata,
        training_metrics={'accuracy': 0.946, 'mae': 1.2}
    )