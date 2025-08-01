"""
Production ML Monitoring and Experimentation Framework
Comprehensive model observability with drift detection, A/B testing, and alerting
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import logging
import redis
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import mlflow
from mlflow.tracking import MlflowClient
import requests
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import boto3
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Custom Metrics Registry
monitoring_registry = CollectorRegistry()

# Prometheus Metrics
model_predictions_total = Counter(
    'ml_model_predictions_total',
    'Total model predictions',
    ['model_name', 'version', 'environment'],
    registry=monitoring_registry
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Model prediction latency',
    ['model_name', 'version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=monitoring_registry
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Model accuracy metrics',
    ['model_name', 'version', 'metric_type'],
    registry=monitoring_registry
)

data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift detection score',
    ['feature_name', 'drift_type'],
    registry=monitoring_registry
)

prediction_drift_score = Gauge(
    'ml_prediction_drift_score',
    'Prediction drift score',
    ['model_name'],
    registry=monitoring_registry
)

ab_test_metrics = Gauge(
    'ml_ab_test_metrics',
    'A/B test performance metrics',
    ['experiment_id', 'variant', 'metric'],
    registry=monitoring_registry
)

feature_importance_drift = Gauge(
    'ml_feature_importance_drift',
    'Feature importance drift score',
    ['model_name', 'feature_name'],
    registry=monitoring_registry
)

Base = declarative_base()
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    version: str
    mae: float
    rmse: float
    r2: float
    mape: float
    prediction_count: int
    timestamp: datetime


@dataclass
class DriftMetrics:
    """Data/prediction drift metrics"""
    feature_name: str
    drift_type: str
    drift_score: float
    p_value: float
    threshold: float
    is_drift: bool
    timestamp: datetime


class ExperimentDB(Base):
    """A/B test experiment database model"""
    __tablename__ = 'ab_experiments'
    
    experiment_id = Column(String, primary_key=True)
    model_a = Column(String)
    model_b = Column(String)
    traffic_split = Column(Float)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    status = Column(String)  # 'running', 'completed', 'stopped'
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLogDB(Base):
    """Prediction logging database model"""
    __tablename__ = 'prediction_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String)
    version = Column(String)
    prediction = Column(Float)
    actual = Column(Float, nullable=True)
    features = Column(JSON)
    latency_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    experiment_id = Column(String, nullable=True)
    variant = Column(String, nullable=True)


class MLMonitoringSystem:
    """Comprehensive ML monitoring and experimentation system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Database setup
        self.engine = create_engine(config['postgres_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis for caching and state
        self.redis_client = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        
        # MLflow client
        self.mlflow_client = MlflowClient(tracking_uri=config['mlflow_uri'])
        
        # Drift detection
        self.drift_detectors = self._initialize_drift_detectors()
        
        # Alert manager
        self.alert_manager = AlertManager(config)
        
        # A/B test manager
        self.ab_test_manager = ABTestManager(config, self.Session)
        
        # Dashboard app
        self.dash_app = self._create_dashboard()
        
        # Start Prometheus metrics server
        start_http_server(config.get('metrics_port', 8000), registry=monitoring_registry)
        
        # Background tasks
        self.monitoring_tasks = []
        
    def _initialize_drift_detectors(self) -> Dict[str, Any]:
        """Initialize various drift detection methods"""
        return {
            'psi': PopulationStabilityIndex(threshold=0.1),
            'ks_test': KolmogorovSmirnovTest(threshold=0.05),
            'wasserstein': WassersteinDistance(threshold=0.1),
            'feature_correlation': FeatureCorrelationDrift(threshold=0.1)
        }
    
    async def log_prediction(self, model_name: str, version: str,
                           prediction: float, features: Dict[str, Any],
                           latency_ms: float, actual: Optional[float] = None,
                           experiment_id: Optional[str] = None,
                           variant: Optional[str] = None) -> None:
        """Log model prediction with metadata"""
        
        # Update Prometheus metrics
        model_predictions_total.labels(
            model_name=model_name,
            version=version,
            environment=self.config['environment']
        ).inc()
        
        prediction_latency.labels(
            model_name=model_name,
            version=version
        ).observe(latency_ms / 1000.0)
        
        # Store in database
        with self.Session() as session:
            log_entry = PredictionLogDB(
                model_name=model_name,
                version=version,
                prediction=prediction,
                actual=actual,
                features=features,
                latency_ms=latency_ms,
                experiment_id=experiment_id,
                variant=variant
            )
            session.add(log_entry)
            session.commit()
        
        # Check for drift if we have enough data
        await self._check_drift_async(model_name, features, prediction)
        
        # Update model performance if actual value provided
        if actual is not None:
            await self._update_model_performance(model_name, version, prediction, actual)
    
    async def _check_drift_async(self, model_name: str, 
                                features: Dict[str, Any],
                                prediction: float) -> None:
        """Asynchronously check for various types of drift"""
        
        # Feature drift detection
        for feature_name, feature_value in features.items():
            drift_results = await self._detect_feature_drift(
                model_name, feature_name, feature_value
            )
            
            for drift_type, result in drift_results.items():
                data_drift_score.labels(
                    feature_name=feature_name,
                    drift_type=drift_type
                ).set(result['score'])
                
                if result['is_drift']:
                    await self.alert_manager.send_drift_alert(
                        model_name, feature_name, drift_type, result
                    )
        
        # Prediction drift detection
        pred_drift = await self._detect_prediction_drift(model_name, prediction)
        if pred_drift:
            prediction_drift_score.labels(model_name=model_name).set(pred_drift['score'])
            
            if pred_drift['is_drift']:
                await self.alert_manager.send_prediction_drift_alert(
                    model_name, pred_drift
                )
    
    async def _detect_feature_drift(self, model_name: str, feature_name: str,
                                  feature_value: float) -> Dict[str, Dict[str, Any]]:
        """Detect drift in individual features"""
        results = {}
        
        # Get reference and current data
        reference_data = await self._get_reference_feature_data(model_name, feature_name)
        current_data = await self._get_current_feature_data(model_name, feature_name)
        
        if len(reference_data) > 100 and len(current_data) > 100:
            # PSI test
            psi_result = self.drift_detectors['psi'].detect(reference_data, current_data)
            results['psi'] = psi_result
            
            # KS test
            ks_result = self.drift_detectors['ks_test'].detect(reference_data, current_data)
            results['ks_test'] = ks_result
            
            # Wasserstein distance
            wasserstein_result = self.drift_detectors['wasserstein'].detect(
                reference_data, current_data
            )
            results['wasserstein'] = wasserstein_result
        
        return results
    
    async def _detect_prediction_drift(self, model_name: str, 
                                     prediction: float) -> Optional[Dict[str, Any]]:
        """Detect drift in model predictions"""
        # Get recent predictions
        recent_predictions = await self._get_recent_predictions(model_name, hours=24)
        reference_predictions = await self._get_reference_predictions(model_name)
        
        if len(recent_predictions) > 100 and len(reference_predictions) > 100:
            # Statistical tests
            ks_stat, p_value = stats.ks_2samp(reference_predictions, recent_predictions)
            
            drift_score = ks_stat
            is_drift = p_value < 0.05
            
            return {
                'score': drift_score,
                'p_value': p_value,
                'is_drift': is_drift,
                'threshold': 0.05
            }
        
        return None
    
    async def _update_model_performance(self, model_name: str, version: str,
                                      prediction: float, actual: float) -> None:
        """Update model performance metrics"""
        # Get recent predictions and actuals
        recent_data = await self._get_recent_predictions_with_actuals(
            model_name, version, hours=1
        )
        
        if len(recent_data) >= 10:
            predictions = [d['prediction'] for d in recent_data]
            actuals = [d['actual'] for d in recent_data]
            
            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
            
            # Update Prometheus metrics
            model_accuracy.labels(
                model_name=model_name,
                version=version,
                metric_type='mae'
            ).set(mae)
            
            model_accuracy.labels(
                model_name=model_name,
                version=version,
                metric_type='rmse'
            ).set(rmse)
            
            model_accuracy.labels(
                model_name=model_name,
                version=version,
                metric_type='r2'
            ).set(r2)
            
            # Check for performance degradation
            if await self._check_performance_degradation(model_name, version, mae, r2):
                await self.alert_manager.send_performance_alert(
                    model_name, version, {'mae': mae, 'r2': r2}
                )
    
    async def start_ab_test(self, experiment_config: Dict[str, Any]) -> str:
        """Start A/B test experiment"""
        experiment_id = await self.ab_test_manager.create_experiment(experiment_config)
        
        logger.info(f"Started A/B test: {experiment_id}")
        
        # Schedule monitoring task
        task = asyncio.create_task(
            self._monitor_ab_test(experiment_id)
        )
        self.monitoring_tasks.append(task)
        
        return experiment_id
    
    async def _monitor_ab_test(self, experiment_id: str) -> None:
        """Monitor running A/B test"""
        while True:
            try:
                # Check if experiment is still running
                status = await self.ab_test_manager.get_experiment_status(experiment_id)
                
                if status != 'running':
                    break
                
                # Analyze current results
                results = await self.ab_test_manager.analyze_experiment(experiment_id)
                
                # Update metrics
                for variant, metrics in results['variant_metrics'].items():
                    for metric_name, value in metrics.items():
                        ab_test_metrics.labels(
                            experiment_id=experiment_id,
                            variant=variant,
                            metric=metric_name
                        ).set(value)
                
                # Check for statistical significance
                if results.get('is_significant'):
                    await self.alert_manager.send_ab_test_completion_alert(
                        experiment_id, results
                    )
                    
                    # Auto-stop if configured
                    if self.config.get('auto_stop_significant_tests'):
                        await self.ab_test_manager.stop_experiment(
                            experiment_id, 'significant_result'
                        )
                        break
                
                # Check for minimum detectable effect
                if results.get('power') >= 0.8:
                    logger.info(f"Experiment {experiment_id} reached statistical power")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring A/B test {experiment_id}: {e}")
                await asyncio.sleep(60)
    
    def _create_dashboard(self) -> dash.Dash:
        """Create monitoring dashboard"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("ML Model Monitoring Dashboard"),
            
            dcc.Tabs(id="tabs", value='performance', children=[
                dcc.Tab(label='Model Performance', value='performance'),
                dcc.Tab(label='Drift Detection', value='drift'),
                dcc.Tab(label='A/B Tests', value='ab_tests'),
                dcc.Tab(label='Feature Importance', value='features')
            ]),
            
            html.Div(id='tabs-content'),
            
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ])
        
        @app.callback(Output('tabs-content', 'children'),
                     Input('tabs', 'value'))
        def render_content(tab):
            if tab == 'performance':
                return self._create_performance_tab()
            elif tab == 'drift':
                return self._create_drift_tab()
            elif tab == 'ab_tests':
                return self._create_ab_test_tab()
            elif tab == 'features':
                return self._create_features_tab()
        
        return app
    
    def _create_performance_tab(self) -> html.Div:
        """Create model performance dashboard tab"""
        return html.Div([
            html.H3("Model Performance Metrics"),
            
            dcc.Graph(id='performance-graph'),
            
            html.Div([
                html.Div([
                    html.H4("MAE Trend"),
                    dcc.Graph(id='mae-trend')
                ], className="six columns"),
                
                html.Div([
                    html.H4("R² Score Trend"),
                    dcc.Graph(id='r2-trend')
                ], className="six columns"),
            ], className="row"),
            
            html.H4("Prediction Distribution"),
            dcc.Graph(id='prediction-distribution')
        ])
    
    def _create_drift_tab(self) -> html.Div:
        """Create drift detection dashboard tab"""
        return html.Div([
            html.H3("Data Drift Detection"),
            
            html.Div([
                html.Div([
                    html.H4("Feature Drift Scores"),
                    dcc.Graph(id='feature-drift-heatmap')
                ], className="six columns"),
                
                html.Div([
                    html.H4("Prediction Drift"),
                    dcc.Graph(id='prediction-drift-trend')
                ], className="six columns"),
            ], className="row"),
            
            html.H4("Drift Detection Timeline"),
            dcc.Graph(id='drift-timeline')
        ])
    
    async def get_model_health_summary(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive model health summary"""
        with self.Session() as session:
            # Get recent performance data
            recent_data = session.query(PredictionLogDB).filter(
                PredictionLogDB.model_name == model_name,
                PredictionLogDB.actual.isnot(None),
                PredictionLogDB.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ).all()
            
            if not recent_data:
                return {'status': 'no_data', 'message': 'No recent predictions with actuals'}
            
            # Calculate performance metrics
            predictions = [d.prediction for d in recent_data]
            actuals = [d.actual for d in recent_data]
            
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = r2_score(actuals, predictions)
            
            # Get drift scores
            feature_drift_scores = {}
            for feature_name in recent_data[0].features.keys():
                drift_score = self._get_latest_drift_score(model_name, feature_name)
                if drift_score:
                    feature_drift_scores[feature_name] = drift_score
            
            # Determine overall health status
            health_status = 'healthy'
            issues = []
            
            if mae > self.config.get('mae_threshold', 2.0):
                health_status = 'degraded'
                issues.append(f'High MAE: {mae:.2f}')
            
            if r2 < self.config.get('r2_threshold', 0.8):
                health_status = 'degraded'
                issues.append(f'Low R²: {r2:.3f}')
            
            for feature, score in feature_drift_scores.items():
                if score > 0.1:
                    health_status = 'warning'
                    issues.append(f'Feature drift in {feature}: {score:.3f}')
            
            return {
                'model_name': model_name,
                'status': health_status,
                'issues': issues,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'prediction_count': len(recent_data)
                },
                'drift_scores': feature_drift_scores,
                'last_updated': datetime.utcnow().isoformat()
            }
    
    async def start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks"""
        tasks = [
            asyncio.create_task(self._periodic_drift_check()),
            asyncio.create_task(self._periodic_performance_check()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._generate_daily_reports())
        ]
        
        self.monitoring_tasks.extend(tasks)
        
        # Wait for tasks
        await asyncio.gather(*tasks)
    
    async def _periodic_drift_check(self) -> None:
        """Periodic drift detection across all models"""
        while True:
            try:
                # Get all active models
                with self.Session() as session:
                    active_models = session.query(PredictionLogDB.model_name).distinct().all()
                
                for (model_name,) in active_models:
                    await self._comprehensive_drift_check(model_name)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in periodic drift check: {e}")
                await asyncio.sleep(300)
    
    async def _comprehensive_drift_check(self, model_name: str) -> None:
        """Comprehensive drift check for a model"""
        try:
            # Feature drift check
            recent_features = await self._get_recent_feature_data(model_name)
            reference_features = await self._get_reference_feature_data_all(model_name)
            
            if recent_features and reference_features:
                for feature_name in recent_features[0].keys():
                    recent_values = [f[feature_name] for f in recent_features if feature_name in f]
                    ref_values = [f[feature_name] for f in reference_features if feature_name in f]
                    
                    if len(recent_values) > 30 and len(ref_values) > 30:
                        # Multiple drift tests
                        psi_score = self.drift_detectors['psi'].detect(ref_values, recent_values)
                        ks_result = self.drift_detectors['ks_test'].detect(ref_values, recent_values)
                        
                        # Update metrics
                        data_drift_score.labels(
                            feature_name=feature_name,
                            drift_type='psi'
                        ).set(psi_score['score'])
                        
                        # Alert if significant drift
                        if psi_score['is_drift'] or ks_result['is_drift']:
                            await self.alert_manager.send_drift_alert(
                                model_name, feature_name, 'comprehensive', {
                                    'psi': psi_score,
                                    'ks_test': ks_result
                                }
                            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive drift check for {model_name}: {e}")


class PopulationStabilityIndex:
    """Population Stability Index drift detector"""
    
    def __init__(self, threshold: float = 0.1, bins: int = 10):
        self.threshold = threshold
        self.bins = bins
    
    def detect(self, reference: List[float], current: List[float]) -> Dict[str, Any]:
        """Calculate PSI between reference and current data"""
        ref_array = np.array(reference)
        curr_array = np.array(current)
        
        # Create bins based on reference data
        _, bin_edges = np.histogram(ref_array, bins=self.bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(ref_array, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_array, bins=bin_edges)
        
        # Normalize to probabilities
        ref_probs = ref_counts / len(reference)
        curr_probs = curr_counts / len(current)
        
        # Add small value to avoid division by zero
        ref_probs = np.where(ref_probs == 0, 1e-6, ref_probs)
        curr_probs = np.where(curr_probs == 0, 1e-6, curr_probs)
        
        # Calculate PSI
        psi = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
        
        return {
            'score': psi,
            'threshold': self.threshold,
            'is_drift': psi > self.threshold
        }


class KolmogorovSmirnovTest:
    """Kolmogorov-Smirnov test for drift detection"""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
    
    def detect(self, reference: List[float], current: List[float]) -> Dict[str, Any]:
        """Perform KS test"""
        statistic, p_value = stats.ks_2samp(reference, current)
        
        return {
            'score': statistic,
            'p_value': p_value,
            'threshold': self.threshold,
            'is_drift': p_value < self.threshold
        }


class WassersteinDistance:
    """Wasserstein distance for drift detection"""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def detect(self, reference: List[float], current: List[float]) -> Dict[str, Any]:
        """Calculate Wasserstein distance"""
        distance = stats.wasserstein_distance(reference, current)
        
        return {
            'score': distance,
            'threshold': self.threshold,
            'is_drift': distance > self.threshold
        }


class FeatureCorrelationDrift:
    """Detect drift in feature correlations"""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def detect(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrix drift"""
        ref_corr = reference_df.corr()
        curr_corr = current_df.corr()
        
        # Calculate Frobenius norm of difference
        diff_norm = np.linalg.norm(ref_corr - curr_corr, 'fro')
        
        return {
            'score': diff_norm,
            'threshold': self.threshold,
            'is_drift': diff_norm > self.threshold
        }


class AlertManager:
    """Alert management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhooks = config.get('alert_webhooks', {})
        
    async def send_drift_alert(self, model_name: str, feature_name: str,
                             drift_type: str, result: Dict[str, Any]) -> None:
        """Send drift detection alert"""
        alert = {
            'type': 'drift_detection',
            'severity': 'warning',
            'model': model_name,
            'feature': feature_name,
            'drift_type': drift_type,
            'score': result['score'],
            'threshold': result['threshold'],
            'timestamp': datetime.utcnow().isoformat(),
            'message': f"Drift detected in feature {feature_name} for model {model_name}"
        }
        
        await self._send_alert(alert)
    
    async def send_performance_alert(self, model_name: str, version: str,
                                   metrics: Dict[str, float]) -> None:
        """Send performance degradation alert"""
        alert = {
            'type': 'performance_degradation',
            'severity': 'critical',
            'model': model_name,
            'version': version,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat(),
            'message': f"Performance degradation detected for model {model_name}"
        }
        
        await self._send_alert(alert)
    
    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to configured channels"""
        if 'slack' in self.webhooks:
            await self._send_slack_alert(alert)
        
        if 'email' in self.config:
            await self._send_email_alert(alert)
        
        logger.warning(f"Alert sent: {alert}")
    
    async def _send_slack_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to Slack"""
        webhook_url = self.webhooks['slack']
        
        message = {
            'text': f"🚨 ML Alert: {alert['message']}",
            'attachments': [{
                'color': 'danger' if alert['severity'] == 'critical' else 'warning',
                'fields': [
                    {'title': 'Model', 'value': alert.get('model', 'N/A'), 'short': True},
                    {'title': 'Type', 'value': alert['type'], 'short': True},
                    {'title': 'Timestamp', 'value': alert['timestamp'], 'short': False}
                ]
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    if response.status != 200:
                        logger.error(f"Failed to send Slack alert: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")


class ABTestManager:
    """A/B testing framework for model experiments"""
    
    def __init__(self, config: Dict[str, Any], session_factory):
        self.config = config
        self.Session = session_factory
    
    async def create_experiment(self, config: Dict[str, Any]) -> str:
        """Create new A/B test experiment"""
        experiment_id = f"exp_{int(time.time())}"
        
        with self.Session() as session:
            experiment = ExperimentDB(
                experiment_id=experiment_id,
                model_a=config['model_a'],
                model_b=config['model_b'],
                traffic_split=config.get('traffic_split', 0.5),
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=config.get('duration_days', 7)),
                status='running'
            )
            session.add(experiment)
            session.commit()
        
        return experiment_id
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        with self.Session() as session:
            # Get experiment data
            experiment = session.query(ExperimentDB).filter_by(
                experiment_id=experiment_id
            ).first()
            
            if not experiment:
                return {'error': 'Experiment not found'}
            
            # Get prediction logs
            logs_a = session.query(PredictionLogDB).filter(
                PredictionLogDB.experiment_id == experiment_id,
                PredictionLogDB.variant == 'A',
                PredictionLogDB.actual.isnot(None)
            ).all()
            
            logs_b = session.query(PredictionLogDB).filter(
                PredictionLogDB.experiment_id == experiment_id,
                PredictionLogDB.variant == 'B',
                PredictionLogDB.actual.isnot(None)
            ).all()
            
            if len(logs_a) < 30 or len(logs_b) < 30:
                return {'error': 'Insufficient data for analysis'}
            
            # Calculate metrics for each variant
            def calculate_metrics(logs):
                predictions = [log.prediction for log in logs]
                actuals = [log.actual for log in logs]
                
                return {
                    'mae': mean_absolute_error(actuals, predictions),
                    'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                    'r2': r2_score(actuals, predictions),
                    'count': len(logs)
                }
            
            metrics_a = calculate_metrics(logs_a)
            metrics_b = calculate_metrics(logs_b)
            
            # Statistical significance test
            errors_a = [abs(log.prediction - log.actual) for log in logs_a]
            errors_b = [abs(log.prediction - log.actual) for log in logs_b]
            
            t_stat, p_value = stats.ttest_ind(errors_a, errors_b)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(errors_a) - 1) * np.var(errors_a) + 
                                (len(errors_b) - 1) * np.var(errors_b)) / 
                               (len(errors_a) + len(errors_b) - 2))
            cohens_d = (np.mean(errors_a) - np.mean(errors_b)) / pooled_std
            
            # Power analysis
            from statsmodels.stats.power import ttest_power
            power = ttest_power(cohens_d, len(errors_a), 0.05)
            
            return {
                'experiment_id': experiment_id,
                'variant_metrics': {
                    'A': metrics_a,
                    'B': metrics_b
                },
                'statistical_test': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d
                },
                'is_significant': p_value < 0.05,
                'power': power,
                'recommendation': self._get_recommendation(metrics_a, metrics_b, p_value)
            }
    
    def _get_recommendation(self, metrics_a: Dict[str, float], 
                          metrics_b: Dict[str, float], p_value: float) -> str:
        """Generate recommendation based on test results"""
        if p_value >= 0.05:
            return "No significant difference detected. Continue testing or implement either variant."
        
        if metrics_b['mae'] < metrics_a['mae'] and metrics_b['r2'] > metrics_a['r2']:
            return "Variant B shows superior performance. Recommend full rollout."
        elif metrics_a['mae'] < metrics_b['mae'] and metrics_a['r2'] > metrics_b['r2']:
            return "Variant A shows superior performance. Maintain current model."
        else:
            return "Mixed results. Consider business context and additional metrics."


if __name__ == "__main__":
    config = {
        'postgres_url': 'postgresql://user:pass@localhost/monitoring',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'mlflow_uri': 'http://localhost:5000',
        'environment': 'production',
        'metrics_port': 8000,
        'alert_webhooks': {
            'slack': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        }
    }
    
    monitoring_system = MLMonitoringSystem(config)
    
    # Start monitoring
    asyncio.run(monitoring_system.start_monitoring_tasks())