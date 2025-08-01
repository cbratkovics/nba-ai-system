"""
Online Learning System for Real-time Model Updates
Implements incremental learning, concept drift detection, and adaptive models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import river
from river import drift, metrics, preprocessing, ensemble, tree, linear_model, stats
from river.datasets import synth
import torch
import torch.nn as nn
from collections import deque
import asyncio
import json
import redis
from prometheus_client import Counter, Gauge, Histogram
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Metrics
online_predictions = Counter('online_predictions_total', 'Total online predictions')
drift_detections = Counter('drift_detections_total', 'Concept drift detections', ['drift_type'])
model_updates = Counter('model_updates_total', 'Model updates', ['update_type'])
online_performance = Gauge('online_model_performance', 'Online model performance', ['model', 'metric'])
adaptation_rate = Gauge('model_adaptation_rate', 'Model adaptation rate')

logger = logging.getLogger(__name__)


@dataclass
class StreamingData:
    """Streaming data point"""
    features: Dict[str, float]
    target: Optional[float]
    timestamp: datetime
    player_id: str
    game_id: str
    importance_weight: float = 1.0


class OnlineLearningSystem:
    """Production online learning system with drift detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Drift detectors
        self.drift_detectors = self._initialize_drift_detectors()
        
        # Performance tracking
        self.metrics = self._initialize_metrics()
        
        # Model history for rollback
        self.model_history = deque(maxlen=config.get('history_size', 10))
        
        # Redis for state persistence
        self.redis_client = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        
        # Adaptive learning rate
        self.learning_rate_scheduler = AdaptiveLearningRate(
            initial_lr=config.get('initial_lr', 0.01)
        )
        
        # Feature statistics for normalization
        self.feature_stats = {}
        
        # Ensemble weights
        self.ensemble_weights = np.ones(len(self.models)) / len(self.models)
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize online learning models"""
        models = {
            # Linear models
            'sgd_regressor': preprocessing.StandardScaler() | linear_model.SGDRegressor(
                learning_rate=0.01,
                loss='squared',
                l2=0.001
            ),
            
            # Tree-based models
            'hoeffding_tree': preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(
                grace_period=50,
                split_confidence=0.01,
                tie_threshold=0.05
            ),
            
            'hoeffding_adaptive_tree': preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=50,
                split_confidence=0.01,
                leaf_prediction='adaptive',
                model_selector_decay=0.95
            ),
            
            # Ensemble models
            'adaptive_random_forest': preprocessing.StandardScaler() | ensemble.AdaptiveRandomForestRegressor(
                n_models=10,
                max_features='sqrt',
                lambda_value=6,
                grace_period=50,
                split_confidence=0.01,
                tie_threshold=0.05,
                leaf_prediction='adaptive',
                model_selector_decay=0.95,
                drift_detector=drift.ADWIN(delta=0.01),
                warning_detector=drift.ADWIN(delta=0.1)
            ),
            
            'bagging': preprocessing.StandardScaler() | ensemble.BaggingRegressor(
                model=tree.HoeffdingTreeRegressor(),
                n_models=10,
                seed=42
            ),
            
            # Neural network (custom implementation)
            'neural_net': OnlineNeuralNetwork(
                input_dim=self.config.get('input_dim', 20),
                hidden_dims=[64, 32],
                learning_rate=0.001
            )
        }
        
        return models
    
    def _initialize_drift_detectors(self) -> Dict[str, Any]:
        """Initialize concept drift detectors"""
        return {
            'adwin': drift.ADWIN(delta=0.002),
            'ddm': drift.DDM(warm_start=30, warning_level=2.0, drift_level=3.0),
            'eddm': drift.EDDM(warm_start=30, alpha_warning=0.95, alpha_drift=0.9),
            'page_hinkley': drift.PageHinkley(delta=0.005, threshold=50, alpha=0.9999),
            'kswin': drift.KSWIN(alpha=0.003, window_size=100, stat_size=30)
        }
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize performance metrics"""
        metric_types = {
            'mae': metrics.MAE(),
            'rmse': metrics.RMSE(),
            'r2': metrics.R2(),
            'rolling_mae': metrics.Rolling(metrics.MAE(), window_size=100),
            'rolling_r2': metrics.Rolling(metrics.R2(), window_size=100)
        }
        
        # Create metrics for each model
        model_metrics = {}
        for model_name in self.models.keys():
            model_metrics[model_name] = {
                metric_name: metric.clone()
                for metric_name, metric in metric_types.items()
            }
        
        return model_metrics
    
    async def process_stream(self, data_point: StreamingData) -> Dict[str, Any]:
        """Process single streaming data point"""
        online_predictions.inc()
        
        # Make predictions with all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict_one(data_point.features)
                predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Prediction error for {model_name}: {e}")
                predictions[model_name] = None
        
        # Ensemble prediction
        valid_preds = [p for p in predictions.values() if p is not None]
        if valid_preds:
            ensemble_pred = np.average(
                valid_preds,
                weights=self.ensemble_weights[:len(valid_preds)]
            )
        else:
            ensemble_pred = None
        
        result = {
            'timestamp': data_point.timestamp,
            'player_id': data_point.player_id,
            'predictions': predictions,
            'ensemble_prediction': ensemble_pred
        }
        
        # If we have true label, update models
        if data_point.target is not None:
            await self.update_models(data_point, predictions)
        
        # Check for drift
        drift_detected = await self.check_drift(data_point, predictions)
        if drift_detected:
            result['drift_detected'] = drift_detected
            
        return result
    
    async def update_models(self, data_point: StreamingData, 
                          predictions: Dict[str, float]) -> None:
        """Update models with new data"""
        model_updates.labels(update_type='incremental').inc()
        
        # Update each model
        for model_name, model in self.models.items():
            if model_name in predictions and predictions[model_name] is not None:
                # Update metrics before learning
                for metric_name, metric in self.metrics[model_name].items():
                    metric.update(data_point.target, predictions[model_name])
                
                # Apply importance weighting
                if hasattr(model, 'learn_one'):
                    if data_point.importance_weight != 1.0:
                        # Weighted update by repeating the sample
                        repeat_times = int(data_point.importance_weight)
                        for _ in range(repeat_times):
                            model.learn_one(data_point.features, data_point.target)
                    else:
                        model.learn_one(data_point.features, data_point.target)
                elif isinstance(model, OnlineNeuralNetwork):
                    model.update(data_point.features, data_point.target)
        
        # Update ensemble weights based on recent performance
        self._update_ensemble_weights()
        
        # Update learning rate
        avg_error = np.mean([
            self.metrics[model_name]['mae'].get()
            for model_name in self.models.keys()
        ])
        self.learning_rate_scheduler.update(avg_error)
        
        # Update Prometheus metrics
        for model_name in self.models.keys():
            online_performance.labels(
                model=model_name,
                metric='mae'
            ).set(self.metrics[model_name]['mae'].get())
    
    async def check_drift(self, data_point: StreamingData,
                        predictions: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Check for concept drift"""
        drift_info = {}
        
        if data_point.target is not None:
            # Check prediction error drift
            ensemble_pred = np.mean(list(predictions.values()))
            error = abs(data_point.target - ensemble_pred)
            
            for detector_name, detector in self.drift_detectors.items():
                detector.update(error)
                
                if detector.drift_detected:
                    drift_detections.labels(drift_type=detector_name).inc()
                    drift_info[detector_name] = {
                        'detected': True,
                        'timestamp': data_point.timestamp
                    }
                    
                    # Handle drift
                    await self.handle_drift(detector_name)
        
        # Check feature drift using statistical tests
        feature_drift = await self._check_feature_drift(data_point.features)
        if feature_drift:
            drift_info['feature_drift'] = feature_drift
        
        return drift_info if drift_info else None
    
    async def handle_drift(self, drift_type: str) -> None:
        """Handle detected concept drift"""
        logger.warning(f"Concept drift detected: {drift_type}")
        
        if self.config.get('drift_adaptation_strategy') == 'reset':
            # Reset poorly performing models
            for model_name, model in self.models.items():
                if self.metrics[model_name]['rolling_mae'].get() > self.config['drift_threshold']:
                    logger.info(f"Resetting model {model_name} due to drift")
                    self.models[model_name] = self._create_fresh_model(model_name)
                    model_updates.labels(update_type='reset').inc()
                    
        elif self.config.get('drift_adaptation_strategy') == 'ensemble_adapt':
            # Increase weight of adaptive models
            adaptive_models = ['hoeffding_adaptive_tree', 'adaptive_random_forest']
            for i, model_name in enumerate(self.models.keys()):
                if model_name in adaptive_models:
                    self.ensemble_weights[i] *= 1.2
            
            # Renormalize weights
            self.ensemble_weights /= self.ensemble_weights.sum()
            
        elif self.config.get('drift_adaptation_strategy') == 'increase_learning':
            # Temporarily increase learning rate
            self.learning_rate_scheduler.boost(factor=2.0, duration=100)
    
    def _update_ensemble_weights(self) -> None:
        """Update ensemble weights based on recent performance"""
        # Get recent performance for each model
        performances = []
        for model_name in self.models.keys():
            mae = self.metrics[model_name]['rolling_mae'].get()
            if mae > 0:
                performance = 1.0 / mae  # Inverse MAE as performance
            else:
                performance = 1.0
            performances.append(performance)
        
        # Convert to weights (softmax-like)
        performances = np.array(performances)
        exp_perf = np.exp(performances - np.max(performances))
        self.ensemble_weights = exp_perf / exp_perf.sum()
        
        # Apply minimum weight threshold
        min_weight = 0.05
        self.ensemble_weights = np.maximum(self.ensemble_weights, min_weight)
        self.ensemble_weights /= self.ensemble_weights.sum()
    
    async def _check_feature_drift(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Check for feature distribution drift"""
        drift_info = {}
        
        for feature_name, value in features.items():
            if feature_name not in self.feature_stats:
                self.feature_stats[feature_name] = stats.Mean() | stats.Var()
            
            # Update statistics
            self.feature_stats[feature_name].update(value)
            
            # Check if value is anomalous (>3 std devs)
            mean = self.feature_stats[feature_name][0].get()
            var = self.feature_stats[feature_name][1].get()
            
            if var > 0:
                std = np.sqrt(var)
                z_score = abs(value - mean) / std
                
                if z_score > 3:
                    drift_info[feature_name] = {
                        'z_score': z_score,
                        'value': value,
                        'expected_range': (mean - 3*std, mean + 3*std)
                    }
        
        return drift_info if drift_info else None
    
    def save_state(self) -> None:
        """Save model state to Redis"""
        state = {
            'ensemble_weights': self.ensemble_weights.tolist(),
            'metrics': {
                model_name: {
                    metric_name: metric.get()
                    for metric_name, metric in model_metrics.items()
                }
                for model_name, model_metrics in self.metrics.items()
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.redis_client.set(
            f"online_learning_state:{self.config['model_id']}",
            json.dumps(state),
            ex=86400  # 24 hour expiry
        )
    
    def load_state(self) -> bool:
        """Load model state from Redis"""
        state_json = self.redis_client.get(f"online_learning_state:{self.config['model_id']}")
        
        if state_json:
            state = json.loads(state_json)
            self.ensemble_weights = np.array(state['ensemble_weights'])
            logger.info(f"Loaded state from {state['timestamp']}")
            return True
        
        return False
    
    def _create_fresh_model(self, model_name: str) -> Any:
        """Create a fresh instance of a model"""
        # Re-initialize the specific model type
        if model_name == 'sgd_regressor':
            return preprocessing.StandardScaler() | linear_model.SGDRegressor(
                learning_rate=0.01,
                loss='squared',
                l2=0.001
            )
        # Add other model types...
        
        return self.models[model_name].clone()


class OnlineNeuralNetwork:
    """Online neural network with incremental updates"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=1000)
        
    def predict_one(self, x: Dict[str, float]) -> float:
        """Make single prediction"""
        # Convert to tensor
        x_tensor = torch.tensor(
            [x[f'feature_{i}'] for i in range(self.input_dim)],
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).item()
        
        return pred
    
    def update(self, x: Dict[str, float], y: float) -> None:
        """Update model with single sample"""
        # Add to replay buffer
        self.replay_buffer.append((x, y))
        
        # Convert to tensor
        x_tensor = torch.tensor(
            [x[f'feature_{i}'] for i in range(self.input_dim)],
            dtype=torch.float32
        ).unsqueeze(0)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        
        # Single sample update
        self.model.train()
        self.optimizer.zero_grad()
        
        pred = self.model(x_tensor).squeeze()
        loss = nn.functional.mse_loss(pred, y_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        # Experience replay
        if len(self.replay_buffer) >= 32:
            self._experience_replay()
    
    def _experience_replay(self, batch_size: int = 32) -> None:
        """Perform experience replay"""
        # Sample from buffer
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Prepare batch
        X_batch = torch.stack([
            torch.tensor(
                [x[f'feature_{i}'] for i in range(self.input_dim)],
                dtype=torch.float32
            )
            for x, _ in batch
        ])
        
        y_batch = torch.tensor([y for _, y in batch], dtype=torch.float32)
        
        # Update
        self.model.train()
        self.optimizer.zero_grad()
        
        pred = self.model(X_batch).squeeze()
        loss = nn.functional.mse_loss(pred, y_batch)
        
        loss.backward()
        self.optimizer.step()


class AdaptiveLearningRate:
    """Adaptive learning rate scheduler"""
    
    def __init__(self, initial_lr: float = 0.01):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.error_history = deque(maxlen=100)
        self.boost_remaining = 0
        self.boost_factor = 1.0
        
    def update(self, error: float) -> float:
        """Update learning rate based on error"""
        self.error_history.append(error)
        
        if len(self.error_history) >= 10:
            # Calculate error trend
            recent_errors = list(self.error_history)[-10:]
            older_errors = list(self.error_history)[-20:-10] if len(self.error_history) >= 20 else recent_errors
            
            recent_avg = np.mean(recent_errors)
            older_avg = np.mean(older_errors)
            
            # Adjust learning rate
            if recent_avg > older_avg * 1.1:  # Error increasing
                self.current_lr *= 1.1  # Increase learning rate
            elif recent_avg < older_avg * 0.9:  # Error decreasing
                self.current_lr *= 0.95  # Decrease learning rate
            
            # Apply bounds
            self.current_lr = np.clip(
                self.current_lr,
                self.initial_lr * 0.1,
                self.initial_lr * 10
            )
        
        # Apply temporary boost if active
        if self.boost_remaining > 0:
            self.boost_remaining -= 1
            return self.current_lr * self.boost_factor
        
        return self.current_lr
    
    def boost(self, factor: float = 2.0, duration: int = 100) -> None:
        """Temporarily boost learning rate"""
        self.boost_factor = factor
        self.boost_remaining = duration
        
    def get(self) -> float:
        """Get current learning rate"""
        if self.boost_remaining > 0:
            return self.current_lr * self.boost_factor
        return self.current_lr


# Specialized drift detectors
class FeatureDriftDetector:
    """Detect drift in feature distributions"""
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        
    def update(self, features: Dict[str, float]) -> bool:
        """Update detector and check for drift"""
        self.current_window.append(features)
        
        if len(self.current_window) == self.window_size and len(self.reference_window) == self.window_size:
            # Perform Kolmogorov-Smirnov test for each feature
            drift_detected = False
            
            for feature_name in features.keys():
                ref_values = [x[feature_name] for x in self.reference_window if feature_name in x]
                curr_values = [x[feature_name] for x in self.current_window if feature_name in x]
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    from scipy.stats import ks_2samp
                    statistic, p_value = ks_2samp(ref_values, curr_values)
                    
                    if p_value < self.threshold:
                        drift_detected = True
                        logger.warning(f"Feature drift detected in {feature_name}: p={p_value}")
            
            # Update reference window if no drift
            if not drift_detected:
                self.reference_window = self.current_window.copy()
            
            return drift_detected
        
        # Initialize reference window
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(features)
        
        return False


if __name__ == "__main__":
    # Example usage
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'model_id': 'nba_online_predictor',
        'input_dim': 20,
        'history_size': 10,
        'initial_lr': 0.01,
        'drift_threshold': 2.0,
        'drift_adaptation_strategy': 'ensemble_adapt'
    }
    
    system = OnlineLearningSystem(config)
    
    # Simulate streaming data
    async def simulate_stream():
        for i in range(1000):
            # Create synthetic data point
            features = {f'feature_{j}': np.random.randn() for j in range(20)}
            
            data_point = StreamingData(
                features=features,
                target=np.random.randn() if i > 10 else None,  # No target for first 10
                timestamp=datetime.utcnow(),
                player_id=f'player_{i % 10}',
                game_id=f'game_{i // 100}',
                importance_weight=1.0 if i % 10 != 0 else 2.0  # Weight every 10th sample
            )
            
            result = await system.process_stream(data_point)
            
            if i % 100 == 0:
                logger.info(f"Processed {i} samples, ensemble prediction: {result['ensemble_prediction']}")
                
                # Save state periodically
                system.save_state()
    
    # Run simulation
    asyncio.run(simulate_stream())