"""
Advanced ML Engineering Pipeline with Automated Hyperparameter Tuning,
Distributed Training, and Online Learning Capabilities
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import optuna
from optuna.integration import MLflowCallback
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
import horovod.tensorflow as hvd
import mlflow
from river import metrics as river_metrics
from river import ensemble as river_ensemble
from river import preprocessing as river_preprocessing
import dask.dataframe as dd
from dask.distributed import Client, as_completed
from dask_ml.model_selection import GridSearchCV
import joblib
from prometheus_client import Counter, Histogram, Gauge
import logging
import warnings
warnings.filterwarnings('ignore')

# Metrics
training_time = Histogram('ml_training_time_seconds', 'Model training time', ['model_type'])
hyperparameter_trials = Counter('ml_hyperparameter_trials_total', 'Hyperparameter optimization trials')
model_performance = Gauge('ml_model_performance', 'Model performance metrics', ['model', 'metric'])
online_updates = Counter('ml_online_updates_total', 'Online learning updates')

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    distributed_config: Optional[Dict[str, Any]] = None


class CustomLoss(nn.Module):
    """Custom loss function for NBA predictions"""
    
    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, predictions, targets, player_weights=None):
        """
        Custom loss that weights errors based on player importance
        and penalizes underestimation more than overestimation
        """
        errors = predictions - targets
        
        # Asymmetric loss - penalize underestimation more
        base_loss = torch.where(
            errors < 0,
            self.alpha * torch.abs(errors) ** 2,  # Underestimation
            torch.abs(errors) ** 2  # Overestimation
        )
        
        # Weight by player importance if provided
        if player_weights is not None:
            weighted_loss = base_loss * player_weights
        else:
            weighted_loss = base_loss
        
        # Add regularization for extreme predictions
        regularization = self.beta * torch.mean(torch.abs(predictions - targets.mean()))
        
        return torch.mean(weighted_loss) + regularization


class NBADeepModel(nn.Module):
    """Deep learning model for NBA predictions with attention mechanism"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=4,
            dropout=dropout
        )
        
        # Output layers for multi-task learning
        self.points_head = nn.Linear(hidden_dims[-1], 1)
        self.rebounds_head = nn.Linear(hidden_dims[-1], 1)
        self.assists_head = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x, return_attention=False):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        features_reshaped = features.unsqueeze(0)  # Add sequence dimension
        attended_features, attention_weights = self.attention(
            features_reshaped, features_reshaped, features_reshaped
        )
        attended_features = attended_features.squeeze(0)
        
        # Multi-task predictions
        points = self.points_head(attended_features)
        rebounds = self.rebounds_head(attended_features)
        assists = self.assists_head(attended_features)
        
        outputs = {
            'points': points,
            'rebounds': rebounds,
            'assists': assists
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs


class MLPipeline:
    """Advanced ML pipeline with sophisticated features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config['mlflow_uri'])
        
        # Initialize Ray for distributed training
        if not ray.is_initialized():
            ray.init(num_cpus=config.get('ray_cpus', 8))
        
        # Initialize Dask for distributed data processing
        self.dask_client = Client(config.get('dask_scheduler', 'localhost:8786'))
        
        # Model registry
        self.models = {}
        self.ensemble_weights = {}
        
        # Online learning components
        self.online_models = {}
        self._initialize_online_models()
        
    def automated_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                      model_type: str = 'random_forest') -> Dict[str, Any]:
        """Automated hyperparameter tuning with Optuna"""
        hyperparameter_trials.inc()
        
        def objective(trial):
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                }
                
                model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
                
            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7])
                }
                
                model = GradientBoostingRegressor(**params, random_state=42)
            
            elif model_type == 'deep_learning':
                params = {
                    'hidden_dims': trial.suggest_categorical(
                        'hidden_dims', 
                        [[256, 128, 64], [512, 256, 128], [128, 64, 32]]
                    ),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'epochs': trial.suggest_int('epochs', 50, 200)
                }
                
                return self._train_deep_model(X_train, y_train, params)
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            return -scores.mean()
        
        # Create study with Optuna
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{model_type}_optimization',
            storage=self.config.get('optuna_storage', 'sqlite:///optuna.db'),
            load_if_exists=True
        )
        
        # Add MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.config['mlflow_uri'],
            metric_name='mae'
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=self.config.get('n_trials', 100),
            callbacks=[mlflow_callback],
            n_jobs=1  # Parallel trials
        )
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best MAE: {study.best_value}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def distributed_training_horovod(self, model: tf.keras.Model, 
                                   train_dataset: tf.data.Dataset,
                                   val_dataset: tf.data.Dataset) -> tf.keras.Model:
        """Distributed training with Horovod for TensorFlow"""
        # Initialize Horovod
        hvd.init()
        
        # Pin GPU to be used to process local rank
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
        # Scale learning rate
        scaled_lr = self.config['learning_rate'] * hvd.size()
        
        # Horovod distributed optimizer
        optimizer = tf.keras.optimizers.Adam(scaled_lr)
        optimizer = hvd.DistributedOptimizer(optimizer)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='mae',
            metrics=['mae', 'mse'],
            experimental_run_tf_function=False
        )
        
        # Callbacks
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=scaled_lr,
                warmup_epochs=5,
                verbose=1
            )
        ]
        
        # Add checkpointing on rank 0
        if hvd.rank() == 0:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    './checkpoint-{epoch}.h5',
                    save_best_only=True
                )
            )
        
        # Train model
        with training_time.labels(model_type='tensorflow_distributed').time():
            history = model.fit(
                train_dataset,
                epochs=self.config['epochs'],
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=1 if hvd.rank() == 0 else 0
            )
        
        return model
    
    def distributed_training_pytorch_ddp(self, model: nn.Module,
                                       train_loader: DataLoader,
                                       val_loader: DataLoader) -> nn.Module:
        """Distributed Data Parallel training for PyTorch"""
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        
        # Get rank and world size
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Move model to GPU
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'] * world_size
        )
        
        # Loss function
        criterion = CustomLoss()
        
        # Training loop
        with training_time.labels(model_type='pytorch_distributed').time():
            for epoch in range(self.config['epochs']):
                model.train()
                epoch_loss = 0
                
                for batch_idx, (features, targets) in enumerate(train_loader):
                    features, targets = features.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs['points'], targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Validation
                if rank == 0:
                    val_loss = self._validate_pytorch_model(model, val_loader, criterion, device)
                    logger.info(f"Epoch {epoch}: Train Loss: {epoch_loss/len(train_loader)}, "
                              f"Val Loss: {val_loss}")
        
        return model
    
    def ray_tune_hyperparameter_search(self, train_fn: Callable,
                                     search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Hyperparameter search with Ray Tune"""
        
        # Configure search algorithm
        optuna_search = OptunaSearch(
            metric="val_loss",
            mode="min"
        )
        
        # Configure scheduler
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=100,
            grace_period=10,
            reduction_factor=3
        )
        
        # Run hyperparameter search
        analysis = tune.run(
            train_fn,
            config=search_space,
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=self.config.get('ray_samples', 50),
            resources_per_trial={
                "cpu": 4,
                "gpu": 1 if torch.cuda.is_available() else 0
            },
            checkpoint_score_attr="val_loss",
            keep_checkpoints_num=5
        )
        
        # Get best config
        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        
        logger.info(f"Best config: {best_config}")
        
        return {
            'best_config': best_config,
            'analysis': analysis
        }
    
    def online_learning_update(self, new_data: pd.DataFrame, 
                             target_col: str = 'points') -> None:
        """Update models with online learning"""
        online_updates.inc()
        
        for index, row in new_data.iterrows():
            features = row.drop(target_col).to_dict()
            target = row[target_col]
            
            # Update each online model
            for model_name, model in self.online_models.items():
                # Make prediction before update
                y_pred = model.predict_one(features)
                
                # Update model
                model.learn_one(features, target)
                
                # Update metrics
                if model_name not in self.online_metrics:
                    self.online_metrics[model_name] = river_metrics.MAE()
                
                self.online_metrics[model_name].update(target, y_pred)
        
        # Log performance
        for model_name, metric in self.online_metrics.items():
            model_performance.labels(
                model=f'online_{model_name}',
                metric='mae'
            ).set(metric.get())
    
    def ensemble_model_management(self, models: List[BaseEstimator],
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Sophisticated ensemble model management"""
        
        # Get predictions from all models
        predictions = np.column_stack([
            model.predict(X_val) for model in models
        ])
        
        # Calculate optimal weights using optimization
        from scipy.optimize import minimize
        
        def ensemble_loss(weights):
            weighted_pred = np.dot(predictions, weights)
            return mean_absolute_error(y_val, weighted_pred)
        
        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]
        
        # Initial weights
        n_models = len(models)
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            ensemble_loss,
            initial_weights,
            method='SLSQP',
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        # Create ensemble predictor
        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict(self, X):
                predictions = np.column_stack([
                    model.predict(X) for model in self.models
                ])
                return np.dot(predictions, self.weights)
        
        ensemble = WeightedEnsemble(models, optimal_weights)
        
        # Calculate ensemble performance
        ensemble_pred = ensemble.predict(X_val)
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        
        logger.info(f"Ensemble weights: {optimal_weights}")
        logger.info(f"Ensemble MAE: {ensemble_mae}, R2: {ensemble_r2}")
        
        return {
            'ensemble': ensemble,
            'weights': optimal_weights,
            'mae': ensemble_mae,
            'r2': ensemble_r2
        }
    
    def uncertainty_quantification(self, model: Any, X: pd.DataFrame,
                                 n_iterations: int = 100) -> Dict[str, np.ndarray]:
        """Quantify prediction uncertainty"""
        
        if hasattr(model, 'estimators_'):  # Random Forest
            # Use individual tree predictions
            predictions = np.array([
                tree.predict(X) for tree in model.estimators_
            ])
            
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            # Calculate percentiles
            lower_bound = np.percentile(predictions, 5, axis=0)
            upper_bound = np.percentile(predictions, 95, axis=0)
            
        else:
            # Use dropout for neural networks or bootstrap for others
            predictions = []
            
            for _ in range(n_iterations):
                if hasattr(model, 'train'):  # Neural network
                    # Enable dropout during prediction
                    model.train()
                    pred = model(torch.tensor(X.values, dtype=torch.float32))
                    predictions.append(pred['points'].detach().numpy())
                else:
                    # Bootstrap sampling
                    indices = np.random.choice(len(X), size=len(X), replace=True)
                    X_bootstrap = X.iloc[indices]
                    pred = model.predict(X_bootstrap)
                    predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            lower_bound = np.percentile(predictions, 5, axis=0)
            upper_bound = np.percentile(predictions, 95, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_interval': upper_bound - lower_bound
        }
    
    def multi_task_learning(self, X_train: pd.DataFrame, 
                          y_train: pd.DataFrame) -> nn.Module:
        """Implement multi-task learning for multiple targets"""
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_points = torch.tensor(y_train['points'].values, dtype=torch.float32)
        y_rebounds = torch.tensor(y_train['rebounds'].values, dtype=torch.float32)
        y_assists = torch.tensor(y_train['assists'].values, dtype=torch.float32)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            X_tensor, y_points, y_rebounds, y_assists
        )
        
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Initialize model
        model = NBADeepModel(input_dim=X_train.shape[1])
        
        # Multi-task loss
        criterion_points = nn.MSELoss()
        criterion_rebounds = nn.MSELoss()
        criterion_assists = nn.MSELoss()
        
        # Task weights (can be learned)
        task_weights = {
            'points': 0.5,
            'rebounds': 0.25,
            'assists': 0.25
        }
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(100):
            epoch_loss = 0
            
            for batch_idx, (features, points, rebounds, assists) in enumerate(train_loader):
                optimizer.zero_grad()
                
                outputs = model(features)
                
                # Calculate losses for each task
                loss_points = criterion_points(outputs['points'].squeeze(), points)
                loss_rebounds = criterion_rebounds(outputs['rebounds'].squeeze(), rebounds)
                loss_assists = criterion_assists(outputs['assists'].squeeze(), assists)
                
                # Weighted multi-task loss
                total_loss = (task_weights['points'] * loss_points +
                            task_weights['rebounds'] * loss_rebounds +
                            task_weights['assists'] * loss_assists)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {epoch_loss/len(train_loader)}")
        
        return model
    
    def _initialize_online_models(self):
        """Initialize online learning models"""
        from river import linear_model, tree, ensemble
        
        self.online_models = {
            'sgd': river_preprocessing.StandardScaler() | linear_model.SGDRegressor(),
            'hoeffding_tree': river_preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(),
            'adaptive_random_forest': river_preprocessing.StandardScaler() | 
                                    river_ensemble.AdaptiveRandomForestRegressor(n_models=10)
        }
        
        self.online_metrics = {}
    
    def _train_deep_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         params: Dict[str, Any]) -> float:
        """Train deep learning model for Optuna"""
        # Convert to tensors
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            dataset, 
            batch_size=params['batch_size'],
            shuffle=True
        )
        
        # Initialize model
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], params['hidden_dims'][0]),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(params['hidden_dims'][0], params['hidden_dims'][1]),
            nn.ReLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(params['hidden_dims'][1], params['hidden_dims'][2]),
            nn.ReLU(),
            nn.Linear(params['hidden_dims'][2], 1)
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train
        model.train()
        for epoch in range(params['epochs']):
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).squeeze()
            mae = torch.mean(torch.abs(predictions - y_tensor)).item()
        
        return mae
    
    def _validate_pytorch_model(self, model: nn.Module, val_loader: DataLoader,
                              criterion: nn.Module, device: torch.device) -> float:
        """Validate PyTorch model"""
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs['points'], targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)


# Example custom loss functions
def quantile_loss(y_true, y_pred, quantile=0.5):
    """Quantile loss for prediction intervals"""
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def focal_mae_loss(y_true, y_pred, gamma=2.0):
    """Focal MAE loss that focuses on hard examples"""
    mae = np.abs(y_true - y_pred)
    focal_weight = mae ** gamma
    return np.mean(focal_weight * mae)


if __name__ == "__main__":
    config = {
        'mlflow_uri': 'http://localhost:5000',
        'ray_cpus': 8,
        'dask_scheduler': 'localhost:8786',
        'learning_rate': 0.001,
        'epochs': 100,
        'n_trials': 100
    }
    
    pipeline = MLPipeline(config)
    
    # Example usage
    # Load your data
    X_train = pd.DataFrame(np.random.randn(1000, 20))
    y_train = pd.Series(np.random.randn(1000))
    
    # Automated hyperparameter tuning
    best_params = pipeline.automated_hyperparameter_tuning(X_train, y_train)
    
    # Multi-task learning
    y_multi = pd.DataFrame({
        'points': np.random.randn(1000),
        'rebounds': np.random.randn(1000),
        'assists': np.random.randn(1000)
    })
    
    multi_task_model = pipeline.multi_task_learning(X_train, y_multi)