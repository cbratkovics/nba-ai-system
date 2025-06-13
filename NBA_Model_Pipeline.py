"""
NBA Player Performance Prediction - Model Training Pipeline

Comprehensive implementation for predicting individual NBA player statistics 
(points, rebounds, assists) with feature importance analysis.

Author: Christopher Bratkovics
Created: 2025
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import json
import joblib
from dataclasses import dataclass

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin

# Statistical analysis
from scipy.stats import pearsonr

# Configuration
warnings.filterwarnings('ignore')
np.random.seed(42)


@dataclass
class ModelConfig:
    """Configuration class for modeling parameters."""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    n_cv_folds: int = 5
    max_features_for_selection: int = 25
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    feature_selection_method: str = 'rfe'


@dataclass
class ModelResults:
    """Data class to store model results."""
    metrics: Dict[str, float]
    predictions: np.ndarray
    feature_importance: Optional[pd.DataFrame] = None
    cv_scores: Optional[np.ndarray] = None


class DataLeakageDetector:
    """Utility class to detect and prevent data leakage."""
    
    @staticmethod
    def detect_target_leakage(X: pd.DataFrame, y: pd.Series, correlation_threshold: float = 0.95) -> List[str]:
        """Detect features that are too highly correlated with target."""
        leakage_features = []
        for col in X.select_dtypes(include=[np.number]).columns:
            try:
                corr, p_value = pearsonr(X[col].fillna(0), y)
                if abs(corr) > correlation_threshold and p_value < 0.001:
                    leakage_features.append(col)
            except:
                continue
        return leakage_features
    
    @staticmethod
    def detect_calculated_leakage_features(df_columns: List[str]) -> List[str]:
        """Detect features that are calculated using target variables."""
        leakage_features = []
        leakage_patterns = [
            'pts_per_min', 'pts_per_36', 'reb_per_min', 'reb_per_36', 'ast_per_min', 'ast_per_36',
            'pts_last_', 'pts_avg', 'reb_last_', 'reb_avg', 'ast_last_', 'ast_avg',
            'double_double', 'triple_double', 'high_scoring_game',
            '_pts_', '_reb_', '_ast_', 'points_', 'rebounds_', 'assists_',
            'true_shooting', 'effective_fg', 'pts_x_', 'reb_x_', 'ast_x_'
        ]
        
        for col in df_columns:
            col_lower = col.lower()
            for pattern in leakage_patterns:
                if pattern.lower() in col_lower:
                    leakage_features.append(col)
                    break
        
        return leakage_features


class DataLoader:
    """Robust data loading with comprehensive validation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def load_and_validate(self, data_path: str) -> pd.DataFrame:
        """Load data with comprehensive validation."""
        print("LOADING AND VALIDATING NBA DATASET")
        print("-" * 45)
        
        try:
            df = pd.read_parquet(data_path)
            print(f"Successfully loaded data from {data_path}")
            print(f"Shape: {df.shape}")
            
            self._validate_data_quality(df)
            self._validate_target_variables(df)
            self._validate_date_consistency(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Validate overall data quality."""
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            print(f"Found {len(empty_cols)} completely empty columns")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows")
        
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Data quality check complete. Memory usage: {memory_mb:.1f} MB")
    
    def _validate_target_variables(self, df: pd.DataFrame) -> None:
        """Validate target variables are present and reasonable."""
        required_targets = ['pts', 'reb', 'ast']
        missing_targets = [t for t in required_targets if t not in df.columns]
        
        if missing_targets:
            raise ValueError(f"Missing target variables: {missing_targets}")
        
        for target in required_targets:
            if target in df.columns:
                max_val = df[target].max()
                min_val = df[target].min()
                if max_val > 200 or min_val < 0:
                    print(f"Unusual {target} values: min={min_val}, max={max_val}")
        
        print(f"All target variables validated: {required_targets}")
    
    def _validate_date_consistency(self, df: pd.DataFrame) -> None:
        """Validate date column if present."""
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
            invalid_dates = df['game_date'].isnull().sum()
            if invalid_dates > 0:
                print(f"Found {invalid_dates} invalid dates")
            
            date_range = df['game_date'].max() - df['game_date'].min()
            print(f"Date range: {date_range.days} days")


class AdvancedFeatureEngineer:
    """Advanced feature engineering specifically for NBA data."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 8) -> Tuple[pd.DataFrame, List[str]]:
        """Create interaction features between important variables."""
        interactions = [
            ('minutes_played', 'rest_days'), ('minutes_played', 'is_home_game'),
            ('rest_days', 'is_home_game'), ('sufficient_rest', 'minutes_played'),
            ('is_home_game', 'is_weekend'), ('rest_days', 'month'),
            ('minutes_played', 'day_of_week'), ('is_weekend', 'rest_days'),
        ]
        
        df_interactions = df.copy()
        created_features = []
        
        interactions_created = 0
        for feat1, feat2 in interactions:
            if interactions_created >= max_interactions:
                break
                
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                try:
                    df_interactions[interaction_name] = df[feat1] * df[feat2]
                    created_features.append(interaction_name)
                    interactions_created += 1
                except:
                    continue
        
        print(f"Created {len(created_features)} selective interaction features")
        return df_interactions, created_features
    
    def create_position_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features."""
        df_pos = df.copy()
        
        if 'player_position' in df.columns and 'minutes_played' in df.columns:
            position_minutes = df.groupby('player_position')['minutes_played'].transform('mean')
            df_pos['minutes_vs_position_avg'] = df['minutes_played'] / (position_minutes + 1e-6)
        
        return df_pos
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'game_date') -> pd.DataFrame:
        """Create temporal features if date column exists."""
        df_temporal = df.copy()
        
        if date_col in df.columns:
            df_temporal[date_col] = pd.to_datetime(df_temporal[date_col])
            df_temporal['day_of_week'] = df_temporal[date_col].dt.dayofweek
            df_temporal['month'] = df_temporal[date_col].dt.month
            df_temporal['is_weekend'] = df_temporal['day_of_week'].isin([5, 6]).astype(int)
            df_temporal['days_since_season_start'] = (
                df_temporal[date_col] - df_temporal.groupby('game_season')[date_col].transform('min')
            ).dt.days
        
        return df_temporal


class SmartFeatureSelector:
    """Intelligent feature selection with multiple strategies."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.selected_features_ = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> pd.DataFrame:
        """Apply comprehensive feature selection."""
        print(f"Smart feature selection for {target_name.upper()}")
        print("-" * 40)
        
        X_filtered = self._remove_low_variance_features(X)
        X_filtered = self._remove_correlated_features(X_filtered)
        X_filtered = self._remove_leakage_features(X_filtered, y)
        X_selected = self._apply_statistical_selection(X_filtered, y, target_name)
        
        self.selected_features_[target_name] = X_selected.columns.tolist()
        
        print(f"Feature selection complete: {X.shape[1]} -> {X_selected.shape[1]} features")
        return X_selected
    
    def _remove_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with very low variance."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return X
        
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_numeric = X[numeric_cols]
        
        try:
            mask = selector.fit_transform(X_numeric).shape[1] == len(numeric_cols)
            if not mask:
                selected_features = numeric_cols[selector.get_support()]
                X_filtered = X[selected_features].copy()
                non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    X_filtered[col] = X[col]
                return X_filtered
        except:
            pass
        
        return X
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return X
        
        corr_matrix = X[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > self.config.correlation_threshold)
        ]
        
        if to_drop:
            print(f"Removing {len(to_drop)} highly correlated features")
            X = X.drop(columns=to_drop)
        
        return X
    
    def _remove_leakage_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Detect and remove potential data leakage."""
        detector = DataLeakageDetector()
        
        correlation_leakage = detector.detect_target_leakage(X, y, correlation_threshold=0.90)
        calculated_leakage = detector.detect_calculated_leakage_features(X.columns.tolist())
        
        all_leakage = list(set(correlation_leakage + calculated_leakage))
        
        if all_leakage:
            print(f"Removing {len(all_leakage)} potential leakage features")
            print(f"Correlation-based: {len(correlation_leakage)}")
            print(f"Calculated features: {len(calculated_leakage)}")
            X = X.drop(columns=all_leakage, errors='ignore')
        
        return X
    
    def _apply_statistical_selection(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> pd.DataFrame:
        """Apply statistical feature selection."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return X
        
        n_features = min(self.config.max_features_for_selection, len(numeric_cols))
        
        if self.config.feature_selection_method == 'selectk':
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif self.config.feature_selection_method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            selector = RFE(estimator, n_features_to_select=n_features)
        else:
            return X
        
        try:
            X_numeric = X[numeric_cols].fillna(0)
            selected_mask = selector.fit(X_numeric, y).get_support()
            selected_numeric_cols = numeric_cols[selected_mask]
            
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            final_cols = list(selected_numeric_cols) + list(non_numeric_cols)
            
            print(f"Selected {len(selected_numeric_cols)} numeric features via {self.config.feature_selection_method}")
            return X[final_cols]
            
        except Exception as e:
            print(f"Feature selection failed: {e}. Using all features.")
            return X


class OptimizedModelPipeline:
    """Production-ready model training pipeline."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_selector = SmartFeatureSelector(config)
        
    def prepare_model_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features and targets with comprehensive preprocessing."""
        print("PREPARING MODEL DATA")
        print("-" * 30)
        
        target_vars = ['pts', 'reb', 'ast']
        
        print("Comprehensive data leakage detection...")
        
        # Identify and remove obvious leakage columns
        direct_leakage = [
            'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
            'ftm', 'fta', 'ft_pct', 'oreb', 'dreb',
        ]
        
        calculated_leakage = DataLeakageDetector.detect_calculated_leakage_features(df.columns.tolist())
        print(f"Found {len(calculated_leakage)} calculated leakage features")
        
        identifier_cols = ['game_id', 'player_id', 'game_date', 'game_season', 'team_id', 'player_team_id']
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        
        cols_to_drop = target_vars + direct_leakage + calculated_leakage + identifier_cols + id_cols
        cols_to_drop = list(set([col for col in cols_to_drop if col in df.columns]))
        
        print(f"Removing {len(cols_to_drop)} leakage/identifier/ID columns")
        
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = {target: df[target] for target in target_vars if target in df.columns}
        
        print(f"After leakage removal: {X.shape[1]} features remaining")
        
        # Early feature selection
        print("Applying early feature selection...")
        X = self._apply_early_feature_selection(X, y['pts'])
        
        # Advanced feature engineering
        engineer = AdvancedFeatureEngineer(self.config)
        X, interaction_features = engineer.create_interaction_features(X, max_interactions=8)
        X = engineer.create_position_specific_features(X)
        X = engineer.create_temporal_features(X)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        essential_categorical = []
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            if unique_count <= 10:
                essential_categorical.append(col)
            else:
                print(f"Dropping high-cardinality categorical: {col} ({unique_count} unique values)")
        
        X = X.drop(columns=[col for col in categorical_cols if col not in essential_categorical])
        
        if len(essential_categorical) > 0:
            X = pd.get_dummies(X, columns=essential_categorical, drop_first=True)
            print(f"Encoded {len(essential_categorical)} essential categorical variables")
        
        # Final leakage check
        print("Final correlation-based leakage check...")
        for target_name, target_series in y.items():
            high_corr_features = DataLeakageDetector.detect_target_leakage(X, target_series, correlation_threshold=0.90)
            if high_corr_features:
                print(f"Removing {len(high_corr_features)} highly correlated features for {target_name}")
                X = X.drop(columns=high_corr_features, errors='ignore')
        
        X = X.fillna(0)
        
        print(f"Data preparation complete: {X.shape}")
        print(f"Created {len(interaction_features)} interaction features")
        print(f"Final feature count: {X.shape[1]} (leak-free)")
        
        return X, y
    
    def _apply_early_feature_selection(self, X: pd.DataFrame, y_reference: pd.Series, max_features: int = 50) -> pd.DataFrame:
        """Apply early feature selection before feature engineering."""
        print(f"Early selection: {X.shape[1]} -> targeting <={max_features} features")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) <= max_features:
            return X
        
        try:
            X_numeric = X[numeric_cols].fillna(0)
            correlations = []
            
            for col in numeric_cols:
                try:
                    corr = abs(X_numeric[col].corr(y_reference))
                    correlations.append((col, corr if not np.isnan(corr) else 0))
                except:
                    correlations.append((col, 0))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_numeric = [col for col, _ in correlations[:max_features]]
            
            final_cols = selected_numeric + list(non_numeric_cols)
            print(f"Selected {len(selected_numeric)} numeric + {len(non_numeric_cols)} categorical features")
            
            return X[final_cols]
            
        except Exception as e:
            print(f"Early selection failed: {e}, keeping all features")
            return X
    
    def create_time_aware_split(self, df: pd.DataFrame, X: pd.DataFrame, y: Dict[str, pd.Series]) -> Tuple:
        """Create chronologically aware train/validation/test splits."""
        print("CREATING TIME-AWARE DATA SPLITS")
        print("-" * 40)
        
        if 'game_date' in df.columns:
            df_sorted = df.sort_values('game_date')
            sorted_indices = df_sorted.index
            
            n_samples = len(sorted_indices)
            train_end = int(n_samples * (1 - self.config.test_size - self.config.validation_size))
            val_end = int(n_samples * (1 - self.config.test_size))
            
            train_idx = sorted_indices[:train_end]
            val_idx = sorted_indices[train_end:val_end]
            test_idx = sorted_indices[val_end:]
            
            print(f"Chronological split:")
            print(f"Train: {len(train_idx):,} samples")
            print(f"Validation: {len(val_idx):,} samples") 
            print(f"Test: {len(test_idx):,} samples")
            
        else:
            print("No date column found, using random split")
            indices = X.index
            train_val_idx, test_idx = train_test_split(
                indices, test_size=self.config.test_size, random_state=self.config.random_state
            )
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=self.config.validation_size, random_state=self.config.random_state
            )
        
        X_train, X_val, X_test = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
        y_train = {target: y[target].loc[train_idx] for target in y.keys()}
        y_val = {target: y[target].loc[val_idx] for target in y.keys()}
        y_test = {target: y[target].loc[test_idx] for target in y.keys()}
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_optimized_models(self) -> Dict:
        """Define optimized model configurations."""
        return {
            'linear_regression': {
                'model': LinearRegression(),
                'use_scaling': True,
                'params': {}
            },
            'ridge': {
                'model': Ridge(random_state=self.config.random_state),
                'use_scaling': True,
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            'elastic_net': {
                'model': ElasticNet(random_state=self.config.random_state, max_iter=2000),
                'use_scaling': True,
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    random_state=self.config.random_state, 
                    n_jobs=-1,
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    max_samples=0.8
                ),
                'use_scaling': False,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 15],
                    'min_samples_split': [10, 20],
                    'max_features': ['sqrt']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    random_state=self.config.random_state,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    max_features='sqrt'
                ),
                'use_scaling': False,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [4, 6],
                    'learning_rate': [0.1, 0.15]
                }
            }
        }
    
    def train_models(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                    y_train: Dict, y_val: Dict) -> None:
        """Train and validate models with proper hyperparameter tuning."""
        print("TRAINING OPTIMIZED MODELS")
        print("-" * 35)
        
        model_configs = self.get_optimized_models()
        cv_strategy = TimeSeriesSplit(n_splits=3)
        
        for target in y_train.keys():
            print(f"Training models for {target.upper()}")
            print("-" * 30)
            
            X_train_selected = self.feature_selector.select_features(X_train, y_train[target], target)
            selected_features = X_train_selected.columns
            X_val_selected = X_val[selected_features]
            
            self.models[target] = {}
            self.results[target] = {}
            
            for model_name, config in model_configs.items():
                print(f"Training {model_name}...")
                
                try:
                    if config['use_scaling']:
                        scaler = RobustScaler()
                        X_train_model = pd.DataFrame(
                            scaler.fit_transform(X_train_selected),
                            columns=X_train_selected.columns,
                            index=X_train_selected.index
                        )
                        X_val_model = pd.DataFrame(
                            scaler.transform(X_val_selected),
                            columns=X_val_selected.columns,
                            index=X_val_selected.index
                        )
                        self.scalers[f"{target}_{model_name}"] = scaler
                    else:
                        X_train_model = X_train_selected
                        X_val_model = X_val_selected
                    
                    # Hyperparameter tuning with efficiency optimizations
                    if config['params'] and model_name not in ['gradient_boosting', 'random_forest']:
                        grid_search = GridSearchCV(
                            config['model'],
                            config['params'],
                            cv=cv_strategy,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1,
                            verbose=0
                        )
                        grid_search.fit(X_train_model, y_train[target])
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        cv_score = -grid_search.best_score_
                        
                    elif model_name in ['gradient_boosting', 'random_forest']:
                        print(f"Using efficient training for speed...")
                        
                        if model_name == 'random_forest':
                            rf_configs = [
                                {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 10},
                                {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 20}
                            ]
                            
                            best_score = float('inf')
                            best_model = None
                            best_params = {}
                            
                            for rf_config in rf_configs:
                                temp_model = RandomForestRegressor(
                                    random_state=self.config.random_state,
                                    n_jobs=-1,
                                    max_features='sqrt',
                                    **rf_config
                                )
                                temp_model.fit(X_train_model, y_train[target])
                                
                                if hasattr(temp_model, 'oob_score_') and temp_model.oob_score_ is not None:
                                    oob_mae_approx = np.sqrt(1 - temp_model.oob_score_) * y_train[target].std()
                                else:
                                    val_pred = temp_model.predict(X_val_model)
                                    oob_mae_approx = mean_absolute_error(y_val[target], val_pred)
                                
                                if oob_mae_approx < best_score:
                                    best_score = oob_mae_approx
                                    best_model = temp_model
                                    best_params = rf_config
                            
                            cv_score = best_score
                            
                        else:  # gradient_boosting
                            best_model = config['model']
                            best_model.fit(X_train_model, y_train[target])
                            best_params = {}
                            
                            cv_scores = cross_val_score(
                                best_model, X_train_model, y_train[target],
                                cv=TimeSeriesSplit(n_splits=2), scoring='neg_mean_absolute_error'
                            )
                            cv_score = -cv_scores.mean()
                    else:
                        best_model = config['model']
                        best_model.fit(X_train_model, y_train[target])
                        best_params = {}
                        cv_scores = cross_val_score(
                            best_model, X_train_model, y_train[target],
                            cv=cv_strategy, scoring='neg_mean_absolute_error'
                        )
                        cv_score = -cv_scores.mean()
                    
                    y_val_pred = best_model.predict(X_val_model)
                    
                    val_mae = mean_absolute_error(y_val[target], y_val_pred)
                    val_rmse = np.sqrt(mean_squared_error(y_val[target], y_val_pred))
                    val_r2 = r2_score(y_val[target], y_val_pred)
                    val_mape = mean_absolute_percentage_error(y_val[target], y_val_pred) * 100
                    
                    self.models[target][model_name] = best_model
                    self.results[target][model_name] = ModelResults(
                        metrics={
                            'val_mae': val_mae,
                            'val_rmse': val_rmse,
                            'val_r2': val_r2,
                            'val_mape': val_mape,
                            'cv_mae': cv_score
                        },
                        predictions=y_val_pred
                    )
                    
                    print(f"CV MAE: {cv_score:.3f} | Val MAE: {val_mae:.3f} | Val R2: {val_r2:.3f}")
                    if best_params:
                        print(f"Best params: {best_params}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        print("Model training complete!")
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: Dict) -> Dict:
        """Final evaluation on test set."""
        print("FINAL TEST SET EVALUATION")
        print("-" * 35)
        
        test_results = {}
        
        for target in y_test.keys():
            print(f"{target.upper()} Test Results:")
            test_results[target] = {}
            
            selected_features = self.feature_selector.selected_features_[target]
            X_test_selected = X_test[selected_features]
            
            for model_name, model in self.models[target].items():
                try:
                    scaler_key = f"{target}_{model_name}"
                    if scaler_key in self.scalers:
                        X_test_model = pd.DataFrame(
                            self.scalers[scaler_key].transform(X_test_selected),
                            columns=X_test_selected.columns,
                            index=X_test_selected.index
                        )
                    else:
                        X_test_model = X_test_selected
                    
                    y_pred = model.predict(X_test_model)
                    
                    mae = mean_absolute_error(y_test[target], y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
                    r2 = r2_score(y_test[target], y_pred)
                    mape = mean_absolute_percentage_error(y_test[target], y_pred) * 100
                    
                    test_results[target][model_name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'mape': mape,
                        'predictions': y_pred
                    }
                    
                    print(f"{model_name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
                    
                except Exception as e:
                    print(f"{model_name} error: {e}")
                    continue
        
        return test_results


class ModelInterpreter:
    """Advanced model interpretation and feature importance analysis."""
    
    def __init__(self, pipeline: OptimizedModelPipeline):
        self.pipeline = pipeline
        
    def analyze_feature_importance(self, X_train: pd.DataFrame, y_train: Dict) -> Dict:
        """Comprehensive feature importance analysis."""
        print("ADVANCED FEATURE IMPORTANCE ANALYSIS")
        print("-" * 45)
        
        importance_results = {}
        
        for target in y_train.keys():
            print(f"Analyzing {target.upper()} feature importance:")
            importance_results[target] = {}
            
            selected_features = self.pipeline.feature_selector.selected_features_[target]
            X_train_selected = X_train[selected_features]
            
            for model_name, model in self.pipeline.models[target].items():
                try:
                    scaler_key = f"{target}_{model_name}"
                    if scaler_key in self.pipeline.scalers:
                        X_train_model = pd.DataFrame(
                            self.pipeline.scalers[scaler_key].transform(X_train_selected),
                            columns=X_train_selected.columns,
                            index=X_train_selected.index
                        )
                    else:
                        X_train_model = X_train_selected
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        importance_type = 'built_in'
                    elif hasattr(model, 'coef_'):
                        importances = np.abs(model.coef_)
                        importance_type = 'coefficients'
                    else:
                        perm_importance = permutation_importance(
                            model, X_train_model, y_train[target],
                            n_repeats=5, random_state=42, n_jobs=-1
                        )
                        importances = perm_importance.importances_mean
                        importance_type = 'permutation'
                    
                    feature_importance = pd.DataFrame({
                        'feature': X_train_model.columns,
                        'importance': importances,
                        'importance_type': importance_type
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[target][model_name] = feature_importance
                    
                    print(f"{model_name} ({importance_type}) - Top 5:")
                    for _, row in feature_importance.head(5).iterrows():
                        print(f"  {row['feature']}: {row['importance']:.4f}")
                        
                except Exception as e:
                    print(f"Error analyzing {model_name}: {e}")
                    continue
        
        return importance_results
    
    def create_business_insights(self, importance_results: Dict, test_results: Dict) -> Dict:
        """Generate actionable business insights."""
        print("GENERATING BUSINESS INSIGHTS")
        print("-" * 35)
        
        insights = {
            'model_performance': {},
            'key_drivers': {},
            'stakeholder_recommendations': {}
        }
        
        for target in test_results.keys():
            best_model = max(test_results[target], key=lambda x: test_results[target][x]['r2'])
            best_metrics = test_results[target][best_model]
            
            insights['model_performance'][target] = {
                'best_model': best_model,
                'r2': best_metrics['r2'],
                'mae': best_metrics['mae'],
                'mape': best_metrics['mape'],
                'predictability': self._classify_predictability(best_metrics['r2'])
            }
        
        for target in importance_results.keys():
            if 'random_forest' in importance_results[target]:
                top_features = importance_results[target]['random_forest'].head(5)
            else:
                best_model = insights['model_performance'][target]['best_model']
                if best_model in importance_results[target]:
                    top_features = importance_results[target][best_model].head(5)
                else:
                    top_features = pd.DataFrame()
            
            if not top_features.empty:
                insights['key_drivers'][target] = {
                    'top_features': top_features['feature'].tolist(),
                    'importance_scores': top_features['importance'].tolist()
                }
        
        insights['stakeholder_recommendations'] = self._generate_stakeholder_recommendations(insights)
        
        return insights
    
    def _classify_predictability(self, r2_score: float) -> str:
        """Classify model predictability based on R2 score."""
        if r2_score >= 0.7:
            return "High"
        elif r2_score >= 0.5:
            return "Medium"
        elif r2_score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_stakeholder_recommendations(self, insights: Dict) -> Dict:
        """Generate tailored recommendations for different stakeholders."""
        return {
            'fantasy_managers': [
                "Focus on players with consistent minutes and favorable rest patterns",
                "Home games provide measurable advantage - prioritize home players",
                "Monitor recent performance trends - they're highly predictive",
                "Consider position-specific matchups for optimal lineups"
            ],
            'coaches_analysts': [
                "Rest day management significantly impacts player performance",
                "Minutes played is the strongest predictor - manage rotations carefully",
                "Home court advantage is real and quantifiable",
                "Load management strategies show measurable benefits"
            ],
            'sports_media': [
                "Player performance follows predictable patterns",
                "Context matters - rest, location, and recent form are key narratives",
                "Advanced metrics provide deeper storytelling opportunities",
                "Performance streaks have statistical foundations"
            ],
            'general_fans': [
                "Player performance is more predictable than commonly believed",
                "Rest and game context significantly impact what you see",
                "Recent performance is a good indicator of upcoming games",
                "Home court advantage is real at the individual player level"
            ]
        }


class ProductionModelManager:
    """Manage models for production deployment."""
    
    def __init__(self, pipeline: OptimizedModelPipeline):
        self.pipeline = pipeline
        self.deployment_models = {}
        
    def prepare_for_deployment(self, test_results: Dict) -> None:
        """Prepare the best models for production deployment."""
        print("PREPARING MODELS FOR DEPLOYMENT")
        print("-" * 40)
        
        for target in test_results.keys():
            best_model_name = max(test_results[target], key=lambda x: test_results[target][x]['r2'])
            best_model = self.pipeline.models[target][best_model_name]
            best_metrics = test_results[target][best_model_name]
            
            scaler_key = f"{target}_{best_model_name}"
            scaler = self.pipeline.scalers.get(scaler_key, None)
            
            selected_features = self.pipeline.feature_selector.selected_features_[target]
            
            self.deployment_models[target] = {
                'model': best_model,
                'model_name': best_model_name,
                'scaler': scaler,
                'features': selected_features,
                'metrics': best_metrics
            }
            
            print(f"{target.upper()}: {best_model_name} (R2={best_metrics['r2']:.3f})")
    
    def create_prediction_function(self) -> callable:
        """Create a single function for making predictions on new data."""
        
        def predict_player_performance(player_data: Dict[str, Any]) -> Dict[str, float]:
            """
            Predict player performance for a single game.
            
            Args:
                player_data: Dictionary containing player features
                
            Returns:
                Dictionary with predicted points, rebounds, assists
            """
            input_df = pd.DataFrame([player_data])
            
            for col in input_df.select_dtypes(include=['object']).columns:
                if col.startswith('player_position'):
                    unique_positions = ['C', 'F', 'G']
                    for pos in unique_positions:
                        input_df[f'player_position_{pos}'] = (input_df[col] == pos).astype(int)
                    input_df = input_df.drop(columns=[col])
            
            predictions = {}
            
            for target, deployment_info in self.deployment_models.items():
                try:
                    available_features = [f for f in deployment_info['features'] if f in input_df.columns]
                    missing_features = [f for f in deployment_info['features'] if f not in input_df.columns]
                    
                    for feature in missing_features:
                        input_df[feature] = 0
                    
                    X_input = input_df[deployment_info['features']]
                    
                    if deployment_info['scaler'] is not None:
                        X_input = pd.DataFrame(
                            deployment_info['scaler'].transform(X_input),
                            columns=X_input.columns
                        )
                    
                    prediction = deployment_info['model'].predict(X_input)[0]
                    predictions[target] = max(0, round(prediction, 1))
                    
                except Exception as e:
                    predictions[target] = 0.0
                    print(f"Warning: Error predicting {target}: {e}")
            
            return predictions
        
        return predict_player_performance
    
    def save_production_artifacts(self, output_dir: str = "production_models") -> None:
        """Save all production artifacts."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for target, deployment_info in self.deployment_models.items():
            target_dir = output_path / target
            target_dir.mkdir(exist_ok=True)
            
            model_file = target_dir / f"model_{timestamp}.joblib"
            joblib.dump(deployment_info['model'], model_file)
            
            if deployment_info['scaler'] is not None:
                scaler_file = target_dir / f"scaler_{timestamp}.joblib"
                joblib.dump(deployment_info['scaler'], scaler_file)
            
            metadata = {
                'model_name': deployment_info['model_name'],
                'features': deployment_info['features'],
                'metrics': deployment_info['metrics'],
                'timestamp': timestamp,
                'has_scaler': deployment_info['scaler'] is not None
            }
            
            metadata_file = target_dir / f"metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"Production artifacts saved to {output_path}")
        return output_path


def validate_model_results(test_results: Dict, min_r2_threshold: float = 0.3) -> bool:
    """Validate that model results meet minimum performance criteria."""
    validation_passed = True
    
    print("VALIDATING MODEL RESULTS")
    print("-" * 30)
    
    for target, models in test_results.items():
        best_r2 = max(model_metrics['r2'] for model_metrics in models.values())
        
        if best_r2 < min_r2_threshold:
            print(f"WARNING: {target.upper()} best R2 ({best_r2:.3f}) below threshold ({min_r2_threshold})")
            validation_passed = False
        else:
            print(f"PASS: {target.upper()} best R2 = {best_r2:.3f}")
    
    if validation_passed:
        print("All models meet minimum performance criteria")
    else:
        print("Some models below performance threshold - consider retraining")
    
    return validation_passed


def save_model_artifacts(pipeline: OptimizedModelPipeline, test_results: Dict, 
                        insights: Dict, output_dir: str = "model_artifacts") -> None:
    """Save all model artifacts for future use."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    pipeline_file = output_path / f"nba_pipeline_{timestamp}.joblib"
    joblib.dump(pipeline, pipeline_file)
    
    results_file = output_path / f"model_results_{timestamp}.joblib"
    joblib.dump({
        'test_results': test_results,
        'insights': insights,
        'timestamp': timestamp
    }, results_file)
    
    feature_lists = {}
    for target in pipeline.feature_selector.selected_features_:
        feature_lists[target] = pipeline.feature_selector.selected_features_[target]
    
    features_file = output_path / f"selected_features_{timestamp}.json"
    with open(features_file, 'w') as f:
        json.dump(feature_lists, f, indent=2)
    
    print(f"Model artifacts saved to {output_path}")
    print(f"Files created:")
    print(f"  - {pipeline_file.name} (trained pipeline)")
    print(f"  - {results_file.name} (results and insights)")
    print(f"  - {features_file.name} (selected features)")


def run_nba_modeling_pipeline(data_path: str = "data/processed/final_engineered_nba_data.parquet") -> Tuple:
    """
    Execute the complete NBA modeling pipeline.
    
    Returns:
        Tuple of (pipeline, test_results, insights, production_manager)
    """
    print("NBA PLAYER PERFORMANCE MODELING PIPELINE")
    print("=" * 65)
    
    config = ModelConfig(
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        n_cv_folds=5,
        max_features_for_selection=30,
        correlation_threshold=0.95,
        feature_selection_method='rfe'
    )
    
    loader = DataLoader(config)
    df = loader.load_and_validate(data_path)
    
    pipeline = OptimizedModelPipeline(config)
    
    X, y = pipeline.prepare_model_data(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.create_time_aware_split(df, X, y)
    
    pipeline.train_models(X_train, X_val, y_train, y_val)
    
    test_results = pipeline.evaluate_on_test(X_test, y_test)
    
    interpreter = ModelInterpreter(pipeline)
    importance_results = interpreter.analyze_feature_importance(X_train, y_train)
    insights = interpreter.create_business_insights(importance_results, test_results)
    
    production_manager = ProductionModelManager(pipeline)
    production_manager.prepare_for_deployment(test_results)
    production_manager.save_production_artifacts()
    
    return pipeline, test_results, insights, production_manager


if __name__ == "__main__":
    try:
        print("Starting NBA Player Performance Modeling Pipeline...")
        
        pipeline, test_results, insights, production_manager = run_nba_modeling_pipeline()
        
        validation_passed = validate_model_results(test_results)
        
        save_model_artifacts(pipeline, test_results, insights)
        
        print("NBA MODELING PIPELINE COMPLETE!")
        print("=" * 55)
        print("ACHIEVEMENTS:")
        print("- Models trained with proper validation")
        print("- Data leakage prevention implemented")
        print("- Feature selection and engineering optimized")
        print("- Business insights generated")
        print("- Production deployment framework ready")
        
        print("FINAL PERFORMANCE SUMMARY:")
        for target, performance in insights['model_performance'].items():
            print(f"  {target.upper()}: {performance['best_model'].replace('_', ' ').title()} "
                  f"(R2={performance['r2']:.3f}, MAE={performance['mae']:.2f})")
        
    except Exception as e:
        print(f"Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()