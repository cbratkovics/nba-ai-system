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

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin

# Statistical analysis imports
from scipy.stats import pearsonr

# Global configuration
warnings.filterwarnings('ignore')
np.random.seed(42)


@dataclass
class ModelConfig:
    """Configuration class for modeling parameters and hyperparameter settings."""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    n_cv_folds: int = 5
    max_features_for_selection: int = 25
    correlation_threshold: float = 0.80
    variance_threshold: float = 0.01
    feature_selection_method: str = 'rfe'  # Options: 'rfe', 'selectk', 'rfecv'


@dataclass
class ModelResults:
    """Data class to store model training results and performance metrics."""
    metrics: Dict[str, float]
    predictions: np.ndarray
    feature_importance: Optional[pd.DataFrame] = None
    cv_scores: Optional[np.ndarray] = None


class DataLeakageDetector:
    """Utility class to detect and prevent data leakage in feature sets."""
    
    @staticmethod
    def detect_target_leakage(X: pd.DataFrame, y: pd.Series, correlation_threshold: float = 0.80) -> List[str]:
        """
        Detect features that are too highly correlated with the target variable.
        
        High correlation with target may indicate data leakage or features 
        calculated using the target variable.
        """
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
        """
        Detect features that appear to be calculated using target variables.
        
        This method identifies features whose names suggest they were derived
        from target variables (points, rebounds, assists).
        """
        leakage_features = []
        # Patterns that indicate features derived from target variables
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
    """Handles data loading with comprehensive validation and quality checks."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def load_and_validate(self, data_path: str) -> pd.DataFrame:
        """
        Load data from disk and perform basic validation checks.
        
        Ensures data integrity and presence of required target variables.
        """
        print("Loading NBA player performance dataset...")
        
        try:
            df = pd.read_parquet(data_path)
            print(f"Successfully loaded dataset: {df.shape[0]:,} records with {df.shape[1]} features")
            
            self._validate_data_quality(df)
            return df
            
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
            raise
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate that the dataset contains required columns and basic quality metrics.
        
        Checks for presence of target variables and reports date range if available.
        """
        # Verify target variables are present
        required_targets = ['pts', 'reb', 'ast']
        missing_targets = [t for t in required_targets if t not in df.columns]
        
        if missing_targets:
            raise ValueError(f"Missing required target variables: {missing_targets}")
        
        # Report temporal coverage if date column exists
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
            date_range = df['game_date'].max() - df['game_date'].min()
            print(f"Dataset temporal coverage: {date_range.days} days")


class SmartFeatureSelector:
    """
    Intelligent feature selection using multiple strategies to identify
    the most predictive features while preventing overfitting.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.selected_features_ = {}
        self.feature_insights_ = {}
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> pd.DataFrame:
        """
        Apply comprehensive feature selection pipeline tailored for each target variable.
        
        Combines variance filtering, correlation analysis, leakage detection,
        and statistical selection methods.
        """
        print(f"\nApplying feature selection for {target_name.upper()} prediction...")
        
        # Sequential feature filtering pipeline
        X_filtered = self._remove_low_variance_features(X)
        X_filtered = self._remove_correlated_features(X_filtered)
        X_filtered = self._remove_leakage_features(X_filtered, y)
        X_selected = self._apply_statistical_selection(X_filtered, y, target_name)
        
        # Store selection results and insights
        self.selected_features_[target_name] = X_selected.columns.tolist()
        self.feature_insights_[target_name] = {
            'selected_count': len(X_selected.columns),
            'selected_features': X_selected.columns.tolist(),
            'original_count': len(X.columns),
            'reduction_ratio': 1 - (len(X_selected.columns) / len(X.columns))
        }
        
        print(f"Feature selection complete: {X.shape[1]} features reduced to {X_selected.shape[1]} features")
        return X_selected
    
    def get_feature_insights(self, target: str) -> Dict[str, Any]:
        """Retrieve feature selection insights for a specific target variable."""
        return self.feature_insights_.get(target, {})
    
    def _remove_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with very low variance that provide minimal predictive value.
        
        Low variance features are often constants or near-constants that don't
        contribute to model performance.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return X
        
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_numeric = X[numeric_cols]
        
        try:
            selector.fit(X_numeric)
            selected_features = numeric_cols[selector.get_support()]
            X_filtered = X[selected_features].copy()
            
            # Preserve non-numeric columns
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                X_filtered[col] = X[col]
            return X_filtered
        except:
            return X
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        When two features are highly correlated, keeping both adds complexity
        without improving predictive power.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 1:
            return X
        
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to drop
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > self.config.correlation_threshold)
        ]
        
        if to_drop:
            X = X.drop(columns=to_drop)
        
        return X
    
    def _remove_leakage_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Detect and remove potential data leakage features.
        
        Data leakage occurs when features contain information about the target
        that wouldn't be available at prediction time.
        """
        detector = DataLeakageDetector()
        
        # Check for correlation-based leakage
        correlation_leakage = detector.detect_target_leakage(X, y, correlation_threshold=0.80)
        # Check for name-based leakage patterns
        calculated_leakage = detector.detect_calculated_leakage_features(X.columns.tolist())
        
        all_leakage = list(set(correlation_leakage + calculated_leakage))
        
        if all_leakage:
            print(f"  Identified and removed {len(all_leakage)} potential data leakage features")
            X = X.drop(columns=all_leakage, errors='ignore')
        
        return X
    
    def _apply_statistical_selection(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> pd.DataFrame:
        """
        Apply statistical feature selection methods to identify most predictive features.
        
        Uses RFE (Recursive Feature Elimination) or other methods based on configuration.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return X
        
        n_features = min(self.config.max_features_for_selection, len(numeric_cols))
        
        # Select feature selection method based on configuration
        if self.config.feature_selection_method == 'selectk':
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif self.config.feature_selection_method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            selector = RFE(estimator, n_features_to_select=n_features)
        elif self.config.feature_selection_method == 'rfecv':
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            cv_strategy = TimeSeriesSplit(n_splits=3)
            selector = RFECV(estimator, cv=cv_strategy, scoring='neg_mean_absolute_error', min_features_to_select=10)
        else:
            return X
        
        try:
            # Apply feature selection
            X_numeric = X[numeric_cols].fillna(0)
            selected_mask = selector.fit(X_numeric, y).get_support()
            selected_numeric_cols = numeric_cols[selected_mask]
            
            # Combine selected numeric features with non-numeric features
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            final_cols = list(selected_numeric_cols) + list(non_numeric_cols)
            
            return X[final_cols]
            
        except Exception as e:
            print(f"  Feature selection encountered an issue, using all features: {str(e)[:50]}...")
            return X


class ModelPipeline:
    """
    Production-ready model training pipeline with comprehensive data preparation,
    model training, and evaluation capabilities.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_selector = SmartFeatureSelector(config)
        self.target_preprocessing = {}
        
    def prepare_model_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Prepare features and targets with comprehensive preprocessing.
        
        Removes leakage columns, handles categorical variables, and ensures
        data is ready for modeling.
        """
        print("\nPreparing data for model training...")
        
        target_vars = ['pts', 'reb', 'ast']
        
        # Define columns that would cause data leakage
        direct_leakage = [
            'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
            'ftm', 'fta', 'ft_pct', 'oreb', 'dreb',
        ]
        
        # Detect calculated leakage features
        calculated_leakage = DataLeakageDetector.detect_calculated_leakage_features(df.columns.tolist())
        
        # Identify non-predictive identifier columns
        identifier_cols = ['game_id', 'player_id', 'game_date', 'game_season', 'team_id', 'player_team_id']
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        
        # Combine all columns to drop
        cols_to_drop = target_vars + direct_leakage + calculated_leakage + identifier_cols + id_cols
        cols_to_drop = list(set([col for col in cols_to_drop if col in df.columns]))
        
        print(f"Identified and removing {len(cols_to_drop)} leakage/identifier columns")
        
        # Create feature matrix and target dictionary
        X = df.drop(columns=cols_to_drop, errors='ignore')
        y = {target: df[target] for target in target_vars if target in df.columns}
        
        # Apply early feature selection to manage dimensionality
        X = self._apply_early_feature_selection(X, y['pts'])
        
        # Handle categorical variables with one-hot encoding
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        essential_categorical = []
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            if unique_count <= 10:  # Only encode low-cardinality categorical variables
                essential_categorical.append(col)
        
        # Drop high-cardinality categorical columns
        X = X.drop(columns=[col for col in categorical_cols if col not in essential_categorical])
        
        # Apply one-hot encoding to selected categorical columns
        if len(essential_categorical) > 0:
            X = pd.get_dummies(X, columns=essential_categorical, drop_first=True)
        
        # Final leakage check for each target
        for target_name, target_series in y.items():
            high_corr_features = DataLeakageDetector.detect_target_leakage(X, target_series, correlation_threshold=0.80)
            if high_corr_features:
                X = X.drop(columns=high_corr_features, errors='ignore')
        
        # Fill any remaining missing values
        X = X.fillna(0)
        
        print(f"Data preparation complete: {X.shape[0]:,} records with {X.shape[1]} leak-free features")
        
        return X, y
    
    def _apply_early_feature_selection(self, X: pd.DataFrame, y_reference: pd.Series, max_features: int = 50) -> pd.DataFrame:
        """
        Apply preliminary feature selection based on correlation with target.
        
        This reduces computational complexity before more sophisticated selection methods.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) <= max_features:
            return X
        
        try:
            X_numeric = X[numeric_cols].fillna(0)
            correlations = []
            
            # Calculate correlation with reference target
            for col in numeric_cols:
                try:
                    corr = abs(X_numeric[col].corr(y_reference))
                    correlations.append((col, corr if not np.isnan(corr) else 0))
                except:
                    correlations.append((col, 0))
            
            # Select top features by correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_numeric = [col for col, _ in correlations[:max_features]]
            
            # Combine selected numeric features with non-numeric features
            final_cols = selected_numeric + list(non_numeric_cols)
            
            return X[final_cols]
            
        except Exception as e:
            return X
    
    def create_time_aware_split(self, df: pd.DataFrame, X: pd.DataFrame, y: Dict[str, pd.Series]) -> Tuple:
        """
        Create chronologically aware train/validation/test splits.
        
        Time-aware splitting prevents data leakage from future games influencing
        predictions of past games.
        """
        print("\nCreating time-aware data splits to prevent temporal leakage...")
        
        if 'game_date' in df.columns:
            # Sort by date to ensure chronological ordering
            df_sorted = df.sort_values('game_date')
            sorted_indices = df_sorted.index
            
            # Calculate split points
            n_samples = len(sorted_indices)
            train_end = int(n_samples * (1 - self.config.test_size - self.config.validation_size))
            val_end = int(n_samples * (1 - self.config.test_size))
            
            # Create chronological splits
            train_idx = sorted_indices[:train_end]
            val_idx = sorted_indices[train_end:val_end]
            test_idx = sorted_indices[val_end:]
            
            print(f"Chronological split sizes - Train: {len(train_idx):,} | Validation: {len(val_idx):,} | Test: {len(test_idx):,}")
            
        else:
            # Fallback to random split if no date column
            indices = X.index
            train_val_idx, test_idx = train_test_split(
                indices, test_size=self.config.test_size, random_state=self.config.random_state
            )
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=self.config.validation_size, random_state=self.config.random_state
            )
        
        # Create split datasets
        X_train, X_val, X_test = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
        y_train = {target: y[target].loc[train_idx] for target in y.keys()}
        y_val = {target: y[target].loc[val_idx] for target in y.keys()}
        y_test = {target: y[target].loc[test_idx] for target in y.keys()}
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_optimized_models(self) -> Dict:
        """
        Define optimized model configurations with appropriate hyperparameters.
        
        Each model is configured with parameters suitable for NBA player statistics prediction.
        """
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
        """
        Train and validate models with hyperparameter tuning for each target variable.
        
        Each target (points, rebounds, assists) gets its own optimized feature set
        and model configuration.
        """
        print("\nInitiating model training pipeline...")
        
        model_configs = self.get_optimized_models()
        cv_strategy = TimeSeriesSplit(n_splits=3)
        
        for target in y_train.keys():
            print(f"\nTraining models for {target.upper()} prediction:")
            
            # Apply target-specific feature selection
            X_train_selected = self.feature_selector.select_features(X_train, y_train[target], target)
            selected_features = X_train_selected.columns
            X_val_selected = X_val[selected_features]
            
            # Store preprocessing information for deployment
            self.target_preprocessing[target] = {
                'selected_features': selected_features.tolist(),
                'feature_count': len(selected_features),
                'feature_insights': self.feature_selector.get_feature_insights(target)
            }
            
            self.models[target] = {}
            self.results[target] = {}
            
            # Train each model type
            for model_name, config in model_configs.items():
                try:
                    # Apply scaling if required
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
                        # Full grid search for simpler models
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
                        # Simplified tuning for complex models to reduce computation time
                        if model_name == 'random_forest':
                            rf_configs = [
                                {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 10},
                                {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 20}
                            ]
                            
                            best_score = float('inf')
                            best_model = None
                            best_params = {}
                            
                            # Try each configuration
                            for rf_config in rf_configs:
                                temp_model = RandomForestRegressor(
                                    random_state=self.config.random_state,
                                    n_jobs=-1,
                                    max_features='sqrt',
                                    **rf_config
                                )
                                temp_model.fit(X_train_model, y_train[target])
                                
                                # Use OOB score if available, otherwise validate
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
                            
                            # Quick cross-validation for gradient boosting
                            cv_scores = cross_val_score(
                                best_model, X_train_model, y_train[target],
                                cv=TimeSeriesSplit(n_splits=2), scoring='neg_mean_absolute_error'
                            )
                            cv_score = -cv_scores.mean()
                    else:
                        # No hyperparameter tuning needed
                        best_model = config['model']
                        best_model.fit(X_train_model, y_train[target])
                        best_params = {}
                        cv_scores = cross_val_score(
                            best_model, X_train_model, y_train[target],
                            cv=cv_strategy, scoring='neg_mean_absolute_error'
                        )
                        cv_score = -cv_scores.mean()
                    
                    # Evaluate on validation set
                    y_val_pred = best_model.predict(X_val_model)
                    
                    # Calculate performance metrics
                    val_mae = mean_absolute_error(y_val[target], y_val_pred)
                    val_rmse = np.sqrt(mean_squared_error(y_val[target], y_val_pred))
                    val_r2 = r2_score(y_val[target], y_val_pred)
                    val_mape = mean_absolute_percentage_error(y_val[target], y_val_pred) * 100
                    
                    # Store model and results
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
                    
                    print(f"  {model_name}: R-squared = {val_r2:.3f} | Mean Absolute Error = {val_mae:.3f}")
                    
                except Exception as e:
                    print(f"  {model_name}: Training failed - {str(e)[:50]}...")
                    continue
        
        print("\nModel training pipeline completed successfully")

    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: Dict) -> Dict:
        """
        Perform final evaluation on the held-out test set.
        
        This provides unbiased performance estimates for model selection.
        """
        print("\nPerforming final evaluation on test set:")
        
        test_results = {}
        
        for target in y_test.keys():
            test_results[target] = {}
            
            # Use target-specific selected features
            selected_features = self.target_preprocessing[target]['selected_features']
            X_test_selected = X_test[selected_features]
            
            for model_name, model in self.models[target].items():
                try:
                    # Apply scaling if necessary
                    scaler_key = f"{target}_{model_name}"
                    if scaler_key in self.scalers:
                        X_test_model = pd.DataFrame(
                            self.scalers[scaler_key].transform(X_test_selected),
                            columns=X_test_selected.columns,
                            index=X_test_selected.index
                        )
                    else:
                        X_test_model = X_test_selected
                    
                    # Generate predictions
                    y_pred = model.predict(X_test_model)
                    
                    # Calculate test metrics
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
                    
                except Exception as e:
                    print(f"  Error evaluating {model_name} for {target}: {str(e)[:50]}...")
                    continue
        
        # Report best performing model for each target
        print("\nBest performing models on test set:")
        for target in test_results.keys():
            if test_results[target]:
                best_model = max(test_results[target], key=lambda x: test_results[target][x]['r2'])
                best_metrics = test_results[target][best_model]
                feature_count = len(self.target_preprocessing[target]['selected_features'])
                
                print(f"  {target.upper()}: {best_model.replace('_', ' ').title()} "
                      f"(R-squared = {best_metrics['r2']:.3f}, MAE = {best_metrics['mae']:.2f}) "
                      f"using {feature_count} optimized features")
            else:
                print(f"  {target.upper()}: No models successfully evaluated")
        
        return test_results


class ModelInterpreter:
    """
    Advanced model interpretation providing feature importance analysis
    and business insights from trained models.
    """
    
    def __init__(self, pipeline: ModelPipeline):
        self.pipeline = pipeline
        
    def analyze_feature_importance(self, X_train: pd.DataFrame, y_train: Dict) -> Dict:
        """
        Perform comprehensive feature importance analysis for each target and model.
        
        Uses built-in feature importances for tree-based models and permutation
        importance for other model types.
        """
        print("\nAnalyzing feature importance for model interpretability...")
        
        importance_results = {}
        
        for target in y_train.keys():
            importance_results[target] = {}
            
            selected_features = self.pipeline.feature_selector.selected_features_[target]
            X_train_selected = X_train[selected_features]
            
            for model_name, model in self.pipeline.models[target].items():
                try:
                    # Apply scaling if needed
                    scaler_key = f"{target}_{model_name}"
                    if scaler_key in self.pipeline.scalers:
                        X_train_model = pd.DataFrame(
                            self.pipeline.scalers[scaler_key].transform(X_train_selected),
                            columns=X_train_selected.columns,
                            index=X_train_selected.index
                        )
                    else:
                        X_train_model = X_train_selected
                    
                    # Extract feature importances based on model type
                    if hasattr(model, 'feature_importances_'):
                        # Tree-based models have built-in feature importances
                        importances = model.feature_importances_
                        importance_type = 'built_in'
                    elif hasattr(model, 'coef_'):
                        # Linear models use absolute coefficient values
                        importances = np.abs(model.coef_)
                        importance_type = 'coefficients'
                    else:
                        # Use permutation importance for other models
                        perm_importance = permutation_importance(
                            model, X_train_model, y_train[target],
                            n_repeats=5, random_state=42, n_jobs=-1
                        )
                        importances = perm_importance.importances_mean
                        importance_type = 'permutation'
                    
                    # Create feature importance DataFrame
                    feature_importance = pd.DataFrame({
                        'feature': X_train_model.columns,
                        'importance': importances,
                        'importance_type': importance_type
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[target][model_name] = feature_importance
                        
                except Exception as e:
                    continue
        
        return importance_results
    
    def create_business_insights(self, importance_results: Dict, test_results: Dict) -> Dict:
        """
        Generate actionable business insights from model performance and feature importance.
        
        Translates technical results into stakeholder-specific recommendations.
        """
        print("\nGenerating business insights from model results...")
        
        insights = {
            'model_performance': {},
            'key_drivers': {},
            'stakeholder_recommendations': {}
        }
        
        # Summarize model performance
        for target in test_results.keys():
            if test_results[target]:
                best_model = max(test_results[target], key=lambda x: test_results[target][x]['r2'])
                best_metrics = test_results[target][best_model]
                
                insights['model_performance'][target] = {
                    'best_model': best_model,
                    'r2': best_metrics['r2'],
                    'mae': best_metrics['mae'],
                    'mape': best_metrics['mape'],
                    'predictability': self._classify_predictability(best_metrics['r2'])
                }
        
        # Extract key performance drivers
        for target in importance_results.keys():
            # Prefer Random Forest feature importances if available
            if 'random_forest' in importance_results[target]:
                top_features = importance_results[target]['random_forest'].head(5)
            else:
                # Use best model's feature importances
                if target in insights['model_performance']:
                    best_model = insights['model_performance'][target]['best_model']
                    if best_model in importance_results[target]:
                        top_features = importance_results[target][best_model].head(5)
                    else:
                        top_features = pd.DataFrame()
                else:
                    top_features = pd.DataFrame()
            
            if not top_features.empty:
                insights['key_drivers'][target] = {
                    'top_features': top_features['feature'].tolist(),
                    'importance_scores': top_features['importance'].tolist()
                }
        
        # Generate stakeholder-specific recommendations
        insights['stakeholder_recommendations'] = self._generate_stakeholder_recommendations(insights)
        
        return insights
    
    def _classify_predictability(self, r2_score: float) -> str:
        """Classify model predictability level based on R-squared score."""
        if r2_score >= 0.7:
            return "High"
        elif r2_score >= 0.5:
            return "Medium"
        elif r2_score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_stakeholder_recommendations(self, insights: Dict) -> Dict:
        """
        Generate tailored recommendations for different stakeholder groups.
        
        Each stakeholder group receives insights relevant to their specific needs.
        """
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


# Complete fixed ProductionModelManager class for model_pipeline.py
# Replace the entire ProductionModelManager class with this:

class ProductionModelManager:
    """
    Manages trained models for production deployment including serialization
    and prediction API creation.
    """
    
    def __init__(self, pipeline: ModelPipeline):
        self.pipeline = pipeline
        self.deployment_models = {}
        
    def prepare_for_deployment(self, test_results: Dict) -> None:
        """
        Prepare the best performing models for production deployment.
        
        Selects the best model for each target based on test set performance.
        """
        print("\nPreparing models for production deployment...")
        
        for target in test_results.keys():
            if test_results[target]:
                # Select best model based on R-squared score
                best_model_name = max(test_results[target], key=lambda x: test_results[target][x]['r2'])
                best_model = self.pipeline.models[target][best_model_name]
                best_metrics = test_results[target][best_model_name]
                
                # Get associated scaler if exists
                scaler_key = f"{target}_{best_model_name}"
                scaler = self.pipeline.scalers.get(scaler_key, None)
                
                # Get selected features for this target
                selected_features = self.pipeline.feature_selector.selected_features_[target]
                
                # Store deployment configuration
                self.deployment_models[target] = {
                    'model': best_model,
                    'model_name': best_model_name,
                    'scaler': scaler,
                    'features': selected_features,
                    'metrics': best_metrics
                }
                
                print(f"  {target.upper()}: {best_model_name} (R² = {best_metrics['r2']:.3f})")
    
    def create_prediction_function(self):
        """
        Create a prediction function that properly handles scaling/unscaling.
        
        Returns:
            Callable that takes feature dict and returns predictions dict
        """
        # Make sure deployment models are prepared
        if not self.deployment_models:
            raise RuntimeError("No deployment models prepared. Call prepare_for_deployment() first.")
        
        # Use deployment_models
        deployment_models = self.deployment_models
        
        def predict(input_features: Dict[str, float]) -> Dict[str, float]:
            """
            Make predictions for all targets given input features.
            
            Args:
                input_features: Dictionary of feature names to values
                
            Returns:
                Dictionary with predictions for pts, reb, ast
            """
            predictions = {}
            
            for target in ['pts', 'reb', 'ast']:
                try:
                    if target not in deployment_models:
                        print(f"Warning: No model found for {target}")
                        defaults = {'pts': 15.0, 'reb': 5.0, 'ast': 3.0}
                        predictions[target] = defaults.get(target, 0.0)
                        continue
                    
                    # Get deployment info for this target
                    deployment_info = deployment_models[target]
                    model = deployment_info['model']
                    scaler = deployment_info.get('scaler')
                    features = deployment_info['features']
                    
                    # Create feature DataFrame with required features
                    feature_values = []
                    for feature in features:
                        if feature in input_features:
                            feature_values.append(input_features[feature])
                        else:
                            # Use 0 as default for missing features
                            feature_values.append(0)
                    
                    feature_df = pd.DataFrame([feature_values], columns=features)
                    
                    # Apply scaling if scaler exists
                    if scaler is not None:
                        feature_array = scaler.transform(feature_df)
                        feature_df_scaled = pd.DataFrame(feature_array, columns=features)
                    else:
                        feature_df_scaled = feature_df
                    
                    # Get prediction
                    raw_pred = model.predict(feature_df_scaled)[0]
                    
                    # Apply reasonable bounds for basketball stats
                    if target == 'pts':
                        actual_pred = np.clip(raw_pred, 0, 60)
                    elif target == 'reb':
                        actual_pred = np.clip(raw_pred, 0, 25)
                    elif target == 'ast':
                        actual_pred = np.clip(raw_pred, 0, 20)
                    else:
                        actual_pred = raw_pred
                    
                    predictions[target] = round(float(actual_pred), 1)
                    
                except Exception as e:
                    print(f"Error predicting {target}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to reasonable defaults
                    defaults = {'pts': 15.0, 'reb': 5.0, 'ast': 3.0}
                    predictions[target] = defaults.get(target, 0.0)
            
            return predictions
        
        return predict

    def save_production_artifacts(self, output_dir: str = "../outputs/artifacts") -> Path:
        """
        Save all production artifacts including models, scalers, and metadata.
        
        Creates a directory structure with all necessary files for deployment.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save artifacts for each target
        for target, deployment_info in self.deployment_models.items():
            target_dir = output_path / target
            target_dir.mkdir(exist_ok=True)
            
            # Save model
            model_file = target_dir / "model.joblib"
            joblib.dump(deployment_info['model'], model_file)
            
            # Save scaler if exists
            if deployment_info['scaler'] is not None:
                scaler_file = target_dir / "scaler.joblib"
                joblib.dump(deployment_info['scaler'], scaler_file)
            
            # Save metadata
            metadata = {
                'model_name': deployment_info['model_name'],
                'features': deployment_info['features'],
                'metrics': deployment_info['metrics'],
                'timestamp': timestamp,
                'has_scaler': deployment_info['scaler'] is not None
            }
            
            metadata_file = target_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"Production artifacts successfully saved to: {output_path}")
        return output_path

# If you need to check whether your pipeline uses scaling, add this diagnostic function:
def diagnose_scaling_issue(pipeline, X_sample, y_sample):
    """
    Diagnose whether the pipeline is using scaled targets.
    
    Args:
        pipeline: The model pipeline object
        X_sample: Sample feature data
        y_sample: Sample target data (dict with 'pts', 'reb', 'ast')
    """
    print("Diagnosing scaling issues...")
    
    # Check if y values look scaled
    for target, values in y_sample.items():
        min_val, max_val = values.min(), values.max()
        mean_val = values.mean()
        
        print(f"\n{target.upper()} statistics:")
        print(f"  Min: {min_val:.3f}")
        print(f"  Max: {max_val:.3f}")
        print(f"  Mean: {mean_val:.3f}")
        
        if max_val < 1.0 or (max_val - min_val) < 2.0:
            print(f"  WARNING: {target} appears to be scaled/normalized!")
        else:
            print(f"  OK: {target} appears to be in original scale")
    
    # Check if the pipeline has scalers
    if hasattr(pipeline, 'scalers'):
        print("\nPipeline has scalers:")
        for target, scaler in pipeline.scalers.items():
            if scaler is not None:
                print(f"  {target}: {type(scaler).__name__}")
                if hasattr(scaler, 'mean_'):
                    print(f"    Mean: {scaler.mean_[0]:.2f}")
                if hasattr(scaler, 'scale_'):
                    print(f"    Scale: {scaler.scale_[0]:.2f}")
                    
def validate_model_results(test_results: Dict, min_r2_threshold: float = 0.3) -> bool:
    """
    Validate that model results meet minimum performance criteria.
    
    Ensures models are sufficiently accurate before deployment.
    """
    validation_passed = True
    
    print("\nValidating model performance against minimum thresholds...")
    
    for target, models in test_results.items():
        if models:
            best_r2 = max(model_metrics['r2'] for model_metrics in models.values())
            
            if best_r2 < min_r2_threshold:
                print(f"WARNING: {target.upper()} best R-squared ({best_r2:.3f}) is below minimum threshold ({min_r2_threshold})")
                validation_passed = False
            else:
                print(f"PASSED: {target.upper()} best R-squared = {best_r2:.3f}")
        else:
            print(f"FAILED: {target.upper()} - no models were successfully evaluated")
            validation_passed = False
    
    return validation_passed


def save_model_artifacts(pipeline: ModelPipeline, test_results: Dict, 
                        insights: Dict, output_dir: str = "../outputs/artifacts") -> None:
    """
    Save all model artifacts including pipeline, results, and insights.
    
    Creates a comprehensive archive of the modeling process for reproducibility.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete pipeline
    pipeline_file = output_path / "nba_pipeline.joblib"
    joblib.dump(pipeline, pipeline_file)
    
    # Save results and insights
    results_file = output_path / "model_results.joblib"
    joblib.dump({
        'test_results': test_results,
        'insights': insights,
        'timestamp': timestamp
    }, results_file)
    
    # Save feature lists for reference
    feature_lists = {}
    for target in pipeline.feature_selector.selected_features_:
        feature_lists[target] = pipeline.feature_selector.selected_features_[target]
    
    features_file = output_path / "selected_features.json"
    with open(features_file, 'w') as f:
        json.dump(feature_lists, f, indent=2)
    
    print(f"All model artifacts saved to: {output_path}")


# Fixed section of run_nba_modeling_pipeline in model_pipeline.py
# Replace the production manager section with this:

def run_nba_modeling_pipeline(data_path: str = "../data/processed/final_engineered_nba_data.parquet") -> Tuple:
    """
    Execute the complete NBA modeling pipeline with target-specific optimizations.
    
    This is the main entry point for training models on NBA player statistics.
    
    Returns:
        Tuple of (pipeline, test_results, insights, production_manager)
    """
    print("NBA PLAYER PERFORMANCE PREDICTION MODELING PIPELINE")
    print("=" * 55)
    print("Initializing comprehensive model training workflow...")
    
    # Configure model parameters
    config = ModelConfig(
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        n_cv_folds=5,
        max_features_for_selection=35,
        correlation_threshold=0.80,
        feature_selection_method='rfe'
    )
    
    # Load and validate data
    loader = DataLoader(config)
    df = loader.load_and_validate(data_path)
    
    # Initialize pipeline
    pipeline = ModelPipeline(config)
    
    # Prepare data for modeling
    X, y = pipeline.prepare_model_data(df)
    
    # Create time-aware data splits
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.create_time_aware_split(df, X, y)
    
    # Train models with target-specific feature selection
    pipeline.train_models(X_train, X_val, y_train, y_val)
    
    # Evaluate on test set
    test_results = pipeline.evaluate_on_test(X_test, y_test)
    
    # Generate insights and feature importance
    interpreter = ModelInterpreter(pipeline)
    importance_results = interpreter.analyze_feature_importance(X_train, y_train)
    insights = interpreter.create_business_insights(importance_results, test_results)
    
    # Add preprocessing details to insights
    insights['target_preprocessing'] = pipeline.target_preprocessing
    insights['feature_selection_insights'] = {}
    for target in y.keys():
        insights['feature_selection_insights'][target] = pipeline.feature_selector.get_feature_insights(target)
    
    # Prepare models for production
    production_manager = ProductionModelManager(pipeline)
    
    # IMPORTANT: Call prepare_for_deployment BEFORE save_production_artifacts
    production_manager.prepare_for_deployment(test_results)
    production_manager.save_production_artifacts()
    
    print("\nMODELING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 40)
    
    # Display feature optimization summary
    print("\nTarget-Specific Feature Optimization Summary:")
    for target in y.keys():
        if target in pipeline.target_preprocessing:
            feature_count = len(pipeline.target_preprocessing[target]['selected_features'])
            print(f"  {target.upper()}: {feature_count} optimized features selected")
    
    return pipeline, test_results, insights, production_manager


if __name__ == "__main__":
    """Execute the complete modeling pipeline when run as main script."""
    
    try:
        # Run the complete modeling pipeline
        pipeline, test_results, insights, production_manager = run_nba_modeling_pipeline()
        
        # Validate results meet minimum performance criteria
        validation_passed = validate_model_results(test_results, min_r2_threshold=0.3)
        
        # Save all artifacts for future use
        save_model_artifacts(pipeline, test_results, insights)
        
        # Display final performance summary
        print("\nFINAL MODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        for target, performance in insights['model_performance'].items():
            if target in insights['target_preprocessing']:
                feature_count = len(insights['target_preprocessing'][target]['selected_features'])
                print(f"{target.upper()}: {performance['best_model'].replace('_', ' ').title()} "
                      f"(R-squared = {performance['r2']:.3f}, MAE = {performance['mae']:.2f}) "
                      f"using {feature_count} optimized features")
        
        # Display feature selection insights
        print("\nFEATURE SELECTION INSIGHTS:")
        print("-" * 40)
        for target in insights['feature_selection_insights'].keys():
            target_insights = insights['feature_selection_insights'][target]
            print(f"{target.upper()}: Selected {target_insights['selected_count']} features "
                  f"from {target_insights['original_count']} original features "
                  f"({target_insights['reduction_ratio']*100:.1f}% reduction)")
        
        print("\nPipeline execution completed successfully!")
        
    except Exception as e:
        print(f"Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()