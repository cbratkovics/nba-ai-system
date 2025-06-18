"""
NBA Player Performance Feature Engineering Module

A comprehensive feature engineering module for NBA player performance prediction
following 2025 data science best practices.

Author: Christopher Bratkovics
Created: 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress common warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class NBAFeatureConfig:
    """Configuration class for NBA feature engineering parameters."""
    
    # Core statistics for per-minute calculations
    RATE_STATS = ['pts', 'reb', 'ast', 'stl', 'blk', 'turnover', 'fga', 'fg3a', 'fta']
    
    # Statistics that can contribute to double-doubles/triple-doubles
    MILESTONE_STATS = ['pts', 'reb', 'ast', 'stl', 'blk']
    
    # Required columns for different feature groups
    REQUIRED_COLUMNS = {
        'rest_days': ['player_id', 'game_date'],
        'shooting_efficiency': ['pts', 'fga', 'fta', 'fgm', 'fg3m'],
        'game_context': ['team_id', 'game_home_team_id'],
        'performance_milestones': ['pts', 'reb', 'ast'],
        'usage': ['fga', 'fta', 'turnover']
    }
    
    # Thresholds for performance categories
    THRESHOLDS = {
        'sufficient_rest_days': 2,
        'high_scoring_game': 30,
        'efficient_game_ts': 0.65,
        'efficient_game_min_pts': 15,
        'high_usage_possessions': 20,
        'meaningful_minutes': 10
    }


class MinutesConverter:
    """Utility class for converting NBA minutes data."""
    
    @staticmethod
    def convert_to_decimal(minutes_value: Union[str, int, float]) -> float:
        """
        Convert minutes from various formats to decimal format.
        
        Args:
            minutes_value: Minutes in string format (e.g., "30:45") or numeric
            
        Returns:
            Minutes as decimal float
            
        Examples:
            >>> MinutesConverter.convert_to_decimal("30:45")
            30.75
            >>> MinutesConverter.convert_to_decimal("30")
            30.0
        """
        if pd.isna(minutes_value) or minutes_value == '':
            return 0.0
        
        minutes_str = str(minutes_value).strip()
        
        try:
            if ':' in minutes_str:
                parts = minutes_str.split(':')
                minutes = float(parts[0])
                seconds = float(parts[1]) if len(parts) > 1 else 0
                return minutes + (seconds / 60)
            else:
                return float(minutes_str)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not convert minutes value '{minutes_value}': {e}")
            return 0.0


class BaseNBATransformer(BaseEstimator, TransformerMixin):
    """Base class for NBA feature transformers following sklearn patterns."""
    
    def __init__(self, validate_input: bool = True):
        self.validate_input = validate_input
        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
    
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if self.validate_input and hasattr(self, 'required_columns_'):
            missing_cols = set(self.required_columns_) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return X.copy()
    
    def _store_input_info(self, X: pd.DataFrame) -> None:
        """Store information about input features."""
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(X.columns)


class RestDaysTransformer(BaseNBATransformer):
    """Transform rest days between games for each player."""
    
    def __init__(self, 
                 sufficient_rest_threshold: int = NBAFeatureConfig.THRESHOLDS['sufficient_rest_days'],
                 default_first_game_rest: int = 7,
                 **kwargs):
        super().__init__(**kwargs)
        self.sufficient_rest_threshold = sufficient_rest_threshold
        self.default_first_game_rest = default_first_game_rest
        self.required_columns_ = NBAFeatureConfig.REQUIRED_COLUMNS['rest_days']
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for rest days)."""
        X = self._validate_input(X)
        self._store_input_info(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate rest days and related features."""
        X = self._validate_input(X)
        
        logger.info("Calculating rest days between games...")
        
        # Ensure game_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(X['game_date']):
            X['game_date'] = pd.to_datetime(X['game_date'])
        
        # Sort by player and date
        X = X.sort_values(['player_id', 'game_date']).reset_index(drop=True)
        
        # Calculate rest days
        X['previous_game_date'] = X.groupby('player_id')['game_date'].shift(1)
        X['rest_days'] = (X['game_date'] - X['previous_game_date']).dt.days
        
        # Handle first games
        first_game_mask = X['previous_game_date'].isna()
        X.loc[first_game_mask, 'rest_days'] = self.default_first_game_rest
        
        # Create categorical features
        X['rest_days_category'] = pd.cut(
            X['rest_days'],
            bins=[-np.inf, 1, 2, 4, np.inf],
            labels=['back_to_back', '1_day', '2_3_days', '4_plus_days'],
            include_lowest=True
        )
        
        # Binary indicators
        X['sufficient_rest'] = X['rest_days'] >= self.sufficient_rest_threshold
        X['is_back_to_back'] = X['rest_days'] == 1
        X['is_first_game'] = first_game_mask
        
        # Clean up
        X = X.drop('previous_game_date', axis=1)
        
        logger.info(f"Rest days distribution:\n{X['rest_days'].value_counts().sort_index().head(10)}")
        
        return X


class ShootingEfficiencyTransformer(BaseNBATransformer):
    """Calculate advanced shooting efficiency metrics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.required_columns_ = NBAFeatureConfig.REQUIRED_COLUMNS['shooting_efficiency']
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        X = self._validate_input(X)
        self._store_input_info(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate shooting efficiency metrics."""
        X = self._validate_input(X)
        
        logger.info("Calculating shooting efficiency metrics...")
        
        # True Shooting Percentage
        if all(col in X.columns for col in ['pts', 'fga', 'fta']):
            denominator = 2 * (X['fga'] + 0.44 * X['fta'])
            X['true_shooting_pct'] = np.where(denominator > 0, X['pts'] / denominator, 0)
        
        # Effective Field Goal Percentage
        if all(col in X.columns for col in ['fgm', 'fg3m', 'fga']):
            X['effective_fg_pct'] = np.where(
                X['fga'] > 0, 
                (X['fgm'] + 0.5 * X['fg3m']) / X['fga'], 
                0
            )
        
        # Shooting accuracy indicators
        X['perfect_ft_game'] = (X['fta'] > 0) & (X['ft_pct'] == 1.0)
        X['good_shooting_game'] = (X['fga'] >= 5) & (X['fg_pct'] >= 0.5)
        
        return X


class PerMinuteRatesTransformer(BaseNBATransformer):
    """Calculate per-minute production rates."""
    
    def __init__(self, 
                 stats_to_transform: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.stats_to_transform = stats_to_transform or NBAFeatureConfig.RATE_STATS
        self.required_columns_ = ['minutes_played'] + self.stats_to_transform
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        X = self._validate_input(X)
        self._store_input_info(X)
        
        # Filter stats to only those present in data
        self.available_stats_ = [stat for stat in self.stats_to_transform if stat in X.columns]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-minute rates."""
        X = self._validate_input(X)
        
        if 'minutes_played' not in X.columns:
            logger.warning("'minutes_played' column not found - skipping per-minute calculations")
            return X
        
        logger.info("Calculating per-minute production rates...")
        
        for stat in self.available_stats_:
            # Per-minute rate
            X[f'{stat}_per_min'] = np.where(
                X['minutes_played'] > 0, 
                X[stat] / X['minutes_played'], 
                0
            )
            
            # Per-36-minute rate (NBA standard)
            X[f'{stat}_per_36min'] = np.where(
                X['minutes_played'] > 0, 
                (X[stat] / X['minutes_played']) * 36, 
                0
            )
        
        return X


class GameContextTransformer(BaseNBATransformer):
    """Create game context indicators."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.required_columns_ = []  # Will be set based on available columns
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        X = self._validate_input(X)
        self._store_input_info(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create game context indicators."""
        X = self._validate_input(X)
        
        logger.info("Creating game context indicators...")
        
        # Home vs Away indicator
        if all(col in X.columns for col in ['team_id', 'game_home_team_id']):
            X['is_home_game'] = (X['team_id'] == X['game_home_team_id'])
            X['is_away_game'] = ~X['is_home_game']
        
        # Playoff indicator
        if 'game_postseason' in X.columns:
            X['is_playoff_game'] = X['game_postseason'].fillna(False)
        
        # Season timing features
        if 'game_date' in X.columns:
            if not pd.api.types.is_datetime64_any_dtype(X['game_date']):
                X['game_date'] = pd.to_datetime(X['game_date'])
            
            X['month'] = X['game_date'].dt.month
            X['day_of_week'] = X['game_date'].dt.dayofweek
            X['is_weekend'] = X['day_of_week'].isin([5, 6])  # Saturday, Sunday
        
        return X


class PerformanceMilestonesTransformer(BaseNBATransformer):
    """Create performance milestone indicators."""
    
    def __init__(self, 
                 milestone_stats: Optional[List[str]] = None,
                 high_scoring_threshold: int = NBAFeatureConfig.THRESHOLDS['high_scoring_game'],
                 **kwargs):
        super().__init__(**kwargs)
        self.milestone_stats = milestone_stats or NBAFeatureConfig.MILESTONE_STATS
        self.high_scoring_threshold = high_scoring_threshold
        self.required_columns_ = NBAFeatureConfig.REQUIRED_COLUMNS['performance_milestones']
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        X = self._validate_input(X)
        self._store_input_info(X)
        
        # Filter stats to only those present in data
        self.available_milestone_stats_ = [
            stat for stat in self.milestone_stats if stat in X.columns
        ]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create performance milestone indicators."""
        X = self._validate_input(X)
        
        logger.info("Creating performance milestone indicators...")
        
        # Count double-digit stats
        double_digit_matrix = pd.DataFrame()
        for stat in self.available_milestone_stats_:
            double_digit_matrix[stat] = X[stat] >= 10
        
        double_digit_count = double_digit_matrix.sum(axis=1)
        
        # Milestone indicators
        X['double_double'] = double_digit_count >= 2
        X['triple_double'] = double_digit_count >= 3
        X['high_scoring_game'] = X['pts'] >= self.high_scoring_threshold
        
        # Efficient game indicator
        if 'true_shooting_pct' in X.columns:
            X['efficient_game'] = (
                (X['true_shooting_pct'] >= NBAFeatureConfig.THRESHOLDS['efficient_game_ts']) & 
                (X['pts'] >= NBAFeatureConfig.THRESHOLDS['efficient_game_min_pts'])
            )
        
        # Perfect free throw game
        if all(col in X.columns for col in ['fta', 'ftm']):
            X['perfect_ft_game'] = (X['fta'] > 0) & (X['fta'] == X['ftm'])
        
        return X


class NBAFeatureEngineer:
    """
    Main feature engineering pipeline for NBA player performance data.
    
    This class orchestrates multiple transformers to create a comprehensive
    set of features for NBA player performance prediction.
    """
    
    def __init__(self, 
                 include_rest_days: bool = True,
                 include_shooting_efficiency: bool = True,
                 include_per_minute_rates: bool = True,
                 include_game_context: bool = True,
                 include_performance_milestones: bool = True,
                 config: Optional[NBAFeatureConfig] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            include_rest_days: Whether to include rest days features
            include_shooting_efficiency: Whether to include shooting efficiency metrics
            include_per_minute_rates: Whether to include per-minute production rates
            include_game_context: Whether to include game context indicators
            include_performance_milestones: Whether to include performance milestones
            config: Configuration object (uses default if None)
        """
        self.include_rest_days = include_rest_days
        self.include_shooting_efficiency = include_shooting_efficiency
        self.include_per_minute_rates = include_per_minute_rates
        self.include_game_context = include_game_context
        self.include_performance_milestones = include_performance_milestones
        self.config = config or NBAFeatureConfig()
        
        self.transformers_ = {}
        self.is_fitted_ = False
    
    def _convert_minutes_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert minutes column to decimal format."""
        if 'minutes_played' not in df.columns and 'min' in df.columns:
            logger.info("Converting 'min' to 'minutes_played'...")
            df['minutes_played'] = df['min'].apply(MinutesConverter.convert_to_decimal)
        elif 'minutes_played' in df.columns and df['minutes_played'].dtype == 'object':
            logger.info("Converting 'minutes_played' to decimal format...")
            df['minutes_played'] = df['minutes_played'].apply(MinutesConverter.convert_to_decimal)
        
        return df
    
    def fit(self, X: pd.DataFrame, y=None) -> 'NBAFeatureEngineer':
        """
        Fit the feature engineering pipeline.
        
        Args:
            X: Input DataFrame with NBA player game stats
            y: Target variable (ignored)
            
        Returns:
            self
        """
        logger.info("Fitting NBA Feature Engineering Pipeline...")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        X_work = X.copy()
        
        # Convert minutes column
        X_work = self._convert_minutes_column(X_work)
        
        # Initialize and fit transformers
        if self.include_rest_days:
            self.transformers_['rest_days'] = RestDaysTransformer()
            self.transformers_['rest_days'].fit(X_work)
        
        if self.include_shooting_efficiency:
            self.transformers_['shooting_efficiency'] = ShootingEfficiencyTransformer()
            self.transformers_['shooting_efficiency'].fit(X_work)
        
        if self.include_per_minute_rates:
            self.transformers_['per_minute_rates'] = PerMinuteRatesTransformer()
            self.transformers_['per_minute_rates'].fit(X_work)
        
        if self.include_game_context:
            self.transformers_['game_context'] = GameContextTransformer()
            self.transformers_['game_context'].fit(X_work)
        
        if self.include_performance_milestones:
            self.transformers_['performance_milestones'] = PerformanceMilestonesTransformer()
            self.transformers_['performance_milestones'].fit(X_work)
        
        self.is_fitted_ = True
        logger.info("Feature engineering pipeline fitted successfully!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using fitted transformers.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before transform")
        
        logger.info("Transforming data with NBA Feature Engineering Pipeline...")
        
        X_transformed = X.copy()
        
        # Convert minutes column
        X_transformed = self._convert_minutes_column(X_transformed)
        
        # Apply transformers in order
        for transformer_name, transformer in self.transformers_.items():
            try:
                X_transformed = transformer.transform(X_transformed)
                logger.debug(f"Applied {transformer_name} transformer")
            except Exception as e:
                logger.warning(f"Error applying {transformer_name} transformer: {e}")
                continue
        
        logger.info(f"Feature engineering complete! "
                   f"Features: {len(X.columns)} → {len(X_transformed.columns)}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the pipeline and transform the data in one step.
        
        Args:
            X: Input DataFrame
            y: Target variable (ignored)
            
        Returns:
            DataFrame with engineered features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features that would be created."""
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted to get feature names")
        
        # This would need to be implemented based on actual transformation results
        # For now, return a placeholder
        return ["feature_names_available_after_transform"]
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the feature engineering process."""
        return {
            'transformers': list(self.transformers_.keys()),
            'is_fitted': self.is_fitted_,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }


# Utility functions for feature analysis and validation
def validate_features_for_hypothesis_testing(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that required features exist for hypothesis testing.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary indicating which hypothesis tests can be performed
    """
    validation_results = {
        'hypothesis_1_rest_days': all(col in df.columns for col in ['rest_days', 'sufficient_rest', 'fg_pct', 'fga']),
        'hypothesis_2_home_away': all(col in df.columns for col in ['is_home_game', 'pts']),
        'hypothesis_3_three_point': all(col in df.columns for col in ['fg3a_per_36min', 'game_season', 'minutes_played'])
    }
    
    logger.info("Hypothesis testing feature validation:")
    for test, valid in validation_results.items():
        status = "✓" if valid else "✗"
        logger.info(f"  {status} {test}: {'Ready' if valid else 'Missing features'}")
    
    return validation_results


def create_feature_engineering_pipeline(
    custom_config: Optional[Dict[str, Any]] = None
) -> NBAFeatureEngineer:
    """
    Factory function to create a configured feature engineering pipeline.
    
    Args:
        custom_config: Custom configuration parameters
        
    Returns:
        Configured NBAFeatureEngineer instance
    """
    config = custom_config or {}
    
    return NBAFeatureEngineer(
        include_rest_days=config.get('include_rest_days', True),
        include_shooting_efficiency=config.get('include_shooting_efficiency', True),
        include_per_minute_rates=config.get('include_per_minute_rates', True),
        include_game_context=config.get('include_game_context', True),
        include_performance_milestones=config.get('include_performance_milestones', True)
    )


def analyze_feature_importance_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze feature engineering insights for model preparation.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Dictionary containing analysis insights
    """
    insights = {}
    
    # Minutes played analysis
    if 'minutes_played' in df.columns:
        min_stats = df['minutes_played'].describe()
        insights['minutes_analysis'] = {
            'mean_minutes': min_stats['mean'],
            'zero_minutes_pct': (df['minutes_played'] == 0).mean() * 100,
            'low_minutes_pct': (df['minutes_played'] < 10).mean() * 100,
            'recommendation': f"Consider filtering for players with >=10 minutes (retains {(df['minutes_played'] >= 10).mean()*100:.1f}% of data)"
        }
    
    # Rest days distribution
    if 'rest_days' in df.columns:
        rest_dist = df['rest_days'].value_counts().sort_index()
        insights['rest_days_analysis'] = {
            'distribution': rest_dist.head(10).to_dict(),
            'sufficient_rest_pct': df['sufficient_rest'].mean() * 100 if 'sufficient_rest' in df.columns else None
        }
    
    # Performance milestones
    milestone_cols = ['double_double', 'triple_double', 'high_scoring_game']
    if any(col in df.columns for col in milestone_cols):
        milestone_stats = {}
        for col in milestone_cols:
            if col in df.columns:
                milestone_stats[col] = df[col].mean() * 100
        insights['performance_milestones'] = milestone_stats
    
    return insights


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_feature_engineering()