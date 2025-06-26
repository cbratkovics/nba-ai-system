"""
NBA Player Performance Feature Engineering Module

A comprehensive feature engineering module for NBA player performance prediction
following 2025 data science best practices.

Author: Christopher Bratkovics
Created: 2025
Enhanced: Added opponent metrics, elite player classification, and interaction features
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
        'usage': ['fga', 'fta', 'turnover'],
        'opponent_metrics': ['team_id', 'game_home_team_id', 'game_visitor_team_id']
    }
    
    # Thresholds for performance categories
    THRESHOLDS = {
        'sufficient_rest_days': 2,
        'high_scoring_game': 30,
        'efficient_game_ts': 0.65,
        'efficient_game_min_pts': 15,
        'high_usage_possessions': 20,
        'meaningful_minutes': 10,
        'elite_player_ppg': 20,
        'elite_player_mpg': 30,
        'elite_player_usage_fga': 12,  # FGA per game
        'elite_player_usage_combined': 15  # Combined reb + ast per game
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


class OpponentMetricsTransformer(BaseNBATransformer):
    """Calculate opponent defensive metrics and pace of play."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.required_columns_ = NBAFeatureConfig.REQUIRED_COLUMNS['opponent_metrics']
        self.opponent_stats_cache_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer by calculating opponent defensive metrics."""
        X = self._validate_input(X)
        self._store_input_info(X)
        
        logger.info("Calculating opponent defensive metrics...")
        
        # Calculate opponent defensive statistics
        self.opponent_stats_cache_ = self._calculate_opponent_defensive_stats(X)
        
        return self
    
    def _calculate_opponent_defensive_stats(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate defensive statistics for each team."""
        opponent_stats = {}
        
        # Calculate overall defensive stats per team
        team_defense = df.groupby('team_id').agg({
            'pts': 'mean',  # Points allowed per game (from opponent perspective)
            'fga': 'mean',  # Field goal attempts allowed
            'minutes_played': 'sum'  # For pace calculation
        }).reset_index()
        
        team_defense.columns = ['team_id', 'pts_allowed_avg', 'fga_allowed_avg', 'total_minutes']
        
        # Calculate pace (possessions per game estimate)
        # Simple pace estimation: (FGA + 0.4 * FTA + TO) per game
        if all(col in df.columns for col in ['fga', 'fta', 'turnover']):
            pace_stats = df.groupby('team_id').agg({
                'fga': 'mean',
                'fta': 'mean', 
                'turnover': 'mean'
            }).reset_index()
            
            pace_stats['pace_estimate'] = (
                pace_stats['fga'] + 0.4 * pace_stats['fta'] + pace_stats['turnover']
            )
            
            team_defense = team_defense.merge(
                pace_stats[['team_id', 'pace_estimate']], 
                on='team_id', 
                how='left'
            )
        
        opponent_stats['team_defense'] = team_defense
        
        # Calculate position-specific defensive ratings
        if 'player_position' in df.columns:
            position_defense = df.groupby(['team_id', 'player_position']).agg({
                'pts': 'mean',
                'reb': 'mean',
                'ast': 'mean'
            }).reset_index()
            
            position_defense.columns = [
                'team_id', 'player_position', 
                'pts_allowed_vs_position', 'reb_allowed_vs_position', 'ast_allowed_vs_position'
            ]
            
            opponent_stats['position_defense'] = position_defense
        
        return opponent_stats
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add opponent metrics to the dataset."""
        X = self._validate_input(X)
        
        logger.info("Adding opponent metrics...")
        
        # Identify opponent team for each game
        X['opponent_team_id'] = X.apply(self._get_opponent_team_id, axis=1)
        
        # Merge opponent defensive stats
        if 'team_defense' in self.opponent_stats_cache_:
            team_defense = self.opponent_stats_cache_['team_defense']
            
            # Rename columns for merging as opponent stats
            opponent_defense = team_defense.copy()
            opponent_defense.columns = [
                'opponent_team_id' if col == 'team_id' else f'opponent_{col}'
                for col in opponent_defense.columns
            ]
            
            X = X.merge(opponent_defense, on='opponent_team_id', how='left')
        
        # Merge position-specific opponent stats
        if 'position_defense' in self.opponent_stats_cache_ and 'player_position' in X.columns:
            position_defense = self.opponent_stats_cache_['position_defense']
            
            # Rename for opponent perspective
            opponent_pos_defense = position_defense.copy()
            opponent_pos_defense.columns = [
                'opponent_team_id' if col == 'team_id' else 
                'player_position' if col == 'player_position' else 
                f'opponent_{col}'
                for col in opponent_pos_defense.columns
            ]
            
            X = X.merge(
                opponent_pos_defense, 
                on=['opponent_team_id', 'player_position'], 
                how='left'
            )
        
        # Fill missing opponent stats with league averages
        opponent_cols = [col for col in X.columns if col.startswith('opponent_')]
        for col in opponent_cols:
            if X[col].isna().any():
                league_avg = X[col].mean()
                X[col] = X[col].fillna(league_avg)
        
        return X
    
    def _get_opponent_team_id(self, row) -> int:
        """Determine the opponent team ID for a given game."""
        player_team = row['team_id']
        home_team = row['game_home_team_id']
        visitor_team = row['game_visitor_team_id']
        
        # Opponent is the other team in the game
        if player_team == home_team:
            return visitor_team
        else:
            return home_team


class ElitePlayerTransformer(BaseNBATransformer):
    """Create elite player classification features."""
    
    def __init__(self, 
                 ppg_threshold: float = NBAFeatureConfig.THRESHOLDS['elite_player_ppg'],
                 mpg_threshold: float = NBAFeatureConfig.THRESHOLDS['elite_player_mpg'],
                 usage_fga_threshold: float = NBAFeatureConfig.THRESHOLDS['elite_player_usage_fga'],
                 usage_combined_threshold: float = NBAFeatureConfig.THRESHOLDS['elite_player_usage_combined'],
                 **kwargs):
        super().__init__(**kwargs)
        self.ppg_threshold = ppg_threshold
        self.mpg_threshold = mpg_threshold
        self.usage_fga_threshold = usage_fga_threshold
        self.usage_combined_threshold = usage_combined_threshold
        self.player_classifications_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit by calculating player season averages for classification."""
        X = self._validate_input(X)
        self._store_input_info(X)
        
        logger.info("Calculating elite player classifications...")
        
        # Calculate player season averages
        player_stats = X.groupby(['player_id', 'game_season']).agg({
            'pts': 'mean',
            'minutes_played': 'mean',
            'fga': 'mean',
            'reb': 'mean',
            'ast': 'mean'
        }).reset_index()
        
        # Apply elite player criteria
        player_stats['elite_scorer'] = player_stats['pts'] >= self.ppg_threshold
        player_stats['elite_minutes'] = player_stats['minutes_played'] >= self.mpg_threshold
        player_stats['elite_usage'] = (
            (player_stats['fga'] >= self.usage_fga_threshold) |
            ((player_stats['reb'] + player_stats['ast']) >= self.usage_combined_threshold)
        )
        
        # Overall elite classification (meet at least 2 of 3 criteria)
        elite_criteria_count = (
            player_stats['elite_scorer'].astype(int) + 
            player_stats['elite_minutes'].astype(int) + 
            player_stats['elite_usage'].astype(int)
        )
        
        player_stats['is_elite_player'] = elite_criteria_count >= 2
        
        # Store classifications
        self.player_classifications_ = player_stats[
            ['player_id', 'game_season', 'is_elite_player', 'elite_scorer', 
             'elite_minutes', 'elite_usage']
        ].copy()
        
        elite_count = player_stats['is_elite_player'].sum()
        total_players = len(player_stats)
        logger.info(f"Classified {elite_count} elite players out of {total_players} player-seasons ({elite_count/total_players*100:.1f}%)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add elite player classification features."""
        X = self._validate_input(X)
        
        logger.info("Adding elite player classifications...")
        
        # Merge player classifications
        X = X.merge(
            self.player_classifications_,
            on=['player_id', 'game_season'],
            how='left'
        )
        
        # Fill missing classifications as non-elite
        elite_cols = ['is_elite_player', 'elite_scorer', 'elite_minutes', 'elite_usage']
        for col in elite_cols:
            if col in X.columns:
                X[col] = X[col].fillna(False)
        
        return X


class InteractionFeaturesTransformer(BaseNBATransformer):
    """Create interaction features between important variables."""
    
    def __init__(self, max_interactions: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_interactions = max_interactions
        self.created_features_ = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for interactions)."""
        X = self._validate_input(X)
        self._store_input_info(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        X = self._validate_input(X)
        
        logger.info("Creating interaction features...")
        
        interactions = [
            ('minutes_played', 'rest_days'), 
            ('minutes_played', 'is_home_game'),
            ('rest_days', 'is_home_game'), 
            ('sufficient_rest', 'minutes_played'),
            ('is_home_game', 'is_weekend'), 
            ('rest_days', 'month'),
            ('minutes_played', 'day_of_week'), 
            ('is_weekend', 'rest_days'),
            ('is_elite_player', 'minutes_played'),
            ('is_elite_player', 'opponent_pts_allowed_avg')
        ]
        
        created_features = []
        interactions_created = 0
        
        for feat1, feat2 in interactions:
            if interactions_created >= self.max_interactions:
                break
                
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                try:
                    X[interaction_name] = X[feat1] * X[feat2]
                    created_features.append(interaction_name)
                    interactions_created += 1
                except Exception as e:
                    logger.warning(f"Could not create interaction {interaction_name}: {e}")
                    continue
        
        self.created_features_ = created_features
        logger.info(f"Created {len(created_features)} interaction features")
        
        return X


class PositionSpecificTransformer(BaseNBATransformer):
    """Create position-specific features."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position_stats_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit by calculating position averages."""
        X = self._validate_input(X)
        self._store_input_info(X)
        
        if 'player_position' in X.columns:
            logger.info("Calculating position-specific statistics...")
            
            # Calculate position averages for key stats
            stats_cols = ['minutes_played', 'pts', 'reb', 'ast', 'fga']
            available_stats = [col for col in stats_cols if col in X.columns]
            
            if available_stats:
                self.position_stats_ = X.groupby('player_position')[available_stats].mean()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features."""
        X = self._validate_input(X)
        
        if 'player_position' in X.columns and len(self.position_stats_) > 0:
            logger.info("Creating position-specific features...")
            
            # Add position vs average comparisons
            for stat in self.position_stats_.columns:
                if stat in X.columns:
                    # Map position averages
                    position_avg_col = f'{stat}_position_avg'
                    X[position_avg_col] = X['player_position'].map(
                        self.position_stats_[stat]
                    )
                    
                    # Calculate ratio vs position average
                    ratio_col = f'{stat}_vs_position_avg'
                    X[ratio_col] = X[stat] / (X[position_avg_col] + 1e-6)
        
        return X


class TemporalFeaturesTransformer(BaseNBATransformer):
    """Enhanced temporal features."""
    
    def __init__(self, date_col: str = 'game_date', **kwargs):
        super().__init__(**kwargs)
        self.date_col = date_col
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        X = self._validate_input(X)
        self._store_input_info(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced temporal features."""
        X = self._validate_input(X)
        
        if self.date_col in X.columns:
            logger.info("Creating enhanced temporal features...")
            
            if not pd.api.types.is_datetime64_any_dtype(X[self.date_col]):
                X[self.date_col] = pd.to_datetime(X[self.date_col])
            
            # Basic temporal features
            X['day_of_week'] = X[self.date_col].dt.dayofweek
            X['month'] = X[self.date_col].dt.month
            X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
            
            # Season progression features
            if 'game_season' in X.columns:
                X['days_since_season_start'] = (
                    X[self.date_col] - 
                    X.groupby('game_season')[self.date_col].transform('min')
                ).dt.days
                
                # Season phase indicators
                X['early_season'] = X['days_since_season_start'] <= 30
                X['mid_season'] = (
                    (X['days_since_season_start'] > 30) & 
                    (X['days_since_season_start'] <= 120)
                )
                X['late_season'] = X['days_since_season_start'] > 120
        
        return X


# Enhanced main classes
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
                 include_opponent_metrics: bool = True,
                 include_elite_player_classification: bool = True,
                 include_interaction_features: bool = True,
                 include_position_specific: bool = True,
                 include_temporal_features: bool = True,
                 config: Optional[NBAFeatureConfig] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            include_rest_days: Whether to include rest days features
            include_shooting_efficiency: Whether to include shooting efficiency metrics
            include_per_minute_rates: Whether to include per-minute production rates
            include_game_context: Whether to include game context indicators
            include_performance_milestones: Whether to include performance milestones
            include_opponent_metrics: Whether to include opponent defensive metrics
            include_elite_player_classification: Whether to include elite player features
            include_interaction_features: Whether to include interaction features
            include_position_specific: Whether to include position-specific features
            include_temporal_features: Whether to include enhanced temporal features
            config: Configuration object (uses default if None)
        """
        self.include_rest_days = include_rest_days
        self.include_shooting_efficiency = include_shooting_efficiency
        self.include_per_minute_rates = include_per_minute_rates
        self.include_game_context = include_game_context
        self.include_performance_milestones = include_performance_milestones
        self.include_opponent_metrics = include_opponent_metrics
        self.include_elite_player_classification = include_elite_player_classification
        self.include_interaction_features = include_interaction_features
        self.include_position_specific = include_position_specific
        self.include_temporal_features = include_temporal_features
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
        logger.info("Fitting Enhanced NBA Feature Engineering Pipeline...")
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        X_work = X.copy()
        
        # Convert minutes column
        X_work = self._convert_minutes_column(X_work)
        
        # Initialize and fit transformers in order
        if self.include_rest_days:
            self.transformers_['rest_days'] = RestDaysTransformer()
            self.transformers_['rest_days'].fit(X_work)
            X_work = self.transformers_['rest_days'].transform(X_work)
        
        if self.include_shooting_efficiency:
            self.transformers_['shooting_efficiency'] = ShootingEfficiencyTransformer()
            self.transformers_['shooting_efficiency'].fit(X_work)
            X_work = self.transformers_['shooting_efficiency'].transform(X_work)
        
        if self.include_per_minute_rates:
            self.transformers_['per_minute_rates'] = PerMinuteRatesTransformer()
            self.transformers_['per_minute_rates'].fit(X_work)
            X_work = self.transformers_['per_minute_rates'].transform(X_work)
        
        if self.include_game_context:
            self.transformers_['game_context'] = GameContextTransformer()
            self.transformers_['game_context'].fit(X_work)
            X_work = self.transformers_['game_context'].transform(X_work)
        
        if self.include_temporal_features:
            self.transformers_['temporal_features'] = TemporalFeaturesTransformer()
            self.transformers_['temporal_features'].fit(X_work)
            X_work = self.transformers_['temporal_features'].transform(X_work)
        
        if self.include_opponent_metrics:
            self.transformers_['opponent_metrics'] = OpponentMetricsTransformer()
            self.transformers_['opponent_metrics'].fit(X_work)
            X_work = self.transformers_['opponent_metrics'].transform(X_work)
        
        if self.include_elite_player_classification:
            self.transformers_['elite_player'] = ElitePlayerTransformer()
            self.transformers_['elite_player'].fit(X_work)
            X_work = self.transformers_['elite_player'].transform(X_work)
        
        if self.include_position_specific:
            self.transformers_['position_specific'] = PositionSpecificTransformer()
            self.transformers_['position_specific'].fit(X_work)
            X_work = self.transformers_['position_specific'].transform(X_work)
        
        if self.include_performance_milestones:
            self.transformers_['performance_milestones'] = PerformanceMilestonesTransformer()
            self.transformers_['performance_milestones'].fit(X_work)
            X_work = self.transformers_['performance_milestones'].transform(X_work)
        
        if self.include_interaction_features:
            self.transformers_['interaction_features'] = InteractionFeaturesTransformer()
            self.transformers_['interaction_features'].fit(X_work)
        
        self.is_fitted_ = True
        logger.info("Enhanced feature engineering pipeline fitted successfully!")
        
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
        
        logger.info("Transforming data with Enhanced NBA Feature Engineering Pipeline...")
        
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
        
        logger.info(f"Enhanced feature engineering complete! "
                   f"Features: {len(X.columns)} -> {len(X_transformed.columns)}")
        
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
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the feature engineering process."""
        feature_counts = {}
        for name, transformer in self.transformers_.items():
            if hasattr(transformer, 'created_features_'):
                feature_counts[name] = len(transformer.created_features_)
            else:
                feature_counts[name] = "Unknown"
        
        return {
            'transformers': list(self.transformers_.keys()),
            'is_fitted': self.is_fitted_,
            'feature_counts_by_transformer': feature_counts,
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
        status = "Ready" if valid else "Missing features"
        logger.info(f"  {test}: {status}")
    
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
        include_performance_milestones=config.get('include_performance_milestones', True),
        include_opponent_metrics=config.get('include_opponent_metrics', True),
        include_elite_player_classification=config.get('include_elite_player_classification', True),
        include_interaction_features=config.get('include_interaction_features', True),
        include_position_specific=config.get('include_position_specific', True),
        include_temporal_features=config.get('include_temporal_features', True)
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
    
    # Elite player distribution
    if 'is_elite_player' in df.columns:
        insights['elite_player_analysis'] = {
            'elite_player_pct': df['is_elite_player'].mean() * 100,
            'elite_scorer_pct': df['elite_scorer'].mean() * 100 if 'elite_scorer' in df.columns else None,
            'elite_usage_pct': df['elite_usage'].mean() * 100 if 'elite_usage' in df.columns else None
        }
    
    # Opponent metrics availability
    opponent_cols = [col for col in df.columns if col.startswith('opponent_')]
    if opponent_cols:
        insights['opponent_metrics'] = {
            'available_metrics': len(opponent_cols),
            'metrics_list': opponent_cols[:5]  # Show first 5
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
    print("Enhanced NBA Feature Engineering Module")
    print("Usage:")
    print("  from feature_engineer import NBAFeatureEngineer, create_feature_engineering_pipeline")
    print("  engineer = create_feature_engineering_pipeline()")
    print("  df_engineered = engineer.fit_transform(df)")