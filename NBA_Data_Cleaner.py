"""
NBA Player Performance Data Cleaning Module

A comprehensive, modular data cleaning pipeline for NBA player performance data
following 2025 data science best practices.

Author: Christopher Bratkovics
Created: 2025
Updated: Following sklearn-style transformers and modern Python patterns
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress common warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


@dataclass
class CleaningConfig:
    """Configuration class for data cleaning parameters."""
    
    # Validation thresholds
    max_minutes_per_game: float = 60.0
    max_reasonable_points: int = 100
    max_reasonable_rebounds: int = 30
    max_reasonable_assists: int = 25
    
    # Missing value handling
    fill_counting_stats_with_zero: bool = True
    drop_threshold_missing_pct: float = 95.0  # Drop columns missing >95% data
    
    # Outlier detection parameters
    outlier_method: str = "iqr"  # "iqr" or "zscore"
    outlier_threshold: float = 3.0  # IQR multiplier or z-score threshold
    outlier_action: str = "flag"  # "flag", "cap", or "remove"
    
    # Text cleaning
    standardize_positions: bool = True
    create_full_names: bool = True
    
    # Data consistency checks
    strict_validation: bool = True
    auto_fix_inconsistencies: bool = True
    
    # Default values
    default_first_game_rest: int = 7
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.outlier_method not in ["iqr", "zscore"]:
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")
        if self.outlier_action not in ["flag", "cap", "remove"]:
            raise ValueError("outlier_action must be 'flag', 'cap', or 'remove'")


class BaseNBATransformer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for NBA data transformers following sklearn patterns."""
    
    def __init__(self, config: Optional[CleaningConfig] = None, verbose: bool = True):
        self.config = config or CleaningConfig()
        self.verbose = verbose
        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False
    
    def _log(self, message: str, level: str = "info"):
        """Log messages if verbose is enabled."""
        if self.verbose:
            getattr(logger, level)(message)
    
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if self.is_fitted_:
            if hasattr(self, 'required_columns_'):
                missing_cols = set(self.required_columns_) - set(X.columns)
                if missing_cols and self.config.strict_validation:
                    raise ValueError(f"Missing required columns: {missing_cols}")
        
        return X.copy()
    
    def _store_input_info(self, X: pd.DataFrame) -> None:
        """Store information about input features."""
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = len(X.columns)
        self.is_fitted_ = True
    
    @abstractmethod
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Implementation of the transformation logic."""
        pass
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer."""
        X = self._validate_input(X)
        self._store_input_info(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data."""
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
        
        X = self._validate_input(X)
        return self._transform_impl(X)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


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
            >>> MinutesConverter.convert_to_decimal(30)
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


class DataTypeConverter(BaseNBATransformer):
    """Convert data types and format columns appropriately."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Define column type mappings
        self.id_columns = [
            'id', 'player_id', 'player_team_id', 'team_id', 
            'game_id', 'game_home_team_id', 'game_visitor_team_id'
        ]
        
        self.stat_columns = [
            'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct',
            'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb',
            'ast', 'stl', 'blk', 'turnover', 'pf', 'pts'
        ]
        
        self.text_columns = [
            'player_first_name', 'player_last_name', 'player_position',
            'team_abbreviation', 'team_full_name'
        ]
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for NBA statistics."""
        self._log("Converting data types...")
        
        # Convert ID columns to integers
        for col in self.id_columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').astype('Int64')
        
        # Convert statistical columns to numeric
        for col in self.stat_columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle minutes played
        if 'min' in X.columns:
            self._log("Converting minutes to decimal format...")
            X['minutes_played'] = X['min'].apply(MinutesConverter.convert_to_decimal)
            X = X.drop('min', axis=1)
        
        # Convert date column
        if 'game_date' in X.columns:
            X['game_date'] = pd.to_datetime(X['game_date'], errors='coerce')
        
        # Convert season to integer
        if 'game_season' in X.columns:
            X['game_season'] = pd.to_numeric(X['game_season'], errors='coerce').astype('Int64')
        
        # Convert boolean columns
        if 'game_postseason' in X.columns:
            X['game_postseason'] = X['game_postseason'].astype(bool)
        
        # Clean text columns
        for col in self.text_columns:
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip()
                X[col] = X[col].replace(['nan', 'None', ''], pd.NA)
        
        return X


class MissingValueHandler(BaseNBATransformer):
    """Handle missing values with basketball-specific logic."""
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on basketball context."""
        self._log("Handling missing values...")
        
        # Report missing values
        missing_summary = X.isnull().sum()
        missing_pct = (missing_summary / len(X)) * 100
        
        if self.verbose:
            significant_missing = missing_pct[missing_pct > 1]
            if len(significant_missing) > 0:
                self._log("Missing values summary:")
                for col, pct in significant_missing.items():
                    self._log(f"  {col}: {missing_summary[col]} ({pct:.2f}%)")
        
        # Handle percentage columns
        percentage_mappings = [
            ('fg_pct', 'fga', 'fgm'),
            ('fg3_pct', 'fg3a', 'fg3m'),
            ('ft_pct', 'fta', 'ftm')
        ]
        
        for pct_col, attempt_col, made_col in percentage_mappings:
            if all(col in X.columns for col in [pct_col, attempt_col]):
                # Set percentage to 0 when attempts = 0 and percentage is NaN
                zero_attempts_mask = (X[attempt_col] == 0) & (X[pct_col].isna())
                X.loc[zero_attempts_mask, pct_col] = 0.0
                
                # Recalculate percentages where missing but attempts > 0
                if made_col in X.columns:
                    missing_pct_mask = X[pct_col].isna() & (X[attempt_col] > 0)
                    if missing_pct_mask.any():
                        X.loc[missing_pct_mask, pct_col] = (
                            X.loc[missing_pct_mask, made_col] / 
                            X.loc[missing_pct_mask, attempt_col]
                        )
        
        # Handle missing statistical values
        if self.config.fill_counting_stats_with_zero:
            counting_stats = [
                'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk',
                'turnover', 'pf', 'pts'
            ]
            
            for col in counting_stats:
                if col in X.columns:
                    filled_count = X[col].isna().sum()
                    if filled_count > 0:
                        X[col] = X[col].fillna(0)
                        self._log(f"  Filled {filled_count} missing values in {col} with 0")
        
        # Handle missing minutes
        if 'minutes_played' in X.columns:
            X['minutes_played'] = X['minutes_played'].fillna(0)
        
        return X


class DataValidator(BaseNBATransformer):
    """Validate data consistency and fix impossible values."""
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data inconsistencies."""
        self._log("Performing data validation...")
        validation_issues = []
        
        # Minutes should be between 0 and max allowed
        if 'minutes_played' in X.columns:
            invalid_minutes = X[X['minutes_played'] > self.config.max_minutes_per_game]
            if len(invalid_minutes) > 0:
                validation_issues.append(f"Found {len(invalid_minutes)} records with >{self.config.max_minutes_per_game} minutes")
                if self.config.auto_fix_inconsistencies:
                    X.loc[X['minutes_played'] > self.config.max_minutes_per_game, 'minutes_played'] = self.config.max_minutes_per_game
        
        # Made shots shouldn't exceed attempted shots
        shot_checks = [('fgm', 'fga'), ('fg3m', 'fg3a'), ('ftm', 'fta')]
        for made_col, attempt_col in shot_checks:
            if made_col in X.columns and attempt_col in X.columns:
                invalid_shots = X[X[made_col] > X[attempt_col]]
                if len(invalid_shots) > 0:
                    validation_issues.append(f"Found {len(invalid_shots)} records where {made_col} > {attempt_col}")
                    if self.config.auto_fix_inconsistencies:
                        X.loc[X[made_col] > X[attempt_col], made_col] = X[attempt_col]
        
        # Total rebounds should equal offensive + defensive rebounds
        if all(col in X.columns for col in ['reb', 'oreb', 'dreb']):
            calculated_reb = X['oreb'] + X['dreb']
            reb_mismatch = X[abs(X['reb'] - calculated_reb) > 0.1]
            if len(reb_mismatch) > 0:
                validation_issues.append(f"Found {len(reb_mismatch)} records with rebound calculation mismatches")
                if self.config.auto_fix_inconsistencies:
                    X['reb'] = calculated_reb
        
        # Percentages should be between 0 and 1
        pct_columns = ['fg_pct', 'fg3_pct', 'ft_pct']
        for col in pct_columns:
            if col in X.columns:
                invalid_pct = X[(X[col] < 0) | (X[col] > 1)]
                if len(invalid_pct) > 0:
                    validation_issues.append(f"Found {len(invalid_pct)} records with invalid {col} values")
                    if self.config.auto_fix_inconsistencies:
                        X[col] = X[col].clip(0, 1)
        
        # Statistical sanity checks
        sanity_checks = [
            ('pts', self.config.max_reasonable_points),
            ('reb', self.config.max_reasonable_rebounds),
            ('ast', self.config.max_reasonable_assists)
        ]
        
        for col, max_val in sanity_checks:
            if col in X.columns:
                extreme_values = X[X[col] > max_val]
                if len(extreme_values) > 0:
                    validation_issues.append(f"Found {len(extreme_values)} records with extreme {col} values (>{max_val})")
        
        if validation_issues and self.verbose:
            self._log("Validation issues found:")
            for issue in validation_issues:
                self._log(f"  - {issue}")
        
        return X


class OutlierDetector(BaseNBATransformer):
    """Detect and handle outliers in statistical data."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
    
    def _calculate_outlier_bounds(self, series: pd.Series) -> Tuple[float, float]:
        """Calculate outlier bounds based on configured method."""
        if self.config.outlier_method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.outlier_threshold * IQR
            upper_bound = Q3 + self.config.outlier_threshold * IQR
        elif self.config.outlier_method == "zscore":
            mean = series.mean()
            std = series.std()
            lower_bound = mean - self.config.outlier_threshold * std
            upper_bound = mean + self.config.outlier_threshold * std
        else:
            raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")
        
        return lower_bound, upper_bound
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the outlier detector by calculating bounds."""
        X = self._validate_input(X)
        
        outlier_columns = ['pts', 'reb', 'ast', 'minutes_played', 'fga', 'fg3a']
        
        for col in outlier_columns:
            if col in X.columns:
                self.outlier_bounds_[col] = self._calculate_outlier_bounds(X[col].dropna())
        
        self._store_input_info(X)
        return self
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers."""
        self._log(f"Detecting outliers using {self.config.outlier_method} method...")
        outlier_summary = {}
        
        for col, (lower_bound, upper_bound) in self.outlier_bounds_.items():
            if col in X.columns:
                outliers_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                outlier_summary[col] = outlier_count
                
                if self.config.outlier_action == "flag":
                    X[f'{col}_outlier_flag'] = outliers_mask
                elif self.config.outlier_action == "cap":
                    X[col] = X[col].clip(lower_bound, upper_bound)
                elif self.config.outlier_action == "remove":
                    X = X[~outliers_mask]
        
        if self.verbose and outlier_summary:
            self._log("Outlier detection summary:")
            for col, count in outlier_summary.items():
                self._log(f"  {col}: {count} outliers detected")
        
        return X


class TextDataCleaner(BaseNBATransformer):
    """Clean and standardize text data."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.position_mapping = {
            'G': 'Guard',
            'F': 'Forward',
            'C': 'Center',
            'G-F': 'Guard-Forward',
            'F-G': 'Guard-Forward',
            'F-C': 'Forward-Center',
            'C-F': 'Forward-Center'
        }
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text data."""
        self._log("Cleaning text data...")
        
        text_columns = ['player_first_name', 'player_last_name', 'player_position', 
                       'team_abbreviation', 'team_full_name']
        
        for col in text_columns:
            if col in X.columns:
                # Strip whitespace and standardize
                X[col] = X[col].astype(str).str.strip()
                # Handle missing values in text columns
                X[col] = X[col].replace(['nan', 'None', ''], pd.NA)
        
        # Standardize position names
        if 'player_position' in X.columns and self.config.standardize_positions:
            X['player_position_standardized'] = (
                X['player_position']
                .map(self.position_mapping)
                .fillna(X['player_position'])
            )
        
        # Create full player name
        if (self.config.create_full_names and 
            all(col in X.columns for col in ['player_first_name', 'player_last_name'])):
            X['player_full_name'] = (
                X['player_first_name'].astype(str) + ' ' + 
                X['player_last_name'].astype(str)
            )
        
        return X


class FinalQualityChecker(BaseNBATransformer):
    """Perform final data quality checks and cleanup."""
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform final quality checks."""
        self._log("Performing final data quality assessment...")
        
        original_length = len(X)
        
        # Check for duplicate records
        if all(col in X.columns for col in ['player_id', 'game_id']):
            duplicate_check = X.duplicated(subset=['player_id', 'game_id']).sum()
            if duplicate_check > 0:
                self._log(f"Found and removing {duplicate_check} duplicate player-game combinations")
                X = X.drop_duplicates(subset=['player_id', 'game_id'])
        
        # Sort by date and player for consistency
        if all(col in X.columns for col in ['game_date', 'player_id']):
            X = X.sort_values(['game_date', 'player_id']).reset_index(drop=True)
        
        # Final missing value check
        final_missing = X.isnull().sum()
        critical_missing = final_missing[final_missing > 0]
        
        if len(critical_missing) > 0 and self.verbose:
            self._log("Remaining missing values:")
            for col, count in critical_missing.items():
                pct = (count / len(X)) * 100
                self._log(f"  {col}: {count} ({pct:.2f}%)")
        
        rows_removed = original_length - len(X)
        if rows_removed > 0:
            self._log(f"Removed {rows_removed} rows during final cleanup")
        
        return X


class NBADataCleaner:
    """
    Main data cleaning pipeline for NBA player performance data.
    
    This class orchestrates multiple transformers to provide comprehensive
    data cleaning for NBA player statistics.
    """
    
    def __init__(self, 
                 config: Optional[CleaningConfig] = None,
                 include_type_conversion: bool = True,
                 include_missing_value_handling: bool = True,
                 include_validation: bool = True,
                 include_outlier_detection: bool = True,
                 include_text_cleaning: bool = True,
                 include_final_checks: bool = True,
                 verbose: bool = True):
        """
        Initialize the data cleaning pipeline.
        
        Args:
            config: Configuration object (uses default if None)
            include_*: Boolean flags to enable/disable specific cleaning steps
            verbose: Whether to print progress messages
        """
        self.config = config or CleaningConfig()
        self.verbose = verbose
        
        # Initialize transformers
        self.transformers = []
        
        if include_type_conversion:
            self.transformers.append(('type_conversion', DataTypeConverter(config=self.config, verbose=verbose)))
        
        if include_missing_value_handling:
            self.transformers.append(('missing_values', MissingValueHandler(config=self.config, verbose=verbose)))
        
        if include_validation:
            self.transformers.append(('validation', DataValidator(config=self.config, verbose=verbose)))
        
        if include_outlier_detection:
            self.transformers.append(('outlier_detection', OutlierDetector(config=self.config, verbose=verbose)))
        
        if include_text_cleaning:
            self.transformers.append(('text_cleaning', TextDataCleaner(config=self.config, verbose=verbose)))
        
        if include_final_checks:
            self.transformers.append(('final_checks', FinalQualityChecker(config=self.config, verbose=verbose)))
        
        self.is_fitted_ = False
        self.cleaning_report_: Optional[Dict[str, Any]] = None
    
    def fit(self, X: pd.DataFrame, y=None) -> 'NBADataCleaner':
        """
        Fit the cleaning pipeline.
        
        Args:
            X: Input DataFrame with NBA player stats
            y: Target variable (ignored)
            
        Returns:
            self
        """
        if self.verbose:
            logger.info("Fitting NBA Data Cleaning Pipeline...")
            logger.info(f"Initial dataset shape: {X.shape}")
        
        X_temp = X.copy()
        
        # Fit transformers that need fitting (e.g., OutlierDetector)
        for name, transformer in self.transformers:
            if hasattr(transformer, 'fit') and hasattr(transformer, 'transform'):
                transformer.fit(X_temp)
                X_temp = transformer.transform(X_temp)
        
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info("Data cleaning pipeline fitted successfully!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data using fitted transformers.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before transform")
        
        if self.verbose:
            logger.info("Cleaning NBA data...")
            logger.info(f"Input shape: {X.shape}")
        
        X_clean = X.copy()
        original_shape = X_clean.shape
        
        # Apply transformers in sequence
        for name, transformer in self.transformers:
            try:
                X_clean = transformer.transform(X_clean)
                if self.verbose:
                    logger.debug(f"Applied {name} transformer")
            except Exception as e:
                logger.warning(f"Error applying {name} transformer: {e}")
                continue
        
        # Generate cleaning report
        self.cleaning_report_ = self._generate_cleaning_report(X, X_clean)
        
        if self.verbose:
            logger.info(f"Cleaning complete! Shape: {original_shape} → {X_clean.shape}")
            logger.info(f"Records removed: {original_shape[0] - X_clean.shape[0]}")
            logger.info(f"Columns added: {X_clean.shape[1] - original_shape[1]}")
        
        return X_clean
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the pipeline and clean the data in one step.
        
        Args:
            X: Input DataFrame
            y: Target variable (ignored)
            
        Returns:
            Cleaned DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _generate_cleaning_report(self, X_original: pd.DataFrame, X_cleaned: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive cleaning report."""
        return {
            'original_shape': X_original.shape,
            'cleaned_shape': X_cleaned.shape,
            'rows_removed': len(X_original) - len(X_cleaned),
            'columns_added': len(X_cleaned.columns) - len(X_original.columns),
            'missing_values_before': X_original.isnull().sum().sum(),
            'missing_values_after': X_cleaned.isnull().sum().sum(),
            'new_columns': [col for col in X_cleaned.columns if col not in X_original.columns],
            'removed_columns': [col for col in X_original.columns if col not in X_cleaned.columns],
            'cleaning_timestamp': pd.Timestamp.now(),
            'config_used': self.config.__dict__
        }
    
    def get_cleaning_report(self) -> Optional[Dict[str, Any]]:
        """Get the cleaning report from the last transform operation."""
        return self.cleaning_report_
    
    def save_cleaned_data(self, 
                         X_cleaned: pd.DataFrame, 
                         output_path: Union[str, Path], 
                         format: str = "parquet") -> Path:
        """
        Save cleaned data with metadata.
        
        Args:
            X_cleaned: Cleaned DataFrame
            output_path: Output file path
            format: Output format ("parquet", "csv", "feather")
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        # Add timestamp to filename if not present
        if not any(char.isdigit() for char in output_path.stem):
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"
        
        # Save data in specified format
        if format == "parquet":
            X_cleaned.to_parquet(output_path, index=False)
        elif format == "csv":
            X_cleaned.to_csv(output_path, index=False)
        elif format == "feather":
            X_cleaned.to_feather(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save cleaning report as JSON
        if self.cleaning_report_:
            report_path = output_path.parent / f"{output_path.stem}_cleaning_report.json"
            
            # Convert report to JSON-serializable format
            import json
            report_serializable = {}
            for key, value in self.cleaning_report_.items():
                if isinstance(value, pd.Timestamp):
                    report_serializable[key] = value.isoformat()
                elif isinstance(value, np.integer):
                    report_serializable[key] = int(value)
                elif isinstance(value, np.floating):
                    report_serializable[key] = float(value)
                else:
                    report_serializable[key] = value
            
            with open(report_path, 'w') as f:
                json.dump(report_serializable, f, indent=2)
            
            logger.info(f"Cleaning report saved to: {report_path}")
        
        logger.info(f"Cleaned data saved to: {output_path}")
        return output_path


# Convenience functions for common use cases
def create_basic_cleaner(strict_validation: bool = True, verbose: bool = True) -> NBADataCleaner:
    """Create a basic NBA data cleaner with sensible defaults."""
    config = CleaningConfig(
        strict_validation=strict_validation,
        auto_fix_inconsistencies=True,
        outlier_action="flag"
    )
    
    return NBADataCleaner(
        config=config,
        verbose=verbose
    )


def create_aggressive_cleaner(remove_outliers: bool = True, verbose: bool = True) -> NBADataCleaner:
    """Create an aggressive cleaner that removes outliers and fixes all issues."""
    config = CleaningConfig(
        strict_validation=True,
        auto_fix_inconsistencies=True,
        outlier_action="remove" if remove_outliers else "cap",
        outlier_threshold=2.5,  # More aggressive outlier detection
        drop_threshold_missing_pct=90.0
    )
    
    return NBADataCleaner(
        config=config,
        verbose=verbose
    )


def create_minimal_cleaner(verbose: bool = True) -> NBADataCleaner:
    """Create a minimal cleaner that only does essential cleaning."""
    config = CleaningConfig(
        strict_validation=False,
        auto_fix_inconsistencies=True,
        outlier_action="flag"
    )
    
    return NBADataCleaner(
        config=config,
        include_outlier_detection=False,
        include_text_cleaning=False,
        verbose=verbose
    )


# Analysis and visualization functions
def analyze_cleaning_impact(df_original: pd.DataFrame, 
                          df_cleaned: pd.DataFrame,
                          target_columns: List[str] = None) -> Dict[str, Any]:
    """
    Analyze the impact of data cleaning on the dataset.
    
    Args:
        df_original: Original dataset before cleaning
        df_cleaned: Dataset after cleaning
        target_columns: Specific columns to analyze (default: ['pts', 'reb', 'ast'])
        
    Returns:
        Dictionary containing analysis results
    """
    if target_columns is None:
        target_columns = ['pts', 'reb', 'ast']
    
    analysis = {
        'shape_change': {
            'before': df_original.shape,
            'after': df_cleaned.shape,
            'rows_removed': len(df_original) - len(df_cleaned),
            'columns_added': len(df_cleaned.columns) - len(df_original.columns)
        },
        'missing_data': {
            'before': df_original.isnull().sum().sum(),
            'after': df_cleaned.isnull().sum().sum(),
            'improvement': df_original.isnull().sum().sum() - df_cleaned.isnull().sum().sum()
        },
        'target_statistics': {}
    }
    
    # Analyze target columns
    for col in target_columns:
        if col in df_original.columns and col in df_cleaned.columns:
            analysis['target_statistics'][col] = {
                'mean_before': df_original[col].mean(),
                'mean_after': df_cleaned[col].mean(),
                'std_before': df_original[col].std(),
                'std_after': df_cleaned[col].std(),
                'missing_before': df_original[col].isnull().sum(),
                'missing_after': df_cleaned[col].isnull().sum()
            }
    
    return analysis


def plot_cleaning_comparison(df_original: pd.DataFrame,
                           df_cleaned: pd.DataFrame,
                           columns: List[str] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create visualizations comparing data before and after cleaning.
    
    Args:
        df_original: Original dataset
        df_cleaned: Cleaned dataset  
        columns: Columns to visualize (default: ['pts', 'reb', 'ast'])
        figsize: Figure size
    """
    if columns is None:
        columns = ['pts', 'reb', 'ast']
    
    # Filter to columns that exist in both datasets
    available_columns = [col for col in columns if col in df_original.columns and col in df_cleaned.columns]
    
    if not available_columns:
        logger.warning("No common columns found for comparison")
        return
    
    n_cols = len(available_columns)
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Data Cleaning Impact: Before vs After', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(available_columns):
        # Before cleaning - top row
        axes[0, i].hist(df_original[col].dropna(), bins=50, alpha=0.7, 
                       color='red', edgecolor='black', density=True)
        axes[0, i].set_title(f'{col.upper()} - Before Cleaning')
        axes[0, i].set_ylabel('Density')
        axes[0, i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_before = df_original[col].mean()
        std_before = df_original[col].std()
        axes[0, i].axvline(mean_before, color='darkred', linestyle='--', linewidth=2)
        axes[0, i].text(0.7, 0.9, f'μ={mean_before:.1f}\nσ={std_before:.1f}', 
                       transform=axes[0, i].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # After cleaning - bottom row
        axes[1, i].hist(df_cleaned[col].dropna(), bins=50, alpha=0.7,
                       color='blue', edgecolor='black', density=True)
        axes[1, i].set_title(f'{col.upper()} - After Cleaning')
        axes[1, i].set_xlabel(f'{col.capitalize()}')
        axes[1, i].set_ylabel('Density')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_after = df_cleaned[col].mean()
        std_after = df_cleaned[col].std()
        axes[1, i].axvline(mean_after, color='darkblue', linestyle='--', linewidth=2)
        axes[1, i].text(0.7, 0.9, f'μ={mean_after:.1f}\nσ={std_after:.1f}', 
                       transform=axes[1, i].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def validate_cleaned_data(df: pd.DataFrame, 
                         expected_columns: List[str] = None,
                         target_stats: List[str] = None) -> Dict[str, bool]:
    """
    Validate that cleaned data meets expected criteria.
    
    Args:
        df: Cleaned DataFrame
        expected_columns: List of columns that should be present
        target_stats: Target statistics to validate
        
    Returns:
        Dictionary of validation results
    """
    if target_stats is None:
        target_stats = ['pts', 'reb', 'ast']
    
    if expected_columns is None:
        expected_columns = ['player_id', 'game_id', 'game_date'] + target_stats
    
    validation_results = {}
    
    # Check required columns exist
    missing_columns = [col for col in expected_columns if col not in df.columns]
    validation_results['has_required_columns'] = len(missing_columns) == 0
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if 'game_date' in df.columns:
        validation_results['date_column_is_datetime'] = pd.api.types.is_datetime64_any_dtype(df['game_date'])
    
    # Check for reasonable data ranges
    validation_results['reasonable_data_ranges'] = True
    for stat in target_stats:
        if stat in df.columns:
            if stat == 'pts' and (df[stat].max() > 150 or df[stat].min() < 0):
                validation_results['reasonable_data_ranges'] = False
            elif stat in ['reb', 'ast'] and (df[stat].max() > 50 or df[stat].min() < 0):
                validation_results['reasonable_data_ranges'] = False
    
    # Check for excessive missing data
    missing_pct = (df.isnull().sum() / len(df) * 100).max()
    validation_results['acceptable_missing_data'] = missing_pct < 10.0
    
    # Check for duplicates
    if all(col in df.columns for col in ['player_id', 'game_id']):
        duplicate_count = df.duplicated(subset=['player_id', 'game_id']).sum()
        validation_results['no_duplicates'] = duplicate_count == 0
    
    # Overall validation
    validation_results['overall_valid'] = all(validation_results.values())
    
    return validation_results


# Integration functions for use in Jupyter notebooks
def quick_clean_nba_data(df: pd.DataFrame, 
                        cleaning_level: str = "standard",
                        save_path: Optional[Union[str, Path]] = None,
                        show_plots: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick cleaning function for use in Jupyter notebooks.
    
    Args:
        df: Input DataFrame
        cleaning_level: "minimal", "standard", or "aggressive"
        save_path: Optional path to save cleaned data
        show_plots: Whether to show before/after plots
        
    Returns:
        Tuple of (cleaned_dataframe, cleaning_report)
    """
    print(f"🧹 Quick NBA Data Cleaning (Level: {cleaning_level.upper()})")
    print("=" * 50)
    
    # Select cleaner based on level
    if cleaning_level == "minimal":
        cleaner = create_minimal_cleaner()
    elif cleaning_level == "aggressive":
        cleaner = create_aggressive_cleaner()
    else:  # standard
        cleaner = create_basic_cleaner()
    
    # Store original for comparison
    df_original = df.copy()
    
    # Clean the data
    df_cleaned = cleaner.fit_transform(df)
    
    # Get cleaning report
    cleaning_report = cleaner.get_cleaning_report()
    
    # Analyze impact
    analysis = analyze_cleaning_impact(df_original, df_cleaned)
    
    # Print summary
    print(f"\n✅ Cleaning Complete!")
    print(f"   Shape: {analysis['shape_change']['before']} → {analysis['shape_change']['after']}")
    print(f"   Rows removed: {analysis['shape_change']['rows_removed']:,}")
    print(f"   Missing values: {analysis['missing_data']['before']:,} → {analysis['missing_data']['after']:,}")
    
    # Show plots if requested
    if show_plots:
        plot_cleaning_comparison(df_original, df_cleaned)
    
    # Save if path provided
    if save_path:
        saved_path = cleaner.save_cleaned_data(df_cleaned, save_path)
        print(f"💾 Data saved to: {saved_path}")
    
    # Validate cleaned data
    validation = validate_cleaned_data(df_cleaned)
    if validation['overall_valid']:
        print("✅ Data validation passed!")
    else:
        print("⚠️  Data validation issues found:")
        for check, passed in validation.items():
            if not passed:
                print(f"   ❌ {check}")
    
    return df_cleaned, cleaning_report


# Example usage and demo function
def demo_nba_data_cleaning():
    """Demonstrate the NBA data cleaning pipeline."""
    print("🏀 NBA Data Cleaning Pipeline Demo")
    print("=" * 40)
    
    # Create sample data with various issues
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'player_id': np.random.randint(1, 101, n_samples),
        'game_id': np.random.randint(1000, 2000, n_samples),
        'game_date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'pts': np.random.poisson(15, n_samples),
        'reb': np.random.poisson(7, n_samples),
        'ast': np.random.poisson(5, n_samples),
        'min': [f"{np.random.randint(10, 45)}:{np.random.randint(0, 60):02d}" for _ in range(n_samples)],
        'fga': np.random.poisson(12, n_samples),
        'fgm': np.random.poisson(6, n_samples),
        'player_position': np.random.choice(['G', 'F', 'C', 'G-F'], n_samples),
        'team_id': np.random.randint(1, 31, n_samples),
        'game_home_team_id': np.random.randint(1, 31, n_samples)
    })
    
    # Introduce some data quality issues
    # Missing values
    missing_indices = np.random.choice(n_samples, 50, replace=False)
    sample_data.loc[missing_indices, 'pts'] = np.nan
    
    # Impossible values
    sample_data.loc[sample_data.index[:10], 'fgm'] = sample_data.loc[sample_data.index[:10], 'fga'] + 5
    
    # Outliers
    sample_data.loc[sample_data.index[:5], 'pts'] = 150  # Impossible high scores
    
    print(f"📊 Created sample dataset with issues: {sample_data.shape}")
    print(f"   Missing values: {sample_data.isnull().sum().sum()}")
    print(f"   Data issues: FGM > FGA in first 10 rows, extreme scores in first 5 rows")
    
    # Clean the data
    df_cleaned, report = quick_clean_nba_data(
        sample_data, 
        cleaning_level="standard",
        show_plots=False  # Set to True to see plots
    )
    
    print(f"\n📈 Cleaning Results:")
    print(f"   Original shape: {report['original_shape']}")
    print(f"   Cleaned shape: {report['cleaned_shape']}")
    print(f"   New columns: {report['new_columns']}")
    
    return df_cleaned, report


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_nba_data_cleaning()