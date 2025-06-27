"""
NBA Hypothesis Testing Module

Author: Christopher Bratkovics
Created: 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NBAHypothesisTester:
    """
    Conducts statistical hypothesis tests on NBA player performance data.
    
    Implements three key hypothesis tests:
    1. Impact of rest days on shooting efficiency
    2. Home court advantage for individual scoring
    3. Evolution of three-point shooting trends over time
    
    Uses both parametric and non-parametric testing methods to ensure
    robust statistical conclusions.
    """
    
    def __init__(self, player_stats_df: pd.DataFrame, games_df: pd.DataFrame = None):
        """
        Initialize the hypothesis tester with NBA data.
        
        Args:
            player_stats_df: DataFrame containing player game statistics
            games_df: Optional DataFrame containing game-level details
        """
        self.player_stats = player_stats_df.copy()
        self.games_df = games_df.copy() if games_df is not None else None
        self.results = {}
        
        # Prepare data for statistical analysis
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare and validate data for hypothesis testing.
        
        Handles both raw and feature-engineered data by detecting available columns
        and creating necessary features if they don't exist.
        """
        print("Preparing data for hypothesis testing...")
        
        # Convert date column to datetime format for time-based calculations
        if 'game_date' in self.player_stats.columns:
            self.player_stats['game_date'] = pd.to_datetime(self.player_stats['game_date'])
        
        # Sort data chronologically by player for accurate rest day calculations
        self.player_stats = self.player_stats.sort_values(['player_id', 'game_date'])
        
        # Detect if data has already undergone feature engineering
        feature_engineered = self._check_feature_engineering_status()
        
        if feature_engineered:
            print("Data appears to be feature engineered already")
            # Map feature engineered column names to standard names
            self._map_feature_engineered_columns()
        else:
            print("Data not feature engineered - creating required features")
            # Calculate per-36 minute statistics for fair comparison
            self.player_stats = self._calculate_per_36_stats()
            
            # Calculate rest days between consecutive games
            self.player_stats['rest_days'] = self._calculate_rest_days()
            
            # Determine home/away game status
            self.player_stats['is_home_game'] = self._determine_home_away_status()
        
        print(f"Data preparation complete. Dataset shape: {self.player_stats.shape}")
        
        # Display available relevant columns for debugging
        relevant_cols = [col for col in self.player_stats.columns 
                        if any(x in col for x in ['rest', 'home', 'minutes', 'fg3a_per_36'])]
        print(f"Key columns available: {relevant_cols}")
        
        # Verify all required columns exist for hypothesis tests
        self._verify_required_columns()
    
    def _check_feature_engineering_status(self) -> bool:
        """
        Check if the data has already been feature engineered.
        
        Returns:
            True if at least 3 feature engineering indicators are present
        """
        feature_eng_indicators = [
            'minutes_played', 'rest_days', 'sufficient_rest', 
            'is_home_game', 'fg3a_per_36min'
        ]
        
        present_indicators = sum(1 for col in feature_eng_indicators if col in self.player_stats.columns)
        return present_indicators >= 3
    
    def _map_feature_engineered_columns(self):
        """Map feature engineered column names to standardized names for consistency."""
        column_mapping = {
            'minutes_played': 'minutes_numeric',
            'fg3a_per_36min': 'fg3a_per_36'
        }
        
        for feature_col, expected_col in column_mapping.items():
            if feature_col in self.player_stats.columns and expected_col not in self.player_stats.columns:
                self.player_stats[expected_col] = self.player_stats[feature_col]
                print(f"Mapped {feature_col} to {expected_col}")
    
    def _verify_required_columns(self):
        """Verify that all required columns exist for each hypothesis test."""
        required_for_tests = {
            'hypothesis_1': ['rest_days', 'fg_pct', 'fga'],
            'hypothesis_2': ['is_home_game', 'pts'],
            'hypothesis_3': ['fg3a_per_36', 'game_season', 'minutes_numeric']
        }
        
        for test_name, required_cols in required_for_tests.items():
            missing_cols = [col for col in required_cols if col not in self.player_stats.columns]
            if missing_cols:
                print(f"WARNING - {test_name}: Missing columns {missing_cols}")
            else:
                print(f"READY - {test_name}: All required columns present")
    
    def _convert_minutes_to_float(self, minutes_str) -> float:
        """
        Convert minutes from various formats to decimal.
        
        Handles string formats like '30:00' and numeric inputs.
        
        Args:
            minutes_str: Minutes in string or numeric format
            
        Returns:
            Minutes as float (e.g., 30.5 for 30:30)
        """
        if pd.isna(minutes_str) or minutes_str == '' or minutes_str is None:
            return 0.0
        
        try:
            # Handle numeric inputs directly
            if isinstance(minutes_str, (int, float)):
                return float(minutes_str)
            
            # Convert to string for consistent processing
            minutes_str = str(minutes_str)
            
            # Parse MM:SS format
            if ':' in minutes_str:
                parts = minutes_str.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes + (seconds / 60)
                else:
                    return float(parts[0])  # Handle malformed time strings
            else:
                # Direct numeric conversion
                return float(minutes_str)
        except (ValueError, TypeError, AttributeError):
            # Return 0 if conversion fails
            print(f"Warning: Could not convert minutes value: {minutes_str}")
            return 0.0
    
    def _calculate_rest_days(self) -> pd.Series:
        """
        Calculate rest days between games for each player.
        
        Rest days = calendar days between games - 1
        First game for each player has NaN rest days.
        
        Returns:
            Series with rest days for each game
        """
        rest_days = []
        
        for player_id in self.player_stats['player_id'].unique():
            player_games = self.player_stats[self.player_stats['player_id'] == player_id].copy()
            player_games = player_games.sort_values('game_date')
            
            # First game has no previous game for comparison
            player_rest_days = [np.nan]
            
            for i in range(1, len(player_games)):
                current_date = player_games.iloc[i]['game_date']
                previous_date = player_games.iloc[i-1]['game_date']
                # Rest days = days between games minus 1
                days_diff = (current_date - previous_date).days - 1
                player_rest_days.append(max(0, days_diff))  # Ensure non-negative
            
            rest_days.extend(player_rest_days)
        
        return pd.Series(rest_days, index=self.player_stats.index)
    
    def _determine_home_away_status(self) -> pd.Series:
        """
        Determine if each game was played at home or away.
        
        A player is at home when their team_id matches the game's home_team_id.
        
        Returns:
            Boolean Series indicating home game status
        """
        if 'game_home_team_id' in self.player_stats.columns and 'team_id' in self.player_stats.columns:
            return self.player_stats['team_id'] == self.player_stats['game_home_team_id']
        else:
            print("Warning: Cannot determine home/away status. Missing required columns.")
            return pd.Series([np.nan] * len(self.player_stats), index=self.player_stats.index)
    
    def _calculate_per_36_stats(self) -> pd.DataFrame:
        """
        Calculate per-36 minute statistics for fair comparison across different playing times.
        
        Per-36 stats are the NBA standard for comparing players with different minutes.
        
        Returns:
            DataFrame with per-36 statistics added
        """
        df = self.player_stats.copy()
        
        # Convert minutes to numeric format
        if 'min' in df.columns:
            print("Converting minutes to numeric format...")
            df['minutes_numeric'] = df['min'].apply(self._convert_minutes_to_float)
            
            # Calculate per-36 stats for key metrics
            stats_to_convert = ['fg3a', 'pts', 'reb', 'ast']
            
            for stat in stats_to_convert:
                if stat in df.columns:
                    df[f'{stat}_per_36'] = np.where(
                        df['minutes_numeric'] > 0,
                        (df[stat] / df['minutes_numeric']) * 36,
                        np.nan
                    )
                    print(f"Created {stat}_per_36 column")
                else:
                    print(f"Warning: {stat} column not found in data")
            
            print(f"Minutes conversion complete. Sample minutes_numeric values:")
            print(df['minutes_numeric'].describe())
        else:
            print("Warning: 'min' column not found. Cannot calculate per-36 stats.")
            # Create default column to prevent errors
            df['minutes_numeric'] = 0.0
        
        return df
    
    def hypothesis_1_rest_days_shooting(self, min_fga: int = 5, alpha: float = 0.05) -> Dict:
        """
        Test Hypothesis 1: Impact of Rest Days on Shooting Efficiency
        
        H0: No significant difference in FG% between well-rested and not well-rested players
        H1: Significant difference exists
        
        Args:
            min_fga: Minimum field goal attempts to include a game (default: 5)
            alpha: Significance level for hypothesis testing (default: 0.05)
            
        Returns:
            Dictionary containing test results and statistics
        """
        print("\n" + "="*60)
        print("HYPOTHESIS 1: IMPACT OF REST DAYS ON SHOOTING EFFICIENCY")
        print("="*60)
        
        # Check for rest day columns (feature engineered or raw)
        rest_col = 'rest_days'
        sufficient_rest_col = 'sufficient_rest'
        
        if sufficient_rest_col in self.player_stats.columns:
            print(f"Using feature engineered column: {sufficient_rest_col}")
            # Use binary sufficient rest indicator
            valid_data = self.player_stats[
                (self.player_stats['fga'] >= min_fga) &
                (self.player_stats[sufficient_rest_col].notna()) &
                (self.player_stats['fg_pct'].notna())
            ].copy()
            
            well_rested = valid_data[valid_data[sufficient_rest_col] == True]['fg_pct']
            not_well_rested = valid_data[valid_data[sufficient_rest_col] == False]['fg_pct']
            
        elif rest_col in self.player_stats.columns:
            print(f"Using rest days column: {rest_col}")
            # Filter for valid shooting data
            valid_data = self.player_stats[
                (self.player_stats['fga'] >= min_fga) &
                (self.player_stats[rest_col].notna()) &
                (self.player_stats['fg_pct'].notna())
            ].copy()
            
            # Split into rest groups (2+ days considered sufficient rest)
            well_rested = valid_data[valid_data[rest_col] >= 2]['fg_pct']
            not_well_rested = valid_data[valid_data[rest_col] < 2]['fg_pct']
        else:
            print("Error: No rest days columns found!")
            available_rest_cols = [col for col in self.player_stats.columns if 'rest' in col.lower()]
            print(f"Available columns with 'rest': {available_rest_cols}")
            return {'error': 'Missing rest days data'}
        
        print(f"Sample sizes:")
        print(f"  Well-rested (2+ days): {len(well_rested)} games")
        print(f"  Not well-rested (<2 days): {len(not_well_rested)} games")
        print(f"  Minimum FGA filter: {min_fga}")
        
        if len(well_rested) == 0 or len(not_well_rested) == 0:
            print("Error: One or both groups have no data")
            return {'error': 'Insufficient data for rest day comparison'}
        
        # Calculate descriptive statistics
        print(f"\nDescriptive Statistics:")
        print(f"  Well-rested FG%: Mean={well_rested.mean():.3f}, Std={well_rested.std():.3f}")
        print(f"  Not well-rested FG%: Mean={not_well_rested.mean():.3f}, Std={not_well_rested.std():.3f}")
        
        # Test assumptions for parametric testing
        print(f"\nAssumption Checks:")
        
        # Normality tests (use appropriate test based on sample size)
        if len(well_rested) < 5000:
            _, p_norm1 = stats.shapiro(well_rested.sample(min(len(well_rested), 5000)))
            _, p_norm2 = stats.shapiro(not_well_rested.sample(min(len(not_well_rested), 5000)))
        else:
            _, p_norm1 = stats.normaltest(well_rested)
            _, p_norm2 = stats.normaltest(not_well_rested)
        
        print(f"  Normality p-values: Well-rested={p_norm1:.6f}, Not well-rested={p_norm2:.6f}")
        
        # Test for equal variances
        _, p_var = stats.levene(well_rested, not_well_rested)
        print(f"  Equal variance test p-value: {p_var:.6f}")
        
        # Perform statistical tests
        results = {}
        
        # Independent t-test (parametric)
        equal_var = p_var > alpha
        t_stat, t_p = stats.ttest_ind(well_rested, not_well_rested, equal_var=equal_var)
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p = stats.mannwhitneyu(well_rested, not_well_rested, alternative='two-sided')
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(well_rested) - 1) * well_rested.var() + 
                             (len(not_well_rested) - 1) * not_well_rested.var()) / 
                            (len(well_rested) + len(not_well_rested) - 2))
        cohens_d = (well_rested.mean() - not_well_rested.mean()) / pooled_std
        
        print(f"\nHypothesis Test Results:")
        print(f"  T-test: t={t_stat:.4f}, p={t_p:.6f}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_p:.6f}")
        print(f"  Cohen's d (effect size): {cohens_d:.4f}")
        
        # Interpret results
        significant = t_p < alpha
        print(f"\nConclusion (α = {alpha}):")
        if significant:
            print(f"  REJECT H0: Significant difference found (p = {t_p:.6f})")
            direction = "better" if well_rested.mean() > not_well_rested.mean() else "worse"
            print(f"  Well-rested players shoot {direction} than those with less rest")
        else:
            print(f"  FAIL TO REJECT H0: No significant difference (p = {t_p:.6f})")
        
        # Store comprehensive results
        results = {
            'hypothesis': 'Rest Days Impact on Shooting Efficiency',
            'sample_sizes': {'well_rested': len(well_rested), 'not_well_rested': len(not_well_rested)},
            'descriptive_stats': {
                'well_rested_mean': well_rested.mean(),
                'not_well_rested_mean': not_well_rested.mean(),
                'difference': well_rested.mean() - not_well_rested.mean()
            },
            'test_statistics': {
                't_statistic': t_stat,
                't_p_value': t_p,
                'u_statistic': u_stat,
                'u_p_value': u_p,
                'cohens_d': cohens_d
            },
            'significant': significant,
            'alpha': alpha,
            'conclusion': 'Reject H0' if significant else 'Fail to reject H0'
        }
        
        self.results['hypothesis_1'] = results
        return results
    
    def hypothesis_2_home_away_scoring(self, alpha: float = 0.05) -> Dict:
        """
        Test Hypothesis 2: Home vs. Away Performance Differentials
        
        H0: Mean points scored at home equals mean points scored away
        H1: Mean points scored at home differs from mean points scored away
        
        Args:
            alpha: Significance level for hypothesis testing (default: 0.05)
            
        Returns:
            Dictionary containing test results and statistics
        """
        print("\n" + "="*60)
        print("HYPOTHESIS 2: HOME VS. AWAY PERFORMANCE DIFFERENTIALS")
        print("="*60)
        
        # Filter for valid scoring data with home/away status
        valid_data = self.player_stats[
            (self.player_stats['pts'].notna()) &
            (self.player_stats['is_home_game'].notna())
        ].copy()
        
        home_points = valid_data[valid_data['is_home_game'] == True]['pts']
        away_points = valid_data[valid_data['is_home_game'] == False]['pts']
        
        print(f"Sample sizes:")
        print(f"  Home games: {len(home_points)} player-games")
        print(f"  Away games: {len(away_points)} player-games")
        
        # Calculate descriptive statistics
        print(f"\nDescriptive Statistics:")
        print(f"  Home points: Mean={home_points.mean():.2f}, Std={home_points.std():.2f}")
        print(f"  Away points: Mean={away_points.mean():.2f}, Std={away_points.std():.2f}")
        
        # Test assumptions
        print(f"\nAssumption Checks:")
        
        # Normality tests
        if len(home_points) < 5000:
            _, p_norm1 = stats.shapiro(home_points.sample(min(len(home_points), 5000)))
            _, p_norm2 = stats.shapiro(away_points.sample(min(len(away_points), 5000)))
        else:
            _, p_norm1 = stats.normaltest(home_points)
            _, p_norm2 = stats.normaltest(away_points)
        
        print(f"  Normality p-values: Home={p_norm1:.6f}, Away={p_norm2:.6f}")
        
        # Equal variance test
        _, p_var = stats.levene(home_points, away_points)
        print(f"  Equal variance test p-value: {p_var:.6f}")
        
        # Perform statistical tests
        equal_var = p_var > alpha
        t_stat, t_p = stats.ttest_ind(home_points, away_points, equal_var=equal_var)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(home_points, away_points, alternative='two-sided')
        
        # Calculate effect size
        pooled_std = np.sqrt(((len(home_points) - 1) * home_points.var() + 
                             (len(away_points) - 1) * away_points.var()) / 
                            (len(home_points) + len(away_points) - 2))
        cohens_d = (home_points.mean() - away_points.mean()) / pooled_std
        
        print(f"\nHypothesis Test Results:")
        print(f"  T-test: t={t_stat:.4f}, p={t_p:.6f}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_p:.6f}")
        print(f"  Cohen's d (effect size): {cohens_d:.4f}")
        
        # Interpret results
        significant = t_p < alpha
        print(f"\nConclusion (α = {alpha}):")
        if significant:
            print(f"  REJECT H0: Significant difference found (p = {t_p:.6f})")
            direction = "higher" if home_points.mean() > away_points.mean() else "lower"
            print(f"  Home scoring is {direction} than away scoring")
        else:
            print(f"  FAIL TO REJECT H0: No significant difference (p = {t_p:.6f})")
        
        # Store results
        results = {
            'hypothesis': 'Home vs. Away Performance Differentials',
            'sample_sizes': {'home_games': len(home_points), 'away_games': len(away_points)},
            'descriptive_stats': {
                'home_mean': home_points.mean(),
                'away_mean': away_points.mean(),
                'difference': home_points.mean() - away_points.mean()
            },
            'test_statistics': {
                't_statistic': t_stat,
                't_p_value': t_p,
                'u_statistic': u_stat,
                'u_p_value': u_p,
                'cohens_d': cohens_d
            },
            'significant': significant,
            'alpha': alpha,
            'conclusion': 'Reject H0' if significant else 'Fail to reject H0'
        }
        
        self.results['hypothesis_2'] = results
        return results
    
    def hypothesis_3_three_point_evolution(self, min_minutes: int = 20, alpha: float = 0.05) -> Dict:
        """
        Test Hypothesis 3: Evolution of 3-Point Attempt Rates
        
        H0: Mean 3PA per 36 minutes in 2023-24 equals mean in 2021-22
        H1: Mean 3PA per 36 minutes in 2023-24 is greater than in 2021-22
        
        Args:
            min_minutes: Minimum minutes played to include a game (default: 20)
            alpha: Significance level for hypothesis testing (default: 0.05)
            
        Returns:
            Dictionary containing test results and statistics
        """
        print("\n" + "="*60)
        print("HYPOTHESIS 3: EVOLUTION OF 3-POINT ATTEMPT RATES")
        print("="*60)
        
        # Look for 3PA per 36 column (may have different names)
        fg3a_per_36_col = None
        minutes_col = None
        
        # Check for 3PA per 36 column variants
        possible_fg3a_cols = ['fg3a_per_36min', 'fg3a_per_36', 'fg3a_per_36_minutes']
        for col in possible_fg3a_cols:
            if col in self.player_stats.columns:
                fg3a_per_36_col = col
                print(f"Using 3PA per 36 column: {fg3a_per_36_col}")
                break
        
        # Check for minutes column variants
        possible_minutes_cols = ['minutes_played', 'minutes_numeric', 'min']
        for col in possible_minutes_cols:
            if col in self.player_stats.columns:
                minutes_col = col
                print(f"Using minutes column: {minutes_col}")
                break
        
        # Create per-36 stats if not present
        if fg3a_per_36_col is None:
            if 'fg3a' in self.player_stats.columns and minutes_col is not None:
                print("Creating fg3a_per_36 column from fg3a and minutes...")
                
                # Convert minutes if needed
                if minutes_col == 'min' or self.player_stats[minutes_col].dtype == 'object':
                    self.player_stats['minutes_numeric'] = self.player_stats[minutes_col].apply(self._convert_minutes_to_float)
                    minutes_col = 'minutes_numeric'
                
                # Calculate per-36 3PA
                self.player_stats['fg3a_per_36'] = np.where(
                    self.player_stats[minutes_col] > 0,
                    (self.player_stats['fg3a'] / self.player_stats[minutes_col]) * 36,
                    np.nan
                )
                fg3a_per_36_col = 'fg3a_per_36'
                print(f"Created {fg3a_per_36_col} column")
            else:
                print("Error: Cannot create fg3a_per_36 - missing required columns")
                print(f"Available columns: fg3a={'fg3a' in self.player_stats.columns}, minutes={minutes_col}")
                return {'error': 'Missing 3PA or minutes data'}
        
        # Verify required columns exist
        required_cols = ['game_season', fg3a_per_36_col]
        missing_cols = [col for col in required_cols if col not in self.player_stats.columns]
        
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print("Available columns:", list(self.player_stats.columns))
            return {'error': 'Missing required columns'}
        
        # Filter data for comparison seasons with minimum minutes
        print(f"Filtering data for seasons 2022 and 2024 with min {min_minutes} minutes...")
        
        # Convert minutes for filtering if needed
        if minutes_col and minutes_col in self.player_stats.columns:
            if minutes_col == 'min' or self.player_stats[minutes_col].dtype == 'object':
                minutes_for_filter = self.player_stats[minutes_col].apply(self._convert_minutes_to_float)
            else:
                minutes_for_filter = self.player_stats[minutes_col]
        else:
            print("Warning: No minutes column found for filtering - proceeding without minutes filter")
            minutes_for_filter = pd.Series([min_minutes] * len(self.player_stats))
        
        valid_data = self.player_stats[
            (self.player_stats['game_season'].isin([2022, 2024])) &  # 2021-22 and 2023-24 seasons
            (minutes_for_filter >= min_minutes) &
            (self.player_stats[fg3a_per_36_col].notna())
        ].copy()
        
        if len(valid_data) == 0:
            print("Error: No valid data found after filtering")
            print(f"Available seasons: {sorted(self.player_stats['game_season'].unique())}")
            if minutes_col:
                print(f"Minutes range: {minutes_for_filter.min()} - {minutes_for_filter.max()}")
            print(f"{fg3a_per_36_col} non-null count: {self.player_stats[fg3a_per_36_col].notna().sum()}")
            return {'error': 'No valid data after filtering'}
        
        season_2022 = valid_data[valid_data['game_season'] == 2022][fg3a_per_36_col]
        season_2024 = valid_data[valid_data['game_season'] == 2024][fg3a_per_36_col]
        
        print(f"Sample sizes (min {min_minutes} minutes):")
        print(f"  2021-22 season: {len(season_2022)} player-games")
        print(f"  2023-24 season: {len(season_2024)} player-games")
        
        if len(season_2022) == 0 or len(season_2024) == 0:
            print("Error: One or both seasons have no data")
            return {'error': 'Insufficient data for one or both seasons'}
        
        # Calculate descriptive statistics
        print(f"\nDescriptive Statistics:")
        print(f"  2021-22 3PA/36: Mean={season_2022.mean():.2f}, Std={season_2022.std():.2f}")
        print(f"  2023-24 3PA/36: Mean={season_2024.mean():.2f}, Std={season_2024.std():.2f}")
        
        # Test assumptions
        print(f"\nAssumption Checks:")
        
        # Normality tests
        if len(season_2022) < 5000:
            _, p_norm1 = stats.shapiro(season_2022.sample(min(len(season_2022), 5000)))
            _, p_norm2 = stats.shapiro(season_2024.sample(min(len(season_2024), 5000)))
        else:
            _, p_norm1 = stats.normaltest(season_2022)
            _, p_norm2 = stats.normaltest(season_2024)
        
        print(f"  Normality p-values: 2021-22={p_norm1:.6f}, 2023-24={p_norm2:.6f}")
        
        # Equal variance test
        _, p_var = stats.levene(season_2022, season_2024)
        print(f"  Equal variance test p-value: {p_var:.6f}")
        
        # Perform one-tailed tests (testing if 2024 > 2022)
        equal_var = p_var > alpha
        t_stat, t_p_two_tailed = stats.ttest_ind(season_2024, season_2022, equal_var=equal_var)
        # Convert to one-tailed p-value
        t_p_one_tailed = t_p_two_tailed / 2 if t_stat > 0 else 1 - (t_p_two_tailed / 2)
        
        # Mann-Whitney U test (one-tailed)
        u_stat, u_p_two_tailed = stats.mannwhitneyu(season_2024, season_2022, alternative='two-sided')
        u_stat_one, u_p_one_tailed = stats.mannwhitneyu(season_2024, season_2022, alternative='greater')
        
        # Calculate effect size
        pooled_std = np.sqrt(((len(season_2022) - 1) * season_2022.var() + 
                             (len(season_2024) - 1) * season_2024.var()) / 
                            (len(season_2022) + len(season_2024) - 2))
        cohens_d = (season_2024.mean() - season_2022.mean()) / pooled_std
        
        print(f"\nHypothesis Test Results (One-tailed):")
        print(f"  T-test: t={t_stat:.4f}, p={t_p_one_tailed:.6f}")
        print(f"  Mann-Whitney U: U={u_stat_one:.0f}, p={u_p_one_tailed:.6f}")
        print(f"  Cohen's d (effect size): {cohens_d:.4f}")
        
        # Interpret results
        significant = t_p_one_tailed < alpha
        print(f"\nConclusion (α = {alpha}, one-tailed test):")
        if significant:
            print(f"  REJECT H0: 3PA rates significantly increased (p = {t_p_one_tailed:.6f})")
            increase = season_2024.mean() - season_2022.mean()
            print(f"  Average increase: {increase:.2f} 3PA per 36 minutes")
        else:
            print(f"  FAIL TO REJECT H0: No significant increase (p = {t_p_one_tailed:.6f})")
        
        # Store results
        results = {
            'hypothesis': 'Evolution of 3-Point Attempt Rates',
            'sample_sizes': {'season_2022': len(season_2022), 'season_2024': len(season_2024)},
            'descriptive_stats': {
                'season_2022_mean': season_2022.mean(),
                'season_2024_mean': season_2024.mean(),
                'difference': season_2024.mean() - season_2022.mean()
            },
            'test_statistics': {
                't_statistic': t_stat,
                't_p_value_one_tailed': t_p_one_tailed,
                'u_statistic': u_stat_one,
                'u_p_value_one_tailed': u_p_one_tailed,
                'cohens_d': cohens_d
            },
            'significant': significant,
            'alpha': alpha,
            'conclusion': 'Reject H0' if significant else 'Fail to reject H0'
        }
        
        self.results['hypothesis_3'] = results
        return results
    
    def run_all_tests(self, min_fga: int = 5, min_minutes: int = 20, alpha: float = 0.05) -> Dict:
        """
        Run all three hypothesis tests in sequence.
        
        Args:
            min_fga: Minimum field goal attempts for shooting efficiency test
            min_minutes: Minimum minutes for 3-point evolution test
            alpha: Significance level for all tests
            
        Returns:
            Dictionary containing all test results
        """
        print("RUNNING ALL NBA HYPOTHESIS TESTS")
        print("=" * 80)
        
        # Execute all hypothesis tests
        self.hypothesis_1_rest_days_shooting(min_fga=min_fga, alpha=alpha)
        self.hypothesis_2_home_away_scoring(alpha=alpha)
        self.hypothesis_3_three_point_evolution(min_minutes=min_minutes, alpha=alpha)
        
        # Display summary of results
        print("\n" + "="*80)
        print("SUMMARY OF ALL HYPOTHESIS TESTS")
        print("="*80)
        
        for i, (key, result) in enumerate(self.results.items(), 1):
            print(f"\nHypothesis {i}: {result['hypothesis']}")
            print(f"  Result: {result['conclusion']}")
            if 't_p_value_one_tailed' in result['test_statistics']:
                print(f"  P-value: {result['test_statistics']['t_p_value_one_tailed']:.6f}")
            else:
                print(f"  P-value: {result['test_statistics']['t_p_value']:.6f}")
            print(f"  Effect size (Cohen's d): {result['test_statistics']['cohens_d']:.4f}")
        
        return self.results
    
    def create_visualization_plots(self, figsize: Tuple[int, int] = (15, 12)):
        """
        Create comprehensive visualizations for all hypothesis tests.
        
        Creates box plots and distributions to visualize the differences
        tested in each hypothesis.
        
        Args:
            figsize: Figure size for the plot grid
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('NBA Hypothesis Testing Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Rest Days vs FG% (Hypothesis 1)
        if 'hypothesis_1' in self.results:
            # Check for appropriate columns
            if 'sufficient_rest' in self.player_stats.columns:
                valid_data = self.player_stats[
                    (self.player_stats['fga'] >= 5) &
                    (self.player_stats['sufficient_rest'].notna()) &
                    (self.player_stats['fg_pct'].notna())
                ]
                
                rest_groups = ['Insufficient Rest', 'Sufficient Rest']
                fg_pct_groups = [
                    valid_data[valid_data['sufficient_rest'] == False]['fg_pct'],
                    valid_data[valid_data['sufficient_rest'] == True]['fg_pct']
                ]
            elif 'rest_days' in self.player_stats.columns:
                valid_data = self.player_stats[
                    (self.player_stats['fga'] >= 5) &
                    (self.player_stats['rest_days'].notna()) &
                    (self.player_stats['fg_pct'].notna())
                ]
                
                rest_groups = ['< 2 days', '≥ 2 days']
                fg_pct_groups = [
                    valid_data[valid_data['rest_days'] < 2]['fg_pct'],
                    valid_data[valid_data['rest_days'] >= 2]['fg_pct']
                ]
            else:
                rest_groups = ['No Data', 'No Data']
                fg_pct_groups = [[], []]
            
            if len(fg_pct_groups[0]) > 0 and len(fg_pct_groups[1]) > 0:
                axes[0, 0].boxplot(fg_pct_groups, labels=rest_groups)
                axes[0, 0].set_title('Shooting Efficiency by Rest Days')
                axes[0, 0].set_ylabel('Field Goal Percentage')
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Shooting Efficiency by Rest Days')
        
        # Plot 2: Home vs Away Points (Hypothesis 2)
        if 'hypothesis_2' in self.results:
            valid_data = self.player_stats[
                (self.player_stats['pts'].notna()) &
                (self.player_stats['is_home_game'].notna())
            ]
            
            if len(valid_data) > 0:
                home_away_groups = ['Away', 'Home']
                points_groups = [
                    valid_data[valid_data['is_home_game'] == False]['pts'],
                    valid_data[valid_data['is_home_game'] == True]['pts']
                ]
                
                axes[0, 1].boxplot(points_groups, labels=home_away_groups)
                axes[0, 1].set_title('Points Scored: Home vs Away')
                axes[0, 1].set_ylabel('Points Scored')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Points Scored: Home vs Away')
        
        # Plot 3: 3PA Evolution (Hypothesis 3)
        if 'hypothesis_3' in self.results:
            # Find 3PA per 36 column
            fg3a_per_36_col = None
            for col in ['fg3a_per_36min', 'fg3a_per_36', 'fg3a_per_36_minutes']:
                if col in self.player_stats.columns:
                    fg3a_per_36_col = col
                    break
            
            if fg3a_per_36_col:
                # Find minutes column for filtering
                minutes_col = None
                for col in ['minutes_played', 'minutes_numeric']:
                    if col in self.player_stats.columns:
                        minutes_col = col
                        break
                
                if minutes_col:
                    valid_data = self.player_stats[
                        (self.player_stats['game_season'].isin([2022, 2024])) &
                        (self.player_stats[minutes_col] >= 20) &
                        (self.player_stats[fg3a_per_36_col].notna())
                    ]
                else:
                    valid_data = self.player_stats[
                        (self.player_stats['game_season'].isin([2022, 2024])) &
                        (self.player_stats[fg3a_per_36_col].notna())
                    ]
                
                if len(valid_data) > 0:
                    season_groups = ['2021-22', '2023-24']
                    fg3a_groups = [
                        valid_data[valid_data['game_season'] == 2022][fg3a_per_36_col],
                        valid_data[valid_data['game_season'] == 2024][fg3a_per_36_col]
                    ]
                    
                    if len(fg3a_groups[0]) > 0 and len(fg3a_groups[1]) > 0:
                        axes[0, 2].boxplot(fg3a_groups, labels=season_groups)
                        axes[0, 2].set_title('3PA per 36 Minutes: Season Comparison')
                        axes[0, 2].set_ylabel('3-Point Attempts per 36 min')
                        axes[0, 2].grid(True, alpha=0.3)
                    else:
                        axes[0, 2].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=axes[0, 2].transAxes)
                        axes[0, 2].set_title('3PA per 36 Minutes: Season Comparison')
                else:
                    axes[0, 2].text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 2].set_title('3PA per 36 Minutes: Season Comparison')
            else:
                axes[0, 2].text(0.5, 0.5, 'No 3PA per 36 Column', ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('3PA per 36 Minutes: Season Comparison')
        
        # Additional distribution plots
        # Plot 4: Rest Days Distribution
        if 'rest_days' in self.player_stats.columns and self.player_stats['rest_days'].notna().any():
            rest_data = self.player_stats['rest_days'].dropna()
            rest_data_capped = np.minimum(rest_data, 10)  # Cap at 10 for visualization
            axes[1, 0].hist(rest_data_capped, bins=range(0, 12), alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Distribution of Rest Days')
            axes[1, 0].set_xlabel('Rest Days (capped at 10)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Rest Days Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Distribution of Rest Days')
        
        # Plot 5: Home/Away Game Distribution
        if 'is_home_game' in self.player_stats.columns and self.player_stats['is_home_game'].notna().any():
            home_counts = self.player_stats['is_home_game'].value_counts()
            if len(home_counts) >= 2:
                axes[1, 1].pie(home_counts.values, labels=['Away', 'Home'], autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('Home vs Away Games Distribution')
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Home vs Away Games Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Home/Away Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Home vs Away Games Distribution')
        
        # Plot 6: Points Distribution by Season
        if 'game_season' in self.player_stats.columns:
            seasons_available = self.player_stats['game_season'].unique()
            if len(seasons_available) > 1:
                colors = plt.cm.tab10(np.linspace(0, 1, len(seasons_available)))
                for i, season in enumerate(sorted(seasons_available)[:4]):  # Show up to 4 seasons
                    season_points = self.player_stats[self.player_stats['game_season'] == season]['pts'].dropna()
                    if len(season_points) > 0:
                        axes[1, 2].hist(season_points, bins=30, alpha=0.5, 
                                      label=f'{season-1}-{str(season)[2:]}', 
                                      density=True, color=colors[i])
                
                axes[1, 2].set_title('Points Distribution by Season')
                axes[1, 2].set_xlabel('Points Scored')
                axes[1, 2].set_ylabel('Density')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'Single Season Only', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Points Distribution by Season')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Season Data', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Points Distribution by Season')
        
        plt.tight_layout()
        plt.savefig("../outputs/visuals/reporting_results/hypothesis_testing_results.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.show()


# Convenience function for easy execution
def run_nba_hypothesis_tests(player_stats_df: pd.DataFrame, games_df: pd.DataFrame = None):
    """
    Convenience function to run all NBA hypothesis tests.
    
    Args:
        player_stats_df: DataFrame with player game statistics
        games_df: Optional DataFrame with game details
        
    Returns:
        Tuple of (results dictionary, tester object)
    """
    # Initialize the hypothesis tester
    tester = NBAHypothesisTester(player_stats_df, games_df)
    
    # Execute all tests
    results = tester.run_all_tests()
    
    # Generate visualizations
    tester.create_visualization_plots()
    
    return results, tester


# Function for feature engineered data integration
def run_hypothesis_tests_with_feature_engineering(raw_data_df: pd.DataFrame, feature_engineering_func=None) -> Tuple[Dict, Any]:
    """
    Run hypothesis tests on feature engineered data.
    
    Args:
        raw_data_df: Raw NBA player stats DataFrame
        feature_engineering_func: Function to apply feature engineering (optional)
        
    Returns:
        Tuple of (results, tester) objects
    """
    print("RUNNING HYPOTHESIS TESTS WITH FEATURE ENGINEERING")
    print("=" * 60)
    
    # Apply feature engineering if provided
    if feature_engineering_func:
        print("Applying feature engineering...")
        engineered_data = feature_engineering_func(raw_data_df)
        print(f"Feature engineering complete. Columns: {len(raw_data_df.columns)} -> {len(engineered_data.columns)}")
    else:
        print("No feature engineering function provided, using data as-is")
        engineered_data = raw_data_df
    
    # Run hypothesis tests
    results, tester = run_nba_hypothesis_tests(engineered_data)
    
    return results, tester


# Data validation utilities
def validate_data_for_testing(player_stats_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate that the DataFrame has the required columns for hypothesis testing.
    
    Args:
        player_stats_df: DataFrame to validate
        
    Returns:
        Dictionary indicating which tests can be run
    """
    # Define required columns for each test (both raw and engineered versions)
    required_columns = {
        'hypothesis_1': {
            'raw': ['player_id', 'game_date', 'fga', 'fg_pct'],
            'engineered': ['rest_days', 'sufficient_rest', 'fga', 'fg_pct']
        },
        'hypothesis_2': {
            'raw': ['pts', 'team_id', 'game_home_team_id'],
            'engineered': ['pts', 'is_home_game']
        },
        'hypothesis_3': {
            'raw': ['game_season', 'fg3a', 'min'],
            'engineered': ['game_season', 'fg3a_per_36min', 'minutes_played']
        }
    }
    
    validation_results = {}
    
    for test_name, col_sets in required_columns.items():
        # Check if either engineered or raw columns exist
        engineered_missing = [col for col in col_sets['engineered'] if col not in player_stats_df.columns]
        raw_missing = [col for col in col_sets['raw'] if col not in player_stats_df.columns]
        
        if len(engineered_missing) == 0:
            validation_results[test_name] = True
            print(f"READY {test_name}: Ready (feature engineered)")
        elif len(raw_missing) == 0:
            validation_results[test_name] = True
            print(f"READY {test_name}: Ready (can create from raw data)")
        else:
            validation_results[test_name] = False
            print(f"MISSING {test_name}: Missing columns. Raw missing: {raw_missing}, Engineered missing: {engineered_missing}")
    
    return validation_results


def generate_hypothesis_report(results: Dict, output_file: str = None) -> str:
    """
    Generate a formatted report of hypothesis test results.
    
    Args:
        results: Dictionary of test results from NBAHypothesisTester
        output_file: Optional file path to save the report
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("NBA PLAYER PERFORMANCE HYPOTHESIS TESTING REPORT")
    report.append("=" * 60)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    for i, (key, result) in enumerate(results.items(), 1):
        report.append(f"\nHYPOTHESIS {i}: {result['hypothesis'].upper()}")
        report.append("-" * 50)
        
        # Sample size information
        sample_info = result['sample_sizes']
        report.append(f"Sample Sizes:")
        for group, size in sample_info.items():
            report.append(f"  {group.replace('_', ' ').title()}: {size:,} observations")
        
        # Descriptive statistics
        desc_stats = result['descriptive_stats']
        report.append(f"\nDescriptive Statistics:")
        for stat, value in desc_stats.items():
            if isinstance(value, float):
                report.append(f"  {stat.replace('_', ' ').title()}: {value:.4f}")
            else:
                report.append(f"  {stat.replace('_', ' ').title()}: {value}")
        
        # Test results
        test_stats = result['test_statistics']
        report.append(f"\nTest Statistics:")
        if 't_p_value_one_tailed' in test_stats:
            report.append(f"  T-statistic: {test_stats['t_statistic']:.4f}")
            report.append(f"  P-value (one-tailed): {test_stats['t_p_value_one_tailed']:.6f}")
        else:
            report.append(f"  T-statistic: {test_stats['t_statistic']:.4f}")
            report.append(f"  P-value (two-tailed): {test_stats['t_p_value']:.6f}")
        
        report.append(f"  Effect Size (Cohen's d): {test_stats['cohens_d']:.4f}")
        
        # Interpretation
        report.append(f"\nConclusion (α = {result['alpha']}):")
        report.append(f"  {result['conclusion']}")
        
        # Effect size interpretation
        d = abs(test_stats['cohens_d'])
        if d < 0.2:
            effect_size = "negligible"
        elif d < 0.5:
            effect_size = "small"
        elif d < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        report.append(f"  Effect size interpretation: {effect_size}")
        
        # NBA Value and Practical Significance
        report.append(f"\nNBA Value & Practical Significance:")
        
        if key == 'hypothesis_1':
            diff = desc_stats['difference']
            if result['significant']:
                report.append(f"  Rest impacts shooting efficiency by {diff:.1%} on average.")
                report.append(f"  This suggests load management strategies may have measurable benefits.")
            else:
                report.append(f"  No significant rest effect found, suggesting other factors may")
                report.append(f"  be more important for shooting performance than rest alone.")
        
        elif key == 'hypothesis_2':
            diff = desc_stats['difference']
            if result['significant']:
                report.append(f"  Home court advantage results in {diff:.2f} additional points per game.")
                report.append(f"  This quantifies the value of playing at home for individual players.")
            else:
                report.append(f"  No significant home court advantage found for individual scoring.")
                report.append(f"  Team-level advantages may not translate to individual statistics.")
        
        elif key == 'hypothesis_3':
            diff = desc_stats['difference']
            if result['significant']:
                report.append(f"  3-point attempts increased by {diff:.2f} per 36 minutes between seasons.")
                report.append(f"  This confirms the continued evolution toward perimeter-oriented offense.")
            else:
                report.append(f"  No significant increase in 3-point attempt rates found.")
                report.append(f"  The evolution may have plateaued or be more position-specific.")
        
        report.append("")
    
    # Overall summary
    report.append("\nOVERALL SUMMARY")
    report.append("-" * 30)
    
    significant_tests = sum(1 for result in results.values() if result['significant'])
    total_tests = len(results)
    
    report.append(f"Tests conducted: {total_tests}")
    report.append(f"Significant results: {significant_tests}")
    report.append(f"Success rate: {significant_tests/total_tests:.1%}")
    
    if significant_tests > 0:
        report.append(f"\nKey findings from significant results:")
        for key, result in results.items():
            if result['significant']:
                hypothesis_num = list(results.keys()).index(key) + 1
                report.append(f"  Hypothesis {hypothesis_num}: {result['hypothesis']}")
    
    report.append(f"\nMethodological Notes:")
    report.append(f"\tAll tests used α = {list(results.values())[0]['alpha']} significance level")
    report.append(f"\tBoth parametric (t-test) and non-parametric (Mann-Whitney U) tests conducted")
    report.append(f"\tEffect sizes calculated using Cohen's d for practical significance assessment")
    report.append(f"\tAssumptions testing included normality and equal variance checks")
    
    report_text = "\n".join(report)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")
    
    return report_text


# Example data loading and testing function
def load_and_test_nba_data():
    """
    Example function to load NBA data and run hypothesis tests.
    
    This function demonstrates the typical workflow for loading data
    and executing the hypothesis testing pipeline.
    """
    try:
        # Load player stats data
        player_stats = pd.read_parquet('data/processed/cleaned_player_stats.parquet')
        
        # Optionally load games data for additional context
        try:
            games_data = pd.read_parquet('data/raw/games_data_seasons_2021_2022_2023_2024_sdk.parquet')
        except:
            games_data = None
            print("Games data not found, proceeding with player stats only")
        
        print(f"Loaded player stats: {player_stats.shape}")
        if games_data is not None:
            print(f"Loaded games data: {games_data.shape}")
        
        # Validate data
        validation = validate_data_for_testing(player_stats)
        print(f"\nData validation results: {validation}")
        
        # Run hypothesis tests
        print("\nRunning hypothesis tests...")
        results, tester = run_nba_hypothesis_tests(player_stats, games_data)
        
        # Generate report
        report = generate_hypothesis_report(results, 'nba_hypothesis_test_report.txt')
        print("\n" + "="*60)
        print("HYPOTHESIS TESTING COMPLETE")
        print("="*60)
        print(report)
        
        return results, tester
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure your data files are in the correct location.")
        print("Expected files:")
        print("  - data/raw/player_game_stats_seasons_2021_2022_2023_2024.parquet")
        print("  - data/raw/games_data_seasons_2021_2022_2023_2024_sdk.parquet (optional)")
        return None, None


# Main execution
if __name__ == "__main__":
    # Example usage - uncomment to run
    # load_and_test_nba_data()
    pass