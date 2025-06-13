"""
NBA Player Performance EDA Module

A comprehensive exploratory data analysis module for NBA player performance prediction
following 2025 data science best practices.

Author: Christopher Bratkovics
Created: 2025
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class EDAConfig:
    """Configuration class for EDA parameters and settings."""
    
    # Plot styling
    plot_style: str = 'seaborn-v0_8-whitegrid'
    color_palette: str = 'husl'
    figure_dpi: int = 300
    save_format: str = 'png'
    
    # Analysis parameters
    target_variables: List[str] = None
    correlation_threshold: float = 0.8
    missing_data_threshold: float = 5.0  # Percentage
    outlier_method: str = 'iqr'  # 'iqr' or 'zscore'
    outlier_threshold: float = 3.0
    
    # File paths
    viz_dir: str = "visuals/EDA"
    report_dir: str = "reports"
    
    def __post_init__(self):
        if self.target_variables is None:
            self.target_variables = ['pts', 'reb', 'ast']


class BaseEDAAnalyzer(ABC):
    """Abstract base class for EDA analyzers."""
    
    def __init__(self, config: EDAConfig):
        self.config = config
        self.results = {}
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform the analysis and return results."""
        pass
    
    def _setup_plotting(self):
        """Setup consistent plotting style."""
        plt.style.use(self.config.plot_style)
        sns.set_palette(self.config.color_palette)


class DataQualityAnalyzer(BaseEDAAnalyzer):
    """Analyzer for data quality assessment including missing values and data types."""
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing data quality metrics
        """
        logger.info("Performing data quality analysis...")
        
        results = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_data': self._analyze_missing_data(df),
            'data_types': self._analyze_data_types(df),
            'duplicates': self._analyze_duplicates(df),
            'recommendations': []
        }
        
        # Generate recommendations
        results['recommendations'] = self._generate_quality_recommendations(results, df)
        
        return results
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_pct.to_dict(),
            'critical_missing': missing_pct[missing_pct > self.config.missing_data_threshold].to_dict(),
            'completely_missing': missing_pct[missing_pct == 100].to_dict(),
            'total_missing_cells': missing_counts.sum()
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types distribution."""
        dtype_counts = df.dtypes.value_counts()
        
        return {
            'dtype_distribution': dtype_counts.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate records."""
        total_duplicates = df.duplicated().sum()
        
        # Check for player-game duplicates if columns exist
        player_game_duplicates = 0
        if all(col in df.columns for col in ['player_id', 'game_id']):
            player_game_duplicates = df.duplicated(subset=['player_id', 'game_id']).sum()
        
        return {
            'total_duplicates': int(total_duplicates),
            'duplicate_percentage': float(total_duplicates / len(df) * 100),
            'player_game_duplicates': int(player_game_duplicates)
        }
    
    def _generate_quality_recommendations(self, results: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on data quality analysis."""
        recommendations = []
        
        # Missing data recommendations
        critical_missing = results['missing_data']['critical_missing']
        if critical_missing:
            recommendations.append(f"ADDRESS: {len(critical_missing)} columns have >{self.config.missing_data_threshold}% missing data")
        
        # Duplicate recommendations
        if results['duplicates']['total_duplicates'] > 0:
            recommendations.append(f"CLEAN: Remove {results['duplicates']['total_duplicates']} duplicate records")
        
        # Memory optimization
        if results['memory_usage_mb'] > 100:
            recommendations.append("OPTIMIZE: Consider data type optimization for large dataset")
        
        return recommendations


class TargetVariableAnalyzer(BaseEDAAnalyzer):
    """Analyzer for target variables (points, rebounds, assists)."""
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze target variable distributions and statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing target variable analysis
        """
        logger.info("Analyzing target variables...")
        
        results = {
            'available_targets': [],
            'target_statistics': {},
            'distribution_analysis': {},
            'position_analysis': {}
        }
        
        for target in self.config.target_variables:
            if target in df.columns:
                results['available_targets'].append(target)
                results['target_statistics'][target] = self._calculate_target_stats(df[target])
                results['distribution_analysis'][target] = self._analyze_distribution(df[target])
                
                # Position analysis if position column exists
                if 'player_position' in df.columns:
                    results['position_analysis'][target] = self._analyze_by_position(df, target)
        
        return results
    
    def _calculate_target_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive statistics for a target variable."""
        return {
            'count': int(series.count()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75)),
            'skewness': float(stats.skew(series.dropna())),
            'kurtosis': float(stats.kurtosis(series.dropna())),
            'zero_games_pct': float((series == 0).sum() / len(series) * 100)
        }
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution characteristics."""
        # Normality test
        if len(series.dropna()) > 3:
            _, normality_p = stats.normaltest(series.dropna()) if len(series.dropna()) > 8 else (np.nan, np.nan)
        else:
            normality_p = np.nan
        
        return {
            'is_normal': bool(normality_p > 0.05) if not np.isnan(normality_p) else None,
            'normality_p_value': float(normality_p) if not np.isnan(normality_p) else None,
            'distribution_type': self._classify_distribution(series)
        }
    
    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify the distribution type based on skewness."""
        skew = stats.skew(series.dropna())
        if abs(skew) < 0.5:
            return 'approximately_normal'
        elif skew > 0.5:
            return 'right_skewed'
        else:
            return 'left_skewed'
    
    def _analyze_by_position(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Analyze target variable by player position."""
        position_stats = df.groupby('player_position')[target].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        
        return {
            'position_statistics': position_stats.to_dict(),
            'position_order': position_stats.sort_values('median', ascending=False).index.tolist()
        }


class CorrelationAnalyzer(BaseEDAAnalyzer):
    """Analyzer for correlation patterns and multicollinearity detection."""
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform correlation analysis for modeling preparation.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing correlation analysis results
        """
        logger.info("Performing correlation analysis...")
        
        # Select numeric columns, excluding ID columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = [col for col in numeric_cols if 'id' in col.lower()]
        model_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(model_cols) <= 1:
            logger.warning("Insufficient numeric columns for correlation analysis")
            return {'error': 'Insufficient numeric columns'}
        
        # Calculate correlation matrix
        corr_matrix = df[model_cols].corr()
        
        # Find high correlation pairs
        high_corr_pairs = self._find_high_correlations(corr_matrix)
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'multicollinearity_risk': len(high_corr_pairs) > 0,
            'features_analyzed': model_cols,
            'correlation_summary': self._summarize_correlations(corr_matrix)
        }
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find pairs of variables with high correlation."""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > self.config.correlation_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        return high_corr_pairs
    
    def _summarize_correlations(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Summarize correlation patterns."""
        # Flatten correlation matrix (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        correlations = corr_matrix.mask(mask).stack().dropna()
        
        return {
            'mean_correlation': float(correlations.mean()),
            'max_correlation': float(correlations.max()),
            'min_correlation': float(correlations.min()),
            'high_correlations_count': int((abs(correlations) > self.config.correlation_threshold).sum())
        }


class OutlierAnalyzer(BaseEDAAnalyzer):
    """Analyzer for outlier detection and analysis."""
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and analyze outliers in target variables.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict containing outlier analysis results
        """
        logger.info("Detecting outliers...")
        
        results = {
            'outlier_summary': {},
            'outlier_flags': {},
            'method_used': self.config.outlier_method
        }
        
        for target in self.config.target_variables:
            if target in df.columns:
                outliers = self._detect_outliers(df[target])
                results['outlier_summary'][target] = {
                    'count': int(outliers.sum()),
                    'percentage': float(outliers.sum() / len(df) * 100)
                }
                results['outlier_flags'][f'{target}_outlier'] = outliers
        
        return results
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using specified method."""
        if self.config.outlier_method == 'iqr':
            return self._iqr_outliers(series)
        elif self.config.outlier_method == 'zscore':
            return self._zscore_outliers(series)
        else:
            raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")
    
    def _iqr_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config.outlier_threshold * IQR
        upper_bound = Q3 + self.config.outlier_threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _zscore_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series.dropna()))
        return pd.Series([False] * len(series), index=series.index).where(
            series.isna(), z_scores > self.config.outlier_threshold
        )


class EDAVisualizer:
    """Class for creating EDA visualizations."""
    
    def __init__(self, config: EDAConfig):
        self.config = config
        self._setup_plotting()
    
    def _setup_plotting(self):
        """Setup consistent plotting style."""
        plt.style.use(self.config.plot_style)
        sns.set_palette(self.config.color_palette)
    
    def create_target_distribution_plots(self, df: pd.DataFrame, save: bool = False) -> plt.Figure:
        """Create comprehensive target variable distribution plots."""
        n_targets = len([t for t in self.config.target_variables if t in df.columns])
        
        if n_targets == 0:
            logger.warning("No target variables found in DataFrame")
            return None
        
        fig, axes = plt.subplots(2, n_targets, figsize=(6*n_targets, 12))
        if n_targets == 1:
            axes = axes.reshape(-1, 1)
        
        for i, target in enumerate(self.config.target_variables):
            if target in df.columns:
                # Distribution plot
                sns.histplot(data=df, x=target, kde=True, ax=axes[0, i])
                axes[0, i].set_title(f'{target.upper()} Distribution')
                axes[0, i].axvline(df[target].mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
                axes[0, i].axvline(df[target].median(), color='darkred', linestyle='-', alpha=0.8, label='Median')
                axes[0, i].legend()
                
                # Box plot by position (if available)
                if 'player_position' in df.columns:
                    sns.boxplot(data=df, x='player_position', y=target, ax=axes[1, i])
                    axes[1, i].set_title(f'{target.upper()} by Position')
                    axes[1, i].tick_params(axis='x', rotation=45)
                else:
                    axes[1, i].text(0.5, 0.5, 'Position data not available', 
                                   ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[1, i].set_title(f'{target.upper()} by Position')
        
        plt.tight_layout()
        
        if save:
            self._save_plot(fig, 'target_distributions')
        
        return fig
    
    def create_correlation_heatmap(self, correlation_results: Dict[str, Any], save: bool = False) -> plt.Figure:
        """Create correlation matrix heatmap."""
        if 'error' in correlation_results:
            logger.warning("Cannot create correlation heatmap: insufficient data")
            return None
        
        corr_matrix = correlation_results['correlation_matrix']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', linewidths=0.5, ax=ax)
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save:
            self._save_plot(fig, 'correlation_matrix')
        
        return fig
    
    def create_missing_data_plot(self, quality_results: Dict[str, Any], save: bool = False) -> plt.Figure:
        """Create missing data visualization."""
        missing_data = quality_results['missing_data']
        missing_pct = pd.Series(missing_data['missing_percentages'])
        
        # Only plot columns with missing data
        missing_cols = missing_pct[missing_pct > 0]
        
        if len(missing_cols) == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Missing Data Found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title('Missing Data Analysis')
        else:
            fig, ax = plt.subplots(figsize=(10, max(6, len(missing_cols) * 0.4)))
            missing_cols.sort_values(ascending=True).plot(kind='barh', ax=ax, color='coral')
            ax.set_title('Missing Data Percentage by Column')
            ax.set_xlabel('Missing Data Percentage (%)')
            
            # Add threshold line
            ax.axvline(self.config.missing_data_threshold, color='red', 
                      linestyle='--', alpha=0.7, label=f'Threshold ({self.config.missing_data_threshold}%)')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            self._save_plot(fig, 'missing_data')
        
        return fig
    
    def create_outlier_plots(self, df: pd.DataFrame, outlier_results: Dict[str, Any], save: bool = False) -> plt.Figure:
        """Create outlier visualization plots."""
        n_targets = len([t for t in self.config.target_variables if t in df.columns])
        
        if n_targets == 0:
            logger.warning("No target variables found for outlier plotting")
            return None
        
        fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 6))
        if n_targets == 1:
            axes = [axes]
        
        for i, target in enumerate(self.config.target_variables):
            if target in df.columns:
                outlier_flag = f'{target}_outlier'
                if outlier_flag in outlier_results['outlier_flags']:
                    outliers = outlier_results['outlier_flags'][outlier_flag]
                    
                    # Box plot with outliers highlighted
                    bp = axes[i].boxplot(df[target], patch_artist=True)
                    axes[i].set_title(f'{target.upper()} - Outlier Detection\n'
                                    f'{outliers.sum()} outliers ({outliers.sum()/len(df)*100:.1f}%)')
                    axes[i].set_ylabel(target.capitalize())
                    
                    # Color the box
                    bp['boxes'][0].set_facecolor('lightblue')
                    bp['boxes'][0].set_alpha(0.7)
        
        plt.tight_layout()
        
        if save:
            self._save_plot(fig, 'outlier_analysis')
        
        return fig
    
    def _save_plot(self, fig: plt.Figure, name: str):
        """Save plot to file."""
        os.makedirs(self.config.viz_dir, exist_ok=True)
        filepath = os.path.join(self.config.viz_dir, f'{name}.{self.config.save_format}')
        fig.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        logger.info(f"Saved plot: {filepath}")


class NBAEDAOrchestrator:
    """
    Main orchestrator class for NBA EDA following 2025 best practices.
    
    This class coordinates multiple analyzers and provides a clean interface
    for comprehensive exploratory data analysis.
    """
    
    def __init__(self, config: Optional[EDAConfig] = None):
        """
        Initialize the EDA orchestrator.
        
        Args:
            config: EDA configuration (uses default if None)
        """
        self.config = config or EDAConfig()
        self.results = {}
        
        # Initialize analyzers
        self.analyzers = {
            'data_quality': DataQualityAnalyzer(self.config),
            'target_variables': TargetVariableAnalyzer(self.config),
            'correlation': CorrelationAnalyzer(self.config),
            'outliers': OutlierAnalyzer(self.config)
        }
        
        # Initialize visualizer
        self.visualizer = EDAVisualizer(self.config)
        
        # Setup directories
        os.makedirs(self.config.viz_dir, exist_ok=True)
        os.makedirs(self.config.report_dir, exist_ok=True)
    
    def run_complete_eda(self, df: pd.DataFrame, 
                        df_comparison: Optional[pd.DataFrame] = None,
                        save_plots: bool = False,
                        save_report: bool = False) -> Dict[str, Any]:
        """
        Run complete EDA analysis on NBA player performance data.
        
        Args:
            df: Primary DataFrame to analyze
            df_comparison: Optional comparison DataFrame (e.g., cleaned vs raw)
            save_plots: Whether to save visualization plots
            save_report: Whether to save analysis report
            
        Returns:
            Dict containing all analysis results
        """
        logger.info("Starting comprehensive NBA EDA analysis...")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Run all analyzers
        for name, analyzer in self.analyzers.items():
            try:
                logger.info(f"Running {name} analysis...")
                self.results[name] = analyzer.analyze(df)
            except Exception as e:
                logger.error(f"Error in {name} analysis: {e}")
                self.results[name] = {'error': str(e)}
        
        # Create visualizations
        if save_plots or not save_plots:  # Always create plots, optionally save
            self._create_all_visualizations(df, save_plots)
        
        # Generate comprehensive insights
        self.results['insights'] = self._generate_insights()
        
        # Generate recommendations
        self.results['recommendations'] = self._generate_recommendations()
        
        # Save report if requested
        if save_report:
            self._save_analysis_report()
        
        logger.info("EDA analysis complete!")
        return self.results
    
    def _create_all_visualizations(self, df: pd.DataFrame, save: bool):
        """Create all EDA visualizations."""
        logger.info("Creating visualizations...")
        
        try:
            # Target variable distributions
            self.visualizer.create_target_distribution_plots(df, save=save)
            
            # Correlation heatmap
            if 'correlation' in self.results and 'error' not in self.results['correlation']:
                self.visualizer.create_correlation_heatmap(self.results['correlation'], save=save)
            
            # Missing data plot
            if 'data_quality' in self.results:
                self.visualizer.create_missing_data_plot(self.results['data_quality'], save=save)
            
            # Outlier plots
            if 'outliers' in self.results:
                self.visualizer.create_outlier_plots(df, self.results['outliers'], save=save)
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate key insights from all analyses."""
        insights = {
            'data_overview': {},
            'target_insights': {},
            'quality_insights': {},
            'modeling_insights': {}
        }
        
        # Data overview insights
        if 'data_quality' in self.results:
            dq = self.results['data_quality']
            insights['data_overview'] = {
                'total_records': dq['shape'][0],
                'total_features': dq['shape'][1],
                'memory_usage_mb': round(dq['memory_usage_mb'], 2),
                'missing_data_quality': 'Good' if len(dq['missing_data']['critical_missing']) == 0 else 'Needs attention'
            }
        
        # Target variable insights
        if 'target_variables' in self.results:
            tv = self.results['target_variables']
            insights['target_insights'] = {
                'available_targets': tv['available_targets'],
                'distribution_summary': {
                    target: tv['target_statistics'][target]['mean'] 
                    for target in tv['available_targets']
                }
            }
        
        # Correlation insights
        if 'correlation' in self.results and 'error' not in self.results['correlation']:
            corr = self.results['correlation']
            insights['modeling_insights']['multicollinearity_risk'] = corr['multicollinearity_risk']
            insights['modeling_insights']['high_correlations'] = len(corr['high_correlation_pairs'])
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on all analyses."""
        recommendations = []
        
        # Data quality recommendations
        if 'data_quality' in self.results:
            recommendations.extend(self.results['data_quality'].get('recommendations', []))
        
        # Feature engineering recommendations
        if 'target_variables' in self.results:
            tv = self.results['target_variables']
            for target in tv['available_targets']:
                stats = tv['target_statistics'][target]
                if stats['zero_games_pct'] > 10:
                    recommendations.append(f"CONSIDER: {target} has {stats['zero_games_pct']:.1f}% zero values - may need special handling")
        
        # Modeling recommendations
        if 'correlation' in self.results and 'error' not in self.results['correlation']:
            if self.results['correlation']['multicollinearity_risk']:
                recommendations.append("MODELING: Address multicollinearity before training models")
        
        # Outlier recommendations
        if 'outliers' in self.results:
            outlier_summary = self.results['outliers']['outlier_summary']
            for target, stats in outlier_summary.items():
                if stats['percentage'] > 5:
                    recommendations.append(f"OUTLIERS: {target} has {stats['percentage']:.1f}% outliers - consider robust methods")
        
        return recommendations
    
    def _save_analysis_report(self):
        """Save comprehensive analysis report."""
        report_path = os.path.join(self.config.report_dir, 'eda_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("NBA PLAYER PERFORMANCE EDA REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Data overview
            if 'insights' in self.results:
                insights = self.results['insights']
                f.write("DATA OVERVIEW\n")
                f.write("-" * 20 + "\n")
                for key, value in insights['data_overview'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Recommendations
            if 'recommendations' in self.results:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                for i, rec in enumerate(self.results['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
        
        logger.info(f"Analysis report saved: {report_path}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a concise summary of analysis results."""
        if not self.results:
            return {'error': 'No analysis has been run yet'}
        
        summary = {}
        
        if 'insights' in self.results:
            summary.update(self.results['insights'])
        
        if 'recommendations' in self.results:
            summary['total_recommendations'] = len(self.results['recommendations'])
        
        return summary


# Convenience functions for easy usage
def quick_eda(df: pd.DataFrame, 
              target_variables: Optional[List[str]] = None,
              save_plots: bool = False) -> Dict[str, Any]:
    """
    Convenience function for quick EDA analysis.
    
    Args:
        df: DataFrame to analyze
        target_variables: List of target variables (defaults to ['pts', 'reb', 'ast'])
        save_plots: Whether to save visualization plots
        
    Returns:
        Dict containing analysis results
    """
    config = EDAConfig()
    if target_variables:
        config.target_variables = target_variables
    
    orchestrator = NBAEDAOrchestrator(config)
    return orchestrator.run_complete_eda(df, save_plots=save_plots)


def compare_datasets(df_raw: pd.DataFrame, 
                    df_clean: pd.DataFrame,
                    target_variables: Optional[List[str]] = None,
                    save_plots: bool = False) -> Dict[str, Any]:
    """
    Compare two datasets (e.g., raw vs cleaned) with comprehensive EDA.
    
    Args:
        df_raw: Raw dataset
        df_clean: Cleaned dataset
        target_variables: List of target variables to analyze
        save_plots: Whether to save plots
        
    Returns:
        Dict containing comparison results
    """
    config = EDAConfig()
    if target_variables:
        config.target_variables = target_variables
    
    orchestrator = NBAEDAOrchestrator(config)
    
    # Analyze both datasets
    raw_results = orchestrator.run_complete_eda(df_raw, save_plots=False)
    clean_results = orchestrator.run_complete_eda(df_clean, save_plots=save_plots)
    
    # Create comparison summary
    comparison = {
        'raw_dataset': raw_results,
        'clean_dataset': clean_results,
        'comparison_summary': _create_comparison_summary(raw_results, clean_results)
    }
    
    return comparison


def _create_comparison_summary(raw_results: Dict[str, Any], 
                              clean_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary comparing raw and clean datasets."""
    summary = {}
    
    # Shape comparison
    if 'data_quality' in raw_results and 'data_quality' in clean_results:
        raw_shape = raw_results['data_quality']['shape']
        clean_shape = clean_results['data_quality']['shape']
        
        summary['shape_comparison'] = {
            'raw_shape': raw_shape,
            'clean_shape': clean_shape,
            'rows_removed': raw_shape[0] - clean_shape[0],
            'columns_added': clean_shape[1] - raw_shape[1]
        }
    
    # Missing data comparison
    if ('data_quality' in raw_results and 'data_quality' in clean_results):
        raw_missing = raw_results['data_quality']['missing_data']['total_missing_cells']
        clean_missing = clean_results['data_quality']['missing_data']['total_missing_cells']
        
        summary['missing_data_improvement'] = {
            'raw_missing_cells': raw_missing,
            'clean_missing_cells': clean_missing,
            'improvement': raw_missing - clean_missing
        }
    
    return summary


def validate_for_hypothesis_testing(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate DataFrame for NBA hypothesis testing requirements.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict indicating which hypothesis tests can be performed
    """
    validation_results = {
        'hypothesis_1_rest_days': all(col in df.columns for col in ['rest_days', 'fg_pct', 'fga']) or
                                 all(col in df.columns for col in ['sufficient_rest', 'fg_pct', 'fga']),
        'hypothesis_2_home_away': all(col in df.columns for col in ['is_home_game', 'pts']) or
                                 all(col in df.columns for col in ['team_id', 'game_home_team_id', 'pts']),
        'hypothesis_3_three_point': all(col in df.columns for col in ['fg3a_per_36min', 'game_season']) or
                                   all(col in df.columns for col in ['fg3a', 'game_season', 'min'])
    }
    
    logger.info("Hypothesis testing validation:")
    for test, valid in validation_results.items():
        status = "✓" if valid else "✗"
        logger.info(f"  {status} {test}: {'Ready' if valid else 'Missing features'}")
    
    return validation_results


def validate_for_modeling(df: pd.DataFrame, 
                         target_variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate DataFrame for machine learning modeling readiness.
    
    Args:
        df: DataFrame to validate
        target_variables: List of target variables to check
        
    Returns:
        Dict containing modeling readiness assessment
    """
    if target_variables is None:
        target_variables = ['pts', 'reb', 'ast']
    
    validation = {
        'targets_available': [t for t in target_variables if t in df.columns],
        'sufficient_samples': len(df) >= 1000,
        'feature_columns': len(df.select_dtypes(include=[np.number]).columns) >= 5,
        'missing_data_acceptable': (df.isnull().sum().sum() / (len(df) * len(df.columns))) < 0.05,
        'ready_for_modeling': True
    }
    
    # Check overall readiness
    validation['ready_for_modeling'] = all([
        len(validation['targets_available']) > 0,
        validation['sufficient_samples'],
        validation['feature_columns'],
        validation['missing_data_acceptable']
    ])
    
    return validation


class EDAReportGenerator:
    """Generate professional EDA reports in multiple formats."""
    
    def __init__(self, results: Dict[str, Any], config: EDAConfig):
        self.results = results
        self.config = config
    
    def generate_markdown_report(self, filepath: str = None) -> str:
        """Generate a comprehensive markdown report."""
        if filepath is None:
            filepath = os.path.join(self.config.report_dir, 'eda_report.md')
        
        markdown_content = self._build_markdown_content()
        
        with open(filepath, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report saved: {filepath}")
        return markdown_content
    
    def _build_markdown_content(self) -> str:
        """Build the markdown content for the report."""
        lines = [
            "# NBA Player Performance EDA Report",
            "",
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Add data overview
        if 'insights' in self.results and 'data_overview' in self.results['insights']:
            overview = self.results['insights']['data_overview']
            lines.extend([
                f"- **Total Records:** {overview.get('total_records', 'N/A'):,}",
                f"- **Total Features:** {overview.get('total_features', 'N/A')}",
                f"- **Memory Usage:** {overview.get('memory_usage_mb', 'N/A')} MB",
                f"- **Data Quality:** {overview.get('missing_data_quality', 'N/A')}",
                ""
            ])
        
        # Add key findings
        lines.extend([
            "## Key Findings",
            ""
        ])
        
        if 'recommendations' in self.results:
            for i, rec in enumerate(self.results['recommendations'][:5], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Add detailed sections
        lines.extend([
            "## Data Quality Analysis",
            "",
            self._format_data_quality_section(),
            "",
            "## Target Variable Analysis", 
            "",
            self._format_target_variables_section(),
            "",
            "## Correlation Analysis",
            "",
            self._format_correlation_section(),
            ""
        ])
        
        return "\n".join(lines)
    
    def _format_data_quality_section(self) -> str:
        """Format the data quality section."""
        if 'data_quality' not in self.results:
            return "Data quality analysis not available."
        
        dq = self.results['data_quality']
        
        sections = [
            f"**Dataset Shape:** {dq['shape'][0]:,} rows × {dq['shape'][1]} columns",
            f"**Memory Usage:** {dq['memory_usage_mb']:.2f} MB",
            ""
        ]
        
        # Missing data summary
        if dq['missing_data']['critical_missing']:
            sections.append("**Critical Missing Data (>{:.1f}%):**".format(self.config.missing_data_threshold))
            for col, pct in dq['missing_data']['critical_missing'].items():
                sections.append(f"- {col}: {pct}%")
        else:
            sections.append("✓ No critical missing data issues found")
        
        return "\n".join(sections)
    
    def _format_target_variables_section(self) -> str:
        """Format the target variables section."""
        if 'target_variables' not in self.results:
            return "Target variable analysis not available."
        
        tv = self.results['target_variables']
        
        sections = [
            f"**Available Targets:** {', '.join(tv['available_targets'])}",
            ""
        ]
        
        # Statistics table
        if tv['target_statistics']:
            sections.append("| Variable | Mean | Median | Std | Min | Max |")
            sections.append("|----------|------|--------|-----|-----|-----|")
            
            for target, stats in tv['target_statistics'].items():
                sections.append(
                    f"| {target.upper()} | {stats['mean']:.2f} | {stats['median']:.2f} | "
                    f"{stats['std']:.2f} | {stats['min']:.0f} | {stats['max']:.0f} |"
                )
        
        return "\n".join(sections)
    
    def _format_correlation_section(self) -> str:
        """Format the correlation analysis section."""
        if 'correlation' not in self.results or 'error' in self.results['correlation']:
            return "Correlation analysis not available."
        
        corr = self.results['correlation']
        
        sections = [
            f"**Features Analyzed:** {len(corr['features_analyzed'])}",
            f"**High Correlations Found:** {len(corr['high_correlation_pairs'])}",
            ""
        ]
        
        if corr['high_correlation_pairs']:
            sections.append("**High Correlation Pairs (>{:.1f}):**".format(self.config.correlation_threshold))
            for var1, var2, corr_val in corr['high_correlation_pairs'][:5]:
                sections.append(f"- {var1} ↔ {var2}: {corr_val:.3f}")
        
        return "\n".join(sections)


# Example usage and integration functions
def demonstrate_enhanced_eda():
    """Demonstrate the enhanced EDA module functionality."""
    logger.info("NBA EDA Enhanced Module Demo")
    
    # This would typically use real data
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'player_id': np.repeat(range(50), 20),
        'game_id': range(1000),
        'pts': np.random.poisson(15, 1000),
        'reb': np.random.poisson(8, 1000),
        'ast': np.random.poisson(5, 1000),
        'fga': np.random.poisson(12, 1000),
        'fg_pct': np.random.beta(2, 3, 1000),
        'player_position': np.random.choice(['G', 'F', 'C'], 1000),
        'game_season': np.random.choice([2022, 2023, 2024], 1000),
        'is_home_game': np.random.choice([True, False], 1000)
    })
    
    # Add some missing values for demonstration
    sample_data.loc[sample_data.sample(50).index, 'fg_pct'] = np.nan
    
    # Run quick EDA
    results = quick_eda(sample_data, save_plots=False)
    
    # Generate report
    config = EDAConfig()
    report_gen = EDAReportGenerator(results, config)
    report_content = report_gen.generate_markdown_report()
    
    # Validation checks
    hypothesis_validation = validate_for_hypothesis_testing(sample_data)
    modeling_validation = validate_for_modeling(sample_data)
    
    logger.info("Demo completed successfully!")
    return {
        'eda_results': results,
        'hypothesis_validation': hypothesis_validation,
        'modeling_validation': modeling_validation
    }


# Factory function for easy instantiation
def create_nba_eda_analyzer(target_variables: Optional[List[str]] = None,
                           correlation_threshold: float = 0.8,
                           missing_threshold: float = 5.0,
                           viz_dir: str = "visuals/EDA") -> NBAEDAOrchestrator:
    """
    Factory function to create a configured NBA EDA analyzer.
    
    Args:
        target_variables: List of target variables to analyze
        correlation_threshold: Threshold for high correlation detection
        missing_threshold: Threshold for critical missing data percentage
        viz_dir: Directory for saving visualizations
        
    Returns:
        Configured NBAEDAOrchestrator instance
    """
    config = EDAConfig()
    
    if target_variables:
        config.target_variables = target_variables
    config.correlation_threshold = correlation_threshold
    config.missing_data_threshold = missing_threshold
    config.viz_dir = viz_dir
    
    return NBAEDAOrchestrator(config)


if __name__ == "__main__":
    # Run demonstration if script is executed directly
    demonstrate_enhanced_eda()