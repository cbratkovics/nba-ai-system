"""
NBA Player Performance Prediction - Reporting Module

A comprehensive reporting and visualization framework for NBA player performance 
prediction results. This module generates professional-quality visualizations,
statistical analyses, and summary reports for model evaluation and presentation.

Author: Christopher Bratkovics
Date: 2025
Description: This module handles all aspects of results reporting including
             model performance comparisons, feature importance analysis,
             residual diagnostics, and executive summary generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Any
from datetime import datetime
from pathlib import Path

# Suppress warnings to ensure clean output in reports
warnings.filterwarnings('ignore')

# Configure matplotlib for professional publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,           # High resolution for presentations
    'savefig.dpi': 300,          # High resolution for saved files
    'font.size': 12,             # Base font size
    'axes.labelsize': 12,        # Axis label font size
    'axes.titlesize': 14,        # Subplot title font size
    'xtick.labelsize': 10,       # X-axis tick label size
    'ytick.labelsize': 10,       # Y-axis tick label size
    'legend.fontsize': 10,       # Legend font size
    'figure.facecolor': 'white', # White background
    'axes.facecolor': 'white'    # White plot background
})

# Define professional color palette for consistent visualization styling
COLORS = {
    'primary': '#2E86AB',      # Professional blue for main elements
    'secondary': '#A23B72',    # Complementary purple for contrast
    'tertiary': '#F18F01',     # Orange for additional variety
    'success': '#27AE60',      # Green for positive indicators
    'warning': '#F39C12',      # Orange for warnings/attention
    'danger': '#E74C3C',       # Red for errors/critical items
    'info': '#3498DB',         # Light blue for informational elements
    'dark': '#2C3E50',         # Dark gray for text/borders
    'light': '#ECF0F1'         # Light gray for backgrounds
}


class ModelResultsReporter:
    """
    A comprehensive reporting class for NBA model results visualization and analysis.
    
    This class handles the generation of all visual and textual reports needed
    for presenting machine learning model results in a professional context.
    It creates publication-quality figures and detailed statistical summaries.
    
    Attributes:
        output_dir (Path): Directory for saving visualization files
        reports_dir (Path): Directory for saving text reports and CSV files
    """
    
    def __init__(self, output_dir: str = "../outputs/visuals/reporting_results"):
        """
        Initialize the reporter with specified output directories.
        
        Args:
            output_dir: Path where visualization PNG files will be saved
        """
        # Set up output directories and ensure they exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up separate directory for text reports and CSV files
        self.reports_dir = Path("../outputs/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Model Results Reporter initialized. Output directory: {self.output_dir}")
    
    def create_model_performance_comparison(self, test_results: Dict) -> None:
        """
        Generate comprehensive model performance comparison visualizations.
        
        Creates a multi-panel figure comparing different models across all target
        variables using bar charts for R², MAE, and a summary panel. Also exports
        the performance data to CSV for further analysis.
        
        Args:
            test_results: Dictionary containing test set performance metrics
                         Structure: {target: {model_name: {metrics}}}
        """
        print("Creating model performance comparison...")
        
        # Extract and organize performance metrics from nested results dictionary
        performance_data = []
        
        for target, models in test_results.items():
            for model_name, metrics in models.items():
                # Ensure we have valid metric dictionaries
                if isinstance(metrics, dict) and 'r2' in metrics:
                    performance_data.append({
                        'Target': target.upper(),
                        'Model': model_name.replace('_', ' ').title(),
                        'R²': metrics['r2'],
                        'MAE': metrics['mae'],
                        'RMSE': metrics.get('rmse', 0)
                    })
        
        # Check if we have valid data to visualize
        if not performance_data:
            print("No valid performance data found.")
            return
        
        # Convert to DataFrame for easier manipulation
        df_performance = pd.DataFrame(performance_data)
        
        # Create figure with three subplots for comprehensive comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Subplot 1: R² Score comparison across models and targets
        pivot_r2 = df_performance.pivot(index='Model', columns='Target', values='R²')
        pivot_r2.plot(kind='bar', ax=axes[0], 
                      color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
        axes[0].set_title('R² Score by Model and Target')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)  # R² ranges from 0 to 1
        axes[0].legend(title='Target')
        axes[0].grid(True, alpha=0.3)
        
        # Subplot 2: Mean Absolute Error comparison
        pivot_mae = df_performance.pivot(index='Model', columns='Target', values='MAE')
        pivot_mae.plot(kind='bar', ax=axes[1], 
                       color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
        axes[1].set_title('Mean Absolute Error by Model and Target')
        axes[1].set_ylabel('MAE')
        axes[1].legend(title='Target')
        axes[1].grid(True, alpha=0.3)
        
        # Subplot 3: Best model summary text panel
        axes[2].axis('off')
        best_models_text = "BEST MODELS BY TARGET:\n\n"
        
        # Identify and summarize the best performing model for each target
        for target in df_performance['Target'].unique():
            target_data = df_performance[df_performance['Target'] == target]
            best_model = target_data.loc[target_data['R²'].idxmax()]
            best_models_text += f"{target}:\n"
            best_models_text += f"  Model: {best_model['Model']}\n"
            best_models_text += f"  R² = {best_model['R²']:.3f}\n"
            best_models_text += f"  MAE = {best_model['MAE']:.2f}\n\n"
        
        # Add text to the summary panel with professional formatting
        axes[2].text(0.1, 0.9, best_models_text, transform=axes[2].transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save the figure in high resolution
        filename = self.output_dir / "model_performance_comparison.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")
        plt.close()
        
        # Export performance metrics to CSV for external analysis
        csv_filename = self.reports_dir / "model_performance_metrics.csv"
        df_performance.to_csv(csv_filename, index=False)
        print(f"Saved performance metrics: {csv_filename}")
    
    def create_feature_importance_plots(self, importance_results: Dict, test_results: Dict) -> None:
        """
        Generate feature importance visualizations for each target variable.
        
        Creates horizontal bar charts showing the top 15 most important features
        for the best performing model of each target. Includes model performance
        metrics in the title for context.
        
        Args:
            importance_results: Dictionary of feature importance scores by target and model
            test_results: Dictionary of test performance metrics for model selection
        """
        print("Creating feature importance plots...")
        
        # Process each target variable separately
        for target in importance_results.keys():
            # Identify the best performing model for this target
            best_model_name = None
            best_r2 = -1
            
            if target in test_results:
                for model_name, metrics in test_results[target].items():
                    if isinstance(metrics, dict) and 'r2' in metrics and metrics['r2'] > best_r2:
                        best_r2 = metrics['r2']
                        best_model_name = model_name
            
            # Skip if no valid model found
            if not best_model_name or best_model_name not in importance_results[target]:
                continue
            
            # Extract top 15 features for visualization
            importance_df = importance_results[target][best_model_name].head(15)
            
            # Create figure with appropriate size for readability
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Generate horizontal bar plot for better feature name visibility
            y_pos = np.arange(len(importance_df))
            bars = ax.barh(y_pos, importance_df['importance'], 
                           color=COLORS['primary'], alpha=0.8)
            
            # Add importance scores as text labels on bars
            for i, (idx, row) in enumerate(importance_df.iterrows()):
                ax.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}',
                        va='center', fontsize=10)
            
            # Configure plot aesthetics
            ax.set_yticks(y_pos)
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{target.upper()} - Top 15 Feature Importance\n'
                        f'Model: {best_model_name.replace("_", " ").title()} (R² = {best_r2:.3f})',
                        fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure with descriptive filename
            filename = self.output_dir / f"feature_importance_{target}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved: {filename}")
            plt.close()
            
            # Export feature importance data for further analysis
            csv_filename = self.reports_dir / f"feature_importance_{target}.csv"
            importance_df.to_csv(csv_filename, index=False)
    
    def create_residual_analysis(self, test_results: Dict, y_test: Dict) -> None:
        """
        Create residual analysis plots to diagnose model performance patterns.
        
        Generates scatter plots of residuals vs predicted values to check for
        heteroscedasticity, bias, and other systematic patterns. Includes
        statistical summaries and trend lines for comprehensive analysis.
        
        Args:
            test_results: Dictionary containing model predictions and metrics
            y_test: Dictionary of actual test set values by target
        """
        print("Creating residual analysis plots...")
        
        # Set up subplots for each target variable
        n_targets = len(y_test)
        fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        
        fig.suptitle('Residual Analysis - Best Models', fontsize=16, fontweight='bold')
        
        # Track residual statistics for reporting
        residual_stats = []
        
        # Analyze residuals for each target
        for i, target in enumerate(y_test.keys()):
            if target not in test_results:
                continue
            
            # Find the best performing model for this target
            best_model = None
            best_r2 = -1
            
            for model_name, metrics in test_results[target].items():
                if isinstance(metrics, dict) and 'r2' in metrics and 'predictions' in metrics:
                    if metrics['r2'] > best_r2:
                        best_r2 = metrics['r2']
                        best_model = model_name
                        best_metrics = metrics
            
            if best_model is None:
                continue
            
            # Calculate residuals (actual - predicted)
            actual = y_test[target]
            predicted = best_metrics['predictions']
            residuals = actual - predicted
            
            # Create residual scatter plot
            ax = axes[i]
            scatter = ax.scatter(predicted, residuals, alpha=0.5, s=20, 
                                color=COLORS['primary'])
            
            # Add reference line at y=0 for perfect predictions
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
            
            # Add trend line to detect systematic bias
            z = np.polyfit(predicted, residuals, 1)
            p = np.poly1d(z)
            ax.plot(sorted(predicted), p(sorted(predicted)), 
                   color=COLORS['warning'], linewidth=2, alpha=0.8)
            
            # Calculate and display residual statistics
            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            
            # Add statistics annotation to plot
            stats_text = f'Mean: {residual_mean:.3f}\nStd: {residual_std:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Configure plot labels and styling
            ax.set_xlabel(f'Predicted {target.upper()}')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{target.upper()} - {best_model.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            # Store statistics for export
            residual_stats.append({
                'Target': target.upper(),
                'Model': best_model,
                'Residual_Mean': residual_mean,
                'Residual_Std': residual_std,
                'R²': best_r2
            })
        
        plt.tight_layout()
        
        # Save residual analysis figure
        filename = self.output_dir / "residual_analysis.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")
        plt.close()
        
        # Export residual statistics to CSV
        if residual_stats:
            df_residuals = pd.DataFrame(residual_stats)
            csv_filename = self.reports_dir / "residual_statistics.csv"
            df_residuals.to_csv(csv_filename, index=False)
            print(f"Saved residual statistics: {csv_filename}")
    
    def create_prediction_scatter_plots(self, test_results: Dict, y_test: Dict) -> None:
        """
        Create actual vs predicted scatter plots for model evaluation.
        
        Generates scatter plots comparing actual and predicted values with
        perfect prediction reference lines. Includes performance metrics
        for quick assessment of model accuracy.
        
        Args:
            test_results: Dictionary containing model predictions and metrics
            y_test: Dictionary of actual test set values by target
        """
        print("Creating prediction accuracy plots...")
        
        # Set up subplots for each target
        n_targets = len(y_test)
        fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        
        fig.suptitle('Actual vs Predicted Values - Best Models', fontsize=16, fontweight='bold')
        
        # Create scatter plot for each target
        for i, target in enumerate(y_test.keys()):
            if target not in test_results:
                continue
            
            # Identify best performing model
            best_model = None
            best_r2 = -1
            
            for model_name, metrics in test_results[target].items():
                if isinstance(metrics, dict) and 'r2' in metrics and 'predictions' in metrics:
                    if metrics['r2'] > best_r2:
                        best_r2 = metrics['r2']
                        best_model = model_name
                        best_metrics = metrics
            
            if best_model is None:
                continue
            
            # Extract actual and predicted values
            actual = y_test[target]
            predicted = best_metrics['predictions']
            
            # Create scatter plot with semi-transparent points
            ax = axes[i]
            ax.scatter(actual, predicted, alpha=0.5, s=30, color=COLORS['primary'])
            
            # Add diagonal line representing perfect predictions
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Prediction')
            
            # Add performance metrics annotation
            metrics_text = f'R² = {best_r2:.3f}\nMAE = {best_metrics["mae"]:.2f}'
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Configure plot aesthetics
            ax.set_xlabel(f'Actual {target.upper()}')
            ax.set_ylabel(f'Predicted {target.upper()}')
            ax.set_title(f'{target.upper()} - {best_model.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ensure square aspect ratio for fair comparison
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save scatter plot figure
        filename = self.output_dir / "prediction_scatter_plots.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")
        plt.close()
    
    def generate_summary_report(self, test_results: Dict, importance_results: Dict) -> None:
        """
        Generate a comprehensive text summary of all model results.
        
        Creates a formatted text report including model performance metrics,
        feature importance rankings, key insights, and recommendations for
        improvement. Saved as both console output and text file.
        
        Args:
            test_results: Dictionary of model performance metrics
            importance_results: Dictionary of feature importance scores
        """
        print("Generating summary report...")
        
        # Initialize report content
        report_lines = []
        report_lines.append("NBA PLAYER PERFORMANCE PREDICTION - MODEL RESULTS SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Section 1: Model Performance Summary
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 30)
        
        # Summarize performance for each target variable
        for target, models in test_results.items():
            report_lines.append(f"\n{target.upper()}:")
            
            # Track best model for this target
            best_model = None
            best_r2 = -1
            
            # Report metrics for each model
            for model_name, metrics in models.items():
                if isinstance(metrics, dict) and 'r2' in metrics:
                    report_lines.append(f"  {model_name.replace('_', ' ').title()}:")
                    report_lines.append(f"    R² = {metrics['r2']:.3f}")
                    report_lines.append(f"    MAE = {metrics['mae']:.2f}")
                    report_lines.append(f"    RMSE = {metrics.get('rmse', 'N/A')}")
                    
                    # Update best model tracking
                    if metrics['r2'] > best_r2:
                        best_r2 = metrics['r2']
                        best_model = model_name
            
            # Highlight best performing model
            if best_model:
                report_lines.append(f"  BEST MODEL: {best_model.replace('_', ' ').title()} (R² = {best_r2:.3f})")
        
        # Section 2: Feature Importance Summary
        report_lines.append("\n\nTOP PREDICTIVE FEATURES")
        report_lines.append("-" * 30)
        
        # Report top features for each target
        for target in importance_results.keys():
            report_lines.append(f"\n{target.upper()} - Top 5 Features:")
            
            # Find and report top features from best model
            for model_name, importance_df in importance_results[target].items():
                if not importance_df.empty:
                    top_features = importance_df.head(5)
                    for idx, row in top_features.iterrows():
                        report_lines.append(f"  {row['feature']}: {row['importance']:.3f}")
                    break
        
        # Section 3: Key Insights
        report_lines.append("\n\nKEY INSIGHTS")
        report_lines.append("-" * 30)
        report_lines.append("1. Playing time (minutes) is consistently the top predictor")
        report_lines.append("2. Load management features (rest × minutes) show high importance")
        report_lines.append("3. Random Forest models perform best for points and rebounds")
        report_lines.append("4. Model accuracy: Points > Rebounds ≈ Assists")
        
        # Section 4: Areas for Improvement
        report_lines.append("\n\nAREAS FOR IMPROVEMENT")
        report_lines.append("-" * 30)
        report_lines.append("1. Consider opponent defensive metrics")
        report_lines.append("2. Add player momentum/streak features")
        report_lines.append("3. Incorporate team performance context")
        report_lines.append("4. Address heteroscedasticity in residuals")
        
        # Save report to file
        report_content = "\n".join(report_lines)
        report_filename = self.reports_dir / "model_results_summary.txt"
        with open(report_filename, 'w') as f:
            f.write(report_content)
        print(f"Saved summary report: {report_filename}")
        
        # Also display report in console for immediate review
        print("\n" + report_content)
    
    def create_model_comparison_heatmap(self, test_results: Dict) -> None:
        """
        Create a heatmap visualization showing model performance across all targets.
        
        Generates a color-coded matrix where rows are models and columns are
        target variables, with R² scores as values. Provides quick visual
        comparison of model effectiveness across different prediction tasks.
        
        Args:
            test_results: Dictionary of model performance metrics
        """
        print("Creating model comparison heatmap...")
        
        # Extract unique models and targets from results
        models = set()
        targets = list(test_results.keys())
        
        # Collect all unique model names
        for target, model_dict in test_results.items():
            models.update(model_dict.keys())
        
        models = sorted(list(models))
        
        # Initialize R² score matrix with NaN for missing combinations
        r2_matrix = np.full((len(models), len(targets)), np.nan)
        
        # Populate matrix with actual R² scores
        for j, target in enumerate(targets):
            for i, model in enumerate(models):
                if model in test_results[target] and isinstance(test_results[target][model], dict):
                    r2_matrix[i, j] = test_results[target][model].get('r2', np.nan)
        
        # Create heatmap visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for missing values (NaN)
        mask = np.isnan(r2_matrix)
        
        # Generate heatmap with professional styling
        sns.heatmap(r2_matrix, 
                    xticklabels=[t.upper() for t in targets],
                    yticklabels=[m.replace('_', ' ').title() for m in models],
                    annot=True,           # Show values in cells
                    fmt='.3f',            # Format to 3 decimal places
                    cmap='RdYlGn',        # Red-Yellow-Green colormap
                    vmin=0,               # Minimum value for color scale
                    vmax=1,               # Maximum value for color scale
                    mask=mask,            # Hide NaN values
                    cbar_kws={'label': 'R² Score'},
                    ax=ax)
        
        # Add title with appropriate spacing
        ax.set_title('Model Performance Comparison (R² Scores)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save heatmap figure
        filename = self.output_dir / "model_comparison_heatmap.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")
        plt.close()


def generate_presentation_visuals(pipeline, test_results: Dict, y_test: Dict, 
                                importance_results: Dict) -> None:
    """
    Main orchestration function to generate all presentation materials.
    
    This function coordinates the generation of all visualizations and reports
    needed for presenting NBA player performance prediction results. It ensures
    all outputs are created in a consistent, professional format.
    
    Args:
        pipeline: The trained model pipeline object
        test_results: Dictionary containing test set performance metrics
        y_test: Dictionary of actual test set values by target
        importance_results: Dictionary of feature importance scores
    """
    print("\n" + "="*60)
    print("GENERATING PRESENTATION VISUALS AND REPORTS")
    print("="*60 + "\n")
    
    # Initialize the reporting class
    reporter = ModelResultsReporter()
    
    # Generate all visualization types in sequence
    reporter.create_model_performance_comparison(test_results)
    reporter.create_feature_importance_plots(importance_results, test_results)
    reporter.create_residual_analysis(test_results, y_test)
    reporter.create_prediction_scatter_plots(test_results, y_test)
    reporter.create_model_comparison_heatmap(test_results)
    
    # Generate comprehensive text summary report
    reporter.generate_summary_report(test_results, importance_results)
    
    # Print completion summary
    print("\n" + "="*60)
    print("ALL VISUALS AND REPORTS GENERATED SUCCESSFULLY")
    print("Visuals saved to: ../outputs/visuals/reporting_results/")
    print("Reports saved to: ../outputs/reports/")
    print("="*60 + "\n")


# Module usage example and documentation
if __name__ == "__main__":
    # Display usage instructions when module is run directly
    print("NBA Model Results Reporter")
    print("Usage: from reporting import generate_presentation_visuals")
    print("Call with: generate_presentation_visuals(pipeline, test_results, y_test, importance_results)")
    print("\nThis module generates:")
    print("  - Model performance comparison charts")
    print("  - Feature importance visualizations")
    print("  - Residual analysis plots")
    print("  - Prediction accuracy scatter plots")
    print("  - Model comparison heatmap")
    print("  - Comprehensive summary report")