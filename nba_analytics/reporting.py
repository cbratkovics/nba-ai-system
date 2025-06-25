"""
NBA Player Performance Prediction - Reporting and Visualization Module

Comprehensive reporting and visualization suite for NBA modeling results.

Author: Christopher Bratkovics
Created: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


class AdvancedVisualizer:
    """Create publication-ready visualizations for model results."""
    
    def __init__(self, pipeline, interpreter):
        self.pipeline = pipeline
        self.interpreter = interpreter
    
    def create_model_comparison_dashboard(self, test_results: Dict) -> None:
        """Create comprehensive model comparison dashboard."""
        print("Creating model performance dashboard...")
        
        metrics_data = []
        for target, models in test_results.items():
            for model_name, metrics in models.items():
                metrics_data.append({
                    'target': target.upper(),
                    'model': model_name.replace('_', ' ').title(),
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'mape': metrics['mape']
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Absolute Error', 'R2 Score', 'RMSE', 'MAPE (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set2
        
        # MAE
        for i, target in enumerate(df_metrics['target'].unique()):
            target_data = df_metrics[df_metrics['target'] == target]
            fig.add_trace(
                go.Bar(x=target_data['model'], y=target_data['mae'], 
                       name=f'{target} MAE', marker_color=colors[i], opacity=0.8),
                row=1, col=1
            )
        
        # R2
        for i, target in enumerate(df_metrics['target'].unique()):
            target_data = df_metrics[df_metrics['target'] == target]
            fig.add_trace(
                go.Bar(x=target_data['model'], y=target_data['r2'], 
                       name=f'{target} R2', marker_color=colors[i], opacity=0.8),
                row=1, col=2
            )
        
        # RMSE
        for i, target in enumerate(df_metrics['target'].unique()):
            target_data = df_metrics[df_metrics['target'] == target]
            fig.add_trace(
                go.Bar(x=target_data['model'], y=target_data['rmse'], 
                       name=f'{target} RMSE', marker_color=colors[i], opacity=0.8),
                row=2, col=1
            )
        
        # MAPE
        for i, target in enumerate(df_metrics['target'].unique()):
            target_data = df_metrics[df_metrics['target'] == target]
            fig.add_trace(
                go.Bar(x=target_data['model'], y=target_data['mape'], 
                       name=f'{target} MAPE', marker_color=colors[i], opacity=0.8),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="NBA Player Performance Model Comparison",
            showlegend=False,
            height=800
        )
        
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        fig.show()
    
    def create_feature_importance_plots(self, importance_results: Dict) -> None:
        """Create interactive feature importance visualizations."""
        print("Creating feature importance visualizations...")
        
        for target in importance_results.keys():
            if 'random_forest' in importance_results[target]:
                importance_df = importance_results[target]['random_forest'].head(15)
                model_name = 'Random Forest'
            else:
                model_name = list(importance_results[target].keys())[0]
                importance_df = importance_results[target][model_name].head(15)
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker_color='steelblue',
                opacity=0.8
            ))
            
            fig.update_layout(
                title=f'{target.upper()} Feature Importance ({model_name})',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            fig.show()
    
    def create_prediction_analysis(self, test_results: Dict, y_test: Dict) -> None:
        """Create prediction vs actual analysis plots."""
        print("Creating prediction analysis visualizations...")
        
        fig = make_subplots(
            rows=1, cols=len(y_test),
            subplot_titles=[f'{target.upper()} Predictions' for target in y_test.keys()]
        )
        
        for i, target in enumerate(y_test.keys(), 1):
            best_model = max(test_results[target], key=lambda x: test_results[target][x]['r2'])
            best_metrics = test_results[target][best_model]
            
            actual = y_test[target]
            predicted = best_metrics['predictions']
            
            fig.add_trace(
                go.Scatter(
                    x=actual, y=predicted,
                    mode='markers',
                    name=f'{target.upper()}',
                    opacity=0.6,
                    marker=dict(
                        size=8,
                        color='steelblue',
                        line=dict(width=1, color='white')
                    )
                ),
                row=1, col=i
            )
            
            min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )
            
            fig.update_xaxes(title_text=f'Actual {target.upper()}', row=1, col=i)
            fig.update_yaxes(title_text=f'Predicted {target.upper()}', row=1, col=i)
        
        fig.update_layout(
            title_text="Predictions vs Actual Values (Best Models)",
            height=500,
            showlegend=True
        )
        
        fig.show()
    
    def create_residual_analysis(self, test_results: Dict, y_test: Dict) -> None:
        """Create residual analysis plots."""
        print("Creating residual analysis...")
        
        fig, axes = plt.subplots(1, len(y_test), figsize=(6*len(y_test), 5))
        if len(y_test) == 1:
            axes = [axes]
        
        for i, target in enumerate(y_test.keys()):
            best_model = max(test_results[target], key=lambda x: test_results[target][x]['r2'])
            best_metrics = test_results[target][best_model]
            
            actual = y_test[target]
            predicted = best_metrics['predictions']
            residuals = actual - predicted
            
            axes[i].scatter(predicted, residuals, alpha=0.6, color='steelblue')
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[i].set_xlabel(f'Predicted {target.upper()}')
            axes[i].set_ylabel('Residuals')
            axes[i].set_title(f'{target.upper()} Residuals ({best_model.replace("_", " ").title()})')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_model_performance_summary(self, test_results: Dict) -> None:
        """Create a comprehensive performance summary visualization."""
        print("Creating model performance summary...")
        
        summary_data = []
        for target, models in test_results.items():
            for model_name, metrics in models.items():
                summary_data.append({
                    'Target': target.upper(),
                    'Model': model_name.replace('_', ' ').title(),
                    'R2': metrics['r2'],
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'MAPE': metrics['mape']
                })
        
        df_summary = pd.DataFrame(summary_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NBA Model Performance Summary', fontsize=16, fontweight='bold')
        
        # R2 comparison
        df_pivot = df_summary.pivot(index='Model', columns='Target', values='R2')
        df_pivot.plot(kind='bar', ax=axes[0,0], width=0.8)
        axes[0,0].set_title('Model R2 Scores by Target')
        axes[0,0].set_ylabel('R2 Score')
        axes[0,0].legend(title='Target')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        df_pivot_mae = df_summary.pivot(index='Model', columns='Target', values='MAE')
        df_pivot_mae.plot(kind='bar', ax=axes[0,1], width=0.8)
        axes[0,1].set_title('Model MAE by Target')
        axes[0,1].set_ylabel('Mean Absolute Error')
        axes[0,1].legend(title='Target')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Best model by target
        best_models = df_summary.loc[df_summary.groupby('Target')['R2'].idxmax()]
        axes[1,0].bar(best_models['Target'], best_models['R2'], color=['steelblue', 'orange', 'green'])
        axes[1,0].set_title('Best Model R2 by Target')
        axes[1,0].set_ylabel('R2 Score')
        
        # Performance distribution
        axes[1,1].hist(df_summary['R2'], bins=10, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1,1].set_title('Distribution of R2 Scores')
        axes[1,1].set_xlabel('R2 Score')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()


class ReportGenerator:
    """Generate comprehensive reports for NBA modeling results."""
    
    def __init__(self, test_results: Dict, insights: Dict):
        self.test_results = test_results
        self.insights = insights
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary."""
        lines = []
        lines.append("NBA PLAYER PERFORMANCE PREDICTION - EXECUTIVE SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("MODEL PERFORMANCE:")
        for target, performance in self.insights['model_performance'].items():
            lines.append(f"   {target.upper()}:")
            lines.append(f"     Best Model: {performance['best_model'].replace('_', ' ').title()}")
            lines.append(f"     Accuracy (R2): {performance['r2']:.3f}")
            lines.append(f"     Average Error: +/-{performance['mae']:.1f} {target}")
            lines.append(f"     Predictability: {performance['predictability']}")
        
        lines.append("")
        lines.append("KEY FINDINGS:")
        lines.append("   1. PREDICTABILITY ANALYSIS:")
        for target, performance in self.insights['model_performance'].items():
            pred_level = performance['predictability']
            r2_score = performance['r2']
            lines.append(f"      • {target.upper()}: {pred_level} (R2 = {r2_score:.3f})")
        
        lines.append("")
        lines.append("   2. TOP PERFORMANCE DRIVERS:")
        for target, drivers in self.insights['key_drivers'].items():
            if 'top_features' in drivers:
                lines.append(f"      • {target.upper()}: {', '.join(drivers['top_features'][:3])}")
        
        lines.append("")
        lines.append("BUSINESS VALUE:")
        lines.append("   • Provides data-driven edge in fantasy sports and betting")
        lines.append("   • Enables evidence-based coaching and player management decisions")
        lines.append("   • Creates opportunities for media content and fan engagement")
        lines.append("   • Supports objective player evaluation and contract negotiations")
        
        lines.append("")
        lines.append("TECHNICAL ACHIEVEMENTS:")
        lines.append("   • Implemented robust data leakage prevention")
        lines.append("   • Applied advanced feature engineering and selection")
        lines.append("   • Used time-aware cross-validation for realistic performance estimates")
        lines.append("   • Created production-ready deployment framework")
        lines.append("   • Achieved meaningful predictive accuracy across all target variables")
        
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        lines.append("   1. DEPLOY: Implement models in production for real-time predictions")
        lines.append("   2. MONITOR: Set up model performance monitoring and drift detection")
        lines.append("   3. ITERATE: Continuously retrain models with new data")
        lines.append("   4. EXPAND: Consider additional metrics and defensive statistics")
        lines.append("   5. INTEGRATE: Build API endpoints for external system integration")
        
        return "\n".join(lines)
    
    def generate_technical_report(self) -> str:
        """Generate detailed technical report."""
        lines = []
        lines.append("NBA PLAYER PERFORMANCE PREDICTION - TECHNICAL REPORT")
        lines.append("=" * 65)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("METHODOLOGY OVERVIEW")
        lines.append("-" * 25)
        lines.append("Data Pipeline: NBA API -> Cleaning -> Feature Engineering -> Modeling")
        lines.append("Validation Strategy: Time-series cross-validation with chronological splits")
        lines.append("Feature Selection: Multi-stage process with leakage detection")
        lines.append("Model Selection: Comprehensive comparison across model families")
        lines.append("")
        
        lines.append("MODEL PERFORMANCE DETAILS")
        lines.append("-" * 30)
        
        for target, models in self.test_results.items():
            lines.append(f"{target.upper()} PREDICTION RESULTS:")
            lines.append("Model".ljust(20) + "MAE".ljust(8) + "RMSE".ljust(8) + "R2".ljust(8) + "MAPE")
            lines.append("-" * 50)
            
            for model_name, metrics in models.items():
                model_display = model_name.replace('_', ' ').title()[:19]
                mae_str = f"{metrics['mae']:.3f}"
                rmse_str = f"{metrics['rmse']:.3f}"
                r2_str = f"{metrics['r2']:.3f}"
                mape_str = f"{metrics['mape']:.1f}%"
                
                lines.append(
                    model_display.ljust(20) + 
                    mae_str.ljust(8) + 
                    rmse_str.ljust(8) + 
                    r2_str.ljust(8) + 
                    mape_str
                )
            
            best_model = max(models, key=lambda x: models[x]['r2'])
            lines.append(f"Best Model: {best_model.replace('_', ' ').title()}")
            lines.append("")
        
        lines.append("FEATURE IMPORTANCE ANALYSIS")
        lines.append("-" * 35)
        for target, drivers in self.insights['key_drivers'].items():
            if 'top_features' in drivers:
                lines.append(f"{target.upper()} - Top Features:")
                for i, (feature, importance) in enumerate(zip(drivers['top_features'][:5], drivers['importance_scores'][:5]), 1):
                    lines.append(f"  {i}. {feature}: {importance:.4f}")
                lines.append("")
        
        lines.append("VALIDATION AND ROBUSTNESS")
        lines.append("-" * 30)
        lines.append("• Chronological data splits prevent data leakage")
        lines.append("• Comprehensive feature selection removes calculated targets")
        lines.append("• Cross-validation ensures generalization capability")
        lines.append("• Multiple model families tested for robustness")
        lines.append("• Statistical significance testing applied")
        lines.append("")
        
        lines.append("LIMITATIONS AND CONSIDERATIONS")
        lines.append("-" * 35)
        lines.append("• Predictions based on historical patterns, not real-time factors")
        lines.append("• Does not account for injuries, trades, or coaching changes")
        lines.append("• Performance may vary with rule changes or league evolution")
        lines.append("• Limited to players with sufficient historical data")
        lines.append("• Model performance dependent on data quality and completeness")
        
        return "\n".join(lines)
    
    def generate_business_report(self) -> str:
        """Generate business-focused report."""
        lines = []
        lines.append("NBA PLAYER PERFORMANCE PREDICTION - BUSINESS REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        lines.append("EXECUTIVE OVERVIEW")
        lines.append("-" * 20)
        lines.append("This project successfully developed an advanced analytical framework")
        lines.append("for predicting NBA player performance statistics and identifying key")
        lines.append("performance drivers through data-driven analysis.")
        lines.append("")
        
        lines.append("BUSINESS VALUE PROPOSITION")
        lines.append("-" * 30)
        
        stakeholder_sections = {
            'fantasy_managers': 'Fantasy Basketball Managers',
            'coaches_analysts': 'Coaches and Team Analysts', 
            'sports_media': 'Media and Broadcasters',
            'general_fans': 'Basketball Fans'
        }
        
        for key, title in stakeholder_sections.items():
            lines.append(f"{title.upper()}:")
            recommendations = self.insights['stakeholder_recommendations'][key]
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        lines.append("MARKET OPPORTUNITY")
        lines.append("-" * 20)
        lines.append("• Fantasy Sports Market: $8+ billion annually with growing engagement")
        lines.append("• Sports Betting: Rapid legalization creating demand for analytics")
        lines.append("• Media Content: Data-driven narratives enhance viewer engagement")
        lines.append("• Team Analytics: Front offices investing heavily in data science")
        lines.append("")
        
        lines.append("COMPETITIVE ADVANTAGES")
        lines.append("-" * 25)
        lines.append("• Comprehensive leakage prevention ensures model reliability")
        lines.append("• Multi-target prediction provides holistic player assessment")
        lines.append("• Production-ready deployment enables immediate implementation")
        lines.append("• Interpretable insights support decision-making processes")
        lines.append("• Scalable architecture accommodates growing data volumes")
        lines.append("")
        
        lines.append("IMPLEMENTATION ROADMAP")
        lines.append("-" * 25)
        lines.append("Phase 1: Deploy core prediction models for fantasy applications")
        lines.append("Phase 2: Develop real-time API for live game predictions")
        lines.append("Phase 3: Integrate opponent matchup data for enhanced accuracy")
        lines.append("Phase 4: Expand to defensive metrics and advanced statistics")
        lines.append("Phase 5: Create automated reporting and alert systems")
        
        return "\n".join(lines)
    
    def save_all_reports(self, output_dir: str = "reports") -> None:
        """Save all generated reports to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Executive Summary
        exec_summary = self.generate_executive_summary()
        exec_file = output_path / f"executive_summary_{timestamp}.txt"
        with open(exec_file, 'w') as f:
            f.write(exec_summary)
        
        # Technical Report
        tech_report = self.generate_technical_report()
        tech_file = output_path / f"technical_report_{timestamp}.txt"
        with open(tech_file, 'w') as f:
            f.write(tech_report)
        
        # Business Report
        business_report = self.generate_business_report()
        business_file = output_path / f"business_report_{timestamp}.txt"
        with open(business_file, 'w') as f:
            f.write(business_report)
        
        print(f"All reports saved to {output_path}")
        print(f"Files created:")
        print(f"  - {exec_file.name}")
        print(f"  - {tech_file.name}")
        print(f"  - {business_file.name}")


class PerformanceAnalyzer:
    """Analyze model performance and generate insights."""
    
    def __init__(self, test_results: Dict, insights: Dict):
        self.test_results = test_results
        self.insights = insights
    
    def analyze_model_convergence(self, pipeline) -> Dict:
        """Analyze how models converged during training."""
        convergence_analysis = {}
        
        for target in self.test_results.keys():
            target_analysis = {}
            
            # Get validation results for convergence analysis
            if hasattr(pipeline, 'results') and target in pipeline.results:
                for model_name, results in pipeline.results[target].items():
                    if hasattr(results, 'cv_scores') and results.cv_scores is not None:
                        target_analysis[model_name] = {
                            'cv_mean': results.cv_scores.mean(),
                            'cv_std': results.cv_scores.std(),
                            'stability': 'High' if results.cv_scores.std() < 1.0 else 'Medium' if results.cv_scores.std() < 2.0 else 'Low'
                        }
            
            convergence_analysis[target] = target_analysis
        
        return convergence_analysis
    
    def compare_target_predictability(self) -> Dict:
        """Compare predictability across different targets."""
        comparison = {}
        
        r2_scores = {}
        mae_scores = {}
        
        for target, performance in self.insights['model_performance'].items():
            r2_scores[target] = performance['r2']
            mae_scores[target] = performance['mae']
        
        # Rank targets by predictability
        sorted_targets = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
        
        comparison['predictability_ranking'] = [
            {'target': target, 'r2': r2, 'rank': i+1} 
            for i, (target, r2) in enumerate(sorted_targets)
        ]
        
        comparison['insights'] = []
        
        if sorted_targets[0][1] > 0.6:
            comparison['insights'].append(f"{sorted_targets[0][0].upper()} shows high predictability - suitable for confident forecasting")
        
        if sorted_targets[-1][1] < 0.3:
            comparison['insights'].append(f"{sorted_targets[-1][0].upper()} shows low predictability - influenced by many external factors")
        
        r2_range = sorted_targets[0][1] - sorted_targets[-1][1]
        if r2_range > 0.3:
            comparison['insights'].append("Significant variation in predictability across targets suggests different underlying mechanisms")
        
        return comparison
    
    def generate_feature_insights(self, importance_results: Dict) -> Dict:
        """Generate insights about feature importance patterns."""
        feature_insights = {}
        
        # Collect all features across targets
        all_features = set()
        for target in importance_results.keys():
            if 'random_forest' in importance_results[target]:
                features = importance_results[target]['random_forest']['feature'].tolist()
                all_features.update(features)
        
        # Find features important across multiple targets
        feature_counts = {}
        for feature in all_features:
            count = 0
            for target in importance_results.keys():
                if 'random_forest' in importance_results[target]:
                    top_features = importance_results[target]['random_forest'].head(10)['feature'].tolist()
                    if feature in top_features:
                        count += 1
            feature_counts[feature] = count
        
        # Universal features (important for all targets)
        universal_features = [f for f, count in feature_counts.items() if count == len(importance_results)]
        
        # Specific features (important for only one target)
        specific_features = [f for f, count in feature_counts.items() if count == 1]
        
        feature_insights['universal_drivers'] = universal_features
        feature_insights['target_specific_drivers'] = specific_features
        
        # Generate insights
        insights = []
        if universal_features:
            insights.append(f"Universal performance drivers: {', '.join(universal_features[:3])}")
        
        if len(specific_features) > len(universal_features):
            insights.append("Performance is largely target-specific, requiring customized approaches")
        else:
            insights.append("Performance shares common drivers across targets, enabling unified strategies")
        
        feature_insights['insights'] = insights
        
        return feature_insights


def create_comprehensive_visualization_suite(pipeline, interpreter, test_results: Dict, 
                                           y_test: Dict, importance_results: Dict) -> None:
    """Create a comprehensive suite of visualizations."""
    print("CREATING COMPREHENSIVE VISUALIZATION SUITE")
    print("-" * 50)
    
    visualizer = AdvancedVisualizer(pipeline, interpreter)
    
    # Main dashboard
    visualizer.create_model_comparison_dashboard(test_results)
    
    # Feature importance
    visualizer.create_feature_importance_plots(importance_results)
    
    # Prediction analysis
    visualizer.create_prediction_analysis(test_results, y_test)
    
    # Residual analysis
    visualizer.create_residual_analysis(test_results, y_test)
    
    # Performance summary
    visualizer.create_model_performance_summary(test_results)
    
    print("Visualization suite complete!")


def generate_comprehensive_reports(test_results: Dict, insights: Dict, 
                                 output_dir: str = "reports") -> None:
    """Generate all comprehensive reports."""
    print("GENERATING COMPREHENSIVE REPORTS")
    print("-" * 40)
    
    report_generator = ReportGenerator(test_results, insights)
    
    # Generate and print executive summary
    exec_summary = report_generator.generate_executive_summary()
    print("\n" + exec_summary)
    
    # Save all reports
    report_generator.save_all_reports(output_dir)
    
    print("Report generation complete!")


def analyze_model_performance(test_results: Dict, insights: Dict, pipeline=None) -> Dict:
    """Perform comprehensive performance analysis."""
    print("PERFORMING COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    analyzer = PerformanceAnalyzer(test_results, insights)
    
    # Target predictability comparison
    predictability_analysis = analyzer.compare_target_predictability()
    
    print("PREDICTABILITY RANKING:")
    for item in predictability_analysis['predictability_ranking']:
        print(f"  {item['rank']}. {item['target']}: R2 = {item['r2']:.3f}")
    
    print("\nKEY INSIGHTS:")
    for insight in predictability_analysis['insights']:
        print(f"  - {insight}")
    
    # Model convergence analysis (if pipeline available)
    convergence_analysis = {}
    if pipeline:
        convergence_analysis = analyzer.analyze_model_convergence(pipeline)
        
        if convergence_analysis:
            print("\nMODEL STABILITY ANALYSIS:")
            for target, models in convergence_analysis.items():
                if models:
                    print(f"  {target.upper()}:")
                    for model_name, stats in models.items():
                        print(f"    {model_name}: {stats['stability']} stability (CV std: {stats['cv_std']:.3f})")
    
    return {
        'predictability_analysis': predictability_analysis,
        'convergence_analysis': convergence_analysis
    }


def create_model_comparison_table(test_results: Dict) -> pd.DataFrame:
    """Create a detailed model comparison table."""
    comparison_data = []
    
    for target, models in test_results.items():
        for model_name, metrics in models.items():
            comparison_data.append({
                'Target': target.upper(),
                'Model': model_name.replace('_', ' ').title(),
                'MAE': round(metrics['mae'], 3),
                'RMSE': round(metrics['rmse'], 3),
                'R2': round(metrics['r2'], 3),
                'MAPE (%)': round(metrics['mape'], 1),
                'Rank_R2': 0  # Will be filled below
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Add ranking within each target
    for target in df_comparison['Target'].unique():
        target_mask = df_comparison['Target'] == target
        target_data = df_comparison[target_mask].copy()
        target_data['Rank_R2'] = target_data['R2'].rank(ascending=False)
        df_comparison.loc[target_mask, 'Rank_R2'] = target_data['Rank_R2']
    
    return df_comparison


def export_results_for_presentation(test_results: Dict, insights: Dict, output_dir: str = "reports") -> None:
    """Export key results in formats suitable for presentations."""
    output_path = Path("../outputs/reports")  # Create Path object
    output_path.mkdir(exist_ok=True, parents=True)  # Ensure directory exists
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model comparison table
    comparison_table = create_model_comparison_table(test_results)
    comparison_file = output_path / f"model_comparison_{timestamp}.csv"
    comparison_table.to_csv(comparison_file, index=False)
    
    # Key metrics summary
    summary_data = {}
    for target, performance in insights['model_performance'].items():
        summary_data[target] = {
            'best_model': performance['best_model'],
            'r2_score': performance['r2'],
            'mae': performance['mae'],
            'predictability': performance['predictability']
        }
    
    summary_file = output_path / f"key_metrics_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Feature importance summary
    feature_summary = {}
    for target, drivers in insights['key_drivers'].items():
        if 'top_features' in drivers:
            feature_summary[target] = {
                'top_5_features': drivers['top_features'][:5],
                'importance_scores': drivers['importance_scores'][:5]
            }
    
    features_file = output_path / f"feature_importance_{timestamp}.json"
    with open(features_file, 'w') as f:
        json.dump(feature_summary, f, indent=2)
    
    print(f"Presentation exports saved to {output_path}")
    print(f"Files created:")
    print(f"  - {comparison_file.name} (model comparison table)")
    print(f"  - {summary_file.name} (key metrics)")
    print(f"  - {features_file.name} (feature importance)")


if __name__ == "__main__":
    print("NBA Reporting and Visualization Module")
    print("This module should be imported and used with NBA_Model_Pipeline results")
    print("Example usage:")
    print("  from Generate_Reports import *")
    print("  create_comprehensive_visualization_suite(pipeline, interpreter, test_results, y_test, importance_results)")
    print("  generate_comprehensive_reports(test_results, insights)")
    print("  analyze_model_performance(test_results, insights, pipeline)")