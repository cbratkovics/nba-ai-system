# 🏀 NBA Player Performance Prediction & Analytics

<div align="center">

![NBA Analytics](https://img.shields.io/badge/NBA-Analytics-orange?style=for-the-badge&logo=basketball)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-green?style=for-the-badge&logo=scikit-learn)
![Data Science](https://img.shields.io/badge/Data-Science-purple?style=for-the-badge&logo=jupyter)

*Predicting NBA player performance with data-driven insights and hypothesis-driven analysis*

[🚀 Quick Start](#-quick-start) • [📊 Features](#-features) • [🔬 Methodology](#-methodology) • [📈 Results](#-results) • [🛠️ Installation](#️-installation)

</div>

---

## 🎯 Project Overview

This project develops an **advanced analytical framework** for predicting individual NBA player statistics while statistically investigating the key factors driving game-to-game performance variability. By combining **predictive modeling** with **rigorous hypothesis testing**, we move beyond anecdotal sports commentary to deliver data-driven insights.

### 🎲 What We Predict
- **🏀 Points** - Player scoring performance
- **📦 Rebounds** - Defensive and offensive rebounding
- **🤝 Assists** - Playmaking and ball distribution

### 🔍 Key Questions Answered
1. **Does rest impact shooting efficiency?** ✅ *Statistically significant but small effect*
2. **Is home court advantage real for individuals?** ✅ *Yes, +0.11 points per game*
3. **Are 3-point attempts still increasing?** ✅ *+0.56 attempts per 36 minutes (2022→2024)*

---

## 🏆 Key Achievements

<div align="center">

| Metric | Points | Rebounds | Assists |
|--------|--------|----------|---------|
| **R² Score** | 🔥 **0.949** | 📈 **0.723** | 📊 **0.714** |
| **Best Model** | Random Forest | Random Forest | Gradient Boosting |
| **Avg Error** | ±1.13 pts | ±1.04 reb | ±0.75 ast |
| **Predictability** | Excellent | Good | Good |

</div>

> **🎯 95% of point scoring variance explained** - Our models achieve exceptional accuracy for NBA prediction standards

---

## 📊 Features

### 🤖 **Advanced Machine Learning Pipeline**
- **5 Model Types**: Linear Regression, Ridge, Elastic Net, Random Forest, Gradient Boosting
- **Smart Feature Selection**: Automated leakage detection and removal
- **Time-Aware Validation**: Chronological splits prevent data leakage
- **Production Ready**: Deployable models with standardized interfaces

### 🔬 **Hypothesis-Driven Analysis**
- **Statistical Rigor**: Proper significance testing with effect size analysis
- **Basketball Context**: Tests designed around real NBA scenarios
- **Visual Insights**: Comprehensive plots and distribution analysis

### 🛠️ **Professional Development Practices**
- **Modular Architecture**: Clean, reusable code components
- **Comprehensive Documentation**: Detailed docstrings and comments
- **Version Control**: Git workflow with proper branching
- **Reproducible Results**: Consistent random seeds and validation

---

## 🔬 Methodology

### 📡 **Data Pipeline**
```
NBA API → Data Cleaning → Feature Engineering → Model Training → Evaluation
```

- **Data Source**: BallDontLie.io API (2021-2025 seasons)
- **Records**: 169,161 player-game observations
- **Features**: 30+ engineered features (post-leakage removal)
- **Validation**: Time-series cross-validation

### 🧪 **Feature Engineering Highlights**
- **Rest Days Analysis**: Days between games for each player
- **Home/Away Context**: Game location impact quantification  
- **Position-Specific Features**: Role-based performance metrics
- **Interaction Features**: Minutes × Rest, Position × Usage patterns
- **Leakage Prevention**: Automated detection of calculated target features

### 📈 **Model Selection Process**
1. **Baseline Models**: Linear regression for interpretability
2. **Ensemble Methods**: Random Forest and Gradient Boosting for accuracy
3. **Hyperparameter Tuning**: Grid search with cross-validation
4. **Performance Metrics**: MAE, RMSE, R², MAPE for comprehensive evaluation

---

## 📈 Results

### 🎯 **Model Performance**

<details>
<summary><b>📊 Detailed Performance Metrics</b></summary>

| Target | Model | MAE | RMSE | R² | Interpretation |
|--------|-------|-----|------|----|----|
| **Points** | Random Forest | 1.13 | 1.67 | 0.949 | Exceptional accuracy |
| **Rebounds** | Random Forest | 1.04 | 1.53 | 0.723 | Strong predictive power |
| **Assists** | Gradient Boosting | 0.75 | 1.15 | 0.714 | Reliable predictions |

</details>

### 🔍 **Feature Importance Insights**

**🏀 Points Prediction:**
1. `minutes_played` - Playing time is king
2. `fga_per_min` - Shot volume drives scoring
3. `minutes_played_x_rest_days` - Quality minutes matter

**📦 Rebounds Prediction:**
1. `minutes_played` - Court time = opportunities  
2. `player_position_C` - Centers dominate rebounding
3. `sufficient_rest_x_minutes_played` - Fresh legs help

**🤝 Assists Prediction:**
1. `ast_outlier_flag` - Identifies primary playmakers
2. `minutes_played` - More time = more distribution opportunities
3. `player_position_G` - Guards facilitate offense

### 📊 **Hypothesis Testing Results**

| Hypothesis | Result | p-value | Effect Size | Practical Impact |
|------------|--------|---------|-------------|------------------|
| **Rest → Shooting** | ✅ Significant | < 0.001 | Small (d=0.034) | +0.58% FG% when rested |
| **Home → Scoring** | ✅ Significant | 0.010 | Small (d=0.013) | +0.11 points at home |
| **3PT Evolution** | ✅ Significant | < 0.0001 | Small (d=0.158) | +0.56 attempts/36min |

---

## 🛠️ Installation

### 📋 **Prerequisites**
- Python 3.8+
- 8GB+ RAM (for model training)
- NBA API access (free)

### ⚡ **Quick Setup**

```bash
# Clone the repository
git clone https://github.com/cbratkovics/NBA_Analytics.git
cd NBA_Analytics

# Create virtual environment
python -m venv nba_env
source nba_env/bin/activate  # On Windows: nba_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Jupyter kernel (optional)
python -m ipykernel install --user --name=nba_env
```

### 📦 **Key Dependencies**
```
pandas>=1.5.0
scikit-learn>=1.2.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.9.0
```

---

## 🚀 Quick Start

### 🎮 **Run Complete Pipeline**

```python
from NBA_Model_Pipeline import run_nba_modeling_pipeline

# Execute full modeling pipeline
pipeline, test_results, insights, production_manager = run_nba_modeling_pipeline()

# View performance summary
for target, performance in insights['model_performance'].items():
    print(f"{target.upper()}: {performance['best_model']} (R²={performance['r2']:.3f})")
```

### 🔮 **Make Predictions**

```python
# Load production models
predict_performance = production_manager.create_prediction_function()

# Example prediction
player_data = {
    'minutes_played': 32,
    'rest_days': 2,
    'is_home_game': True,
    'player_position': 'G'
}

predictions = predict_performance(player_data)
print(f"Predicted: {predictions['pts']:.1f} pts, {predictions['reb']:.1f} reb, {predictions['ast']:.1f} ast")
```

### 📊 **Run Hypothesis Tests**

```python
from NBA_Hypothesis_Tester import run_nba_hypothesis_tests
import pandas as pd

# Load your data
df = pd.read_parquet('data/processed/final_engineered_nba_data.parquet')

# Run all hypothesis tests
results, tester = run_nba_hypothesis_tests(df)

# Generate visualizations
tester.create_visualization_plots()
```

---

## 📁 Project Structure

```
NBA_Analytics/
├── 📊 data/
│   ├── raw/                     # Original API data
│   └── processed/               # Cleaned and engineered data
├── 🧠 model_artifacts/          # Trained models and results
├── 📈 production_models/        # Deployment-ready models
├── 📋 reports/                  # Generated analysis reports
├── 🎨 visuals/                  # EDA and result visualizations
├── 🔧 NBA_Data_Cleaner.py       # Data cleaning pipeline
├── ⚙️ NBA_Feature_Engineer.py   # Feature engineering tools
├── 🧪 NBA_Hypothesis_Tester.py  # Statistical testing framework
├── 🤖 NBA_Model_Pipeline.py     # Main modeling pipeline
├── 📊 NBA_EDA.py               # Exploratory data analysis
├── 📑 Generate_Reports.py       # Automated reporting
└── 📓 notebooks/               # Jupyter analysis notebooks
```

---

## 🎯 Business Applications

### 🏀 **Fantasy Sports**
- **Player Selection**: Identify consistent performers
- **Lineup Optimization**: Rest and matchup considerations
- **Streaming Strategy**: Target players with favorable conditions

### 📺 **Sports Media**
- **Data-Driven Narratives**: Quantify common assumptions
- **Performance Context**: Explain why players excel/struggle
- **Trend Analysis**: Track league evolution with evidence

### 🏟️ **Team Analytics**
- **Load Management**: Optimize rest strategies
- **Player Evaluation**: Context-aware performance assessment
- **Strategic Insights**: Home court and opponent analysis

---

## 🔮 Future Enhancements

- [ ] **Real-time API Integration**: Live game predictions
- [ ] **Defensive Metrics**: Expand beyond offensive stats
- [ ] **Injury Impact Modeling**: Quantify recovery effects
- [ ] **Team Chemistry Features**: Lineup-based interactions
- [ ] **Advanced Visualizations**: Interactive dashboards
- [ ] **Model Ensemble**: Combine predictions across models

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### 🛠️ **Development Setup**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **BallDontLie.io** for providing comprehensive NBA API access
- **NBA** for the rich statistical ecosystem
- **Scikit-learn community** for robust ML tools
- **Sports analytics community** for inspiration and best practices

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

*Built with ❤️ for the intersection of sports and data science*

</div>
