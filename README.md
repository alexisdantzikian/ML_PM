# Macro Factor Timing

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for portfolio optimization and investment strategy backtesting using Machine Learning.

The `skfin` library is derived from the **Machine Learning for Portfolio Management and Trading** course at [ENSAE Paris](https://www.ensae.fr/).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Main Notebook](#main-notebook-nbsmainipynb)
- [Using skfin](#using-the-skfin-library)
- [Authors](#authors)

## Features

- **Regime Detection**: Factor rotation framework based on Jump Model segmentation
- **Portfolio Optimization**: Mean-Variance and MBJ (Britten-Jones) estimators
- **Data Loading**: Loaders for Ken French and FRED-MD data
- **Metrics**: Sharpe ratio and drawdown indicators
- **Visualization**: Line plots and heatmaps for performance analysis

## Installation

### Prerequisites

- Python 3.13
- macOS (tested) / Linux

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "ML for PM"

# Create a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Project Structure

```
ML for PM/
├── skfin/                      # Main package
│   ├── __init__.py
│   ├── datasets_.py            # Dataset loading functions
│   ├── metrics.py              # Performance metrics
│   ├── mv_estimators.py        # Mean-Variance and MBJ optimization
│   ├── plot.py                 # Visualization tools
│   └── dataloaders/            # Data loaders
│       ├── cache.py            # Cache management
│       ├── cleaners.py         # Data cleaning
│       ├── loaders.py          # Main loaders
│       └── constants/          # Constants and mappings
├── nbs/                        # Jupyter notebooks
│   └── main.ipynb
├── data/                       # Data (Fama-French, etc.)
├── requirements.txt
└── pyproject.toml
```

## Main Notebook (`nbs/main.ipynb`)

The `main.ipynb` notebook implements a **regime-based factor rotation framework**, inspired by **Yu, Mulvey, and Nie (2025)**. The goal is to dynamically predict which investment factor (Momentum, Value, Size) will outperform under current market conditions.

### Two-Stage Pipeline

1. **Labeling (Jump Model)**: A penalized segmentation algorithm identifies periods of persistent factor outperformance relative to the benchmark ($r_{\text{factor}} > r_{\text{benchmark}}$).

2. **Forecasting (XGBoost)**: A classifier predicts these regimes using enriched macroeconomic data (FRED-MD), with transformations such as exponential moving averages (EMA) and slopes to capture non-linear market dynamics.

### Notebook Sections

| Section | Description |
|---------|-------------|
| **I. Data Preprocessing** | Loading and transforming FRED-MD data (128 macro variables) |
| **II. Labeling** | Regime detection via Jump Model (`ruptures` library) |
| **III. Feature Engineering** | Creating predictive features (EMA, slopes, spreads) |
| **IV. Modeling** | Training XGBoost classifier to predict regimes |
| **V. Backtesting** | Evaluating the dynamic multi-factor strategy |

### Data Sources

- **FRED-MD**: ~128 macroeconomic variables (output, employment, prices, rates, etc.)
- **Fama-French Factors**: Mkt-RF, SMB, HML, Momentum (via Ken French Data Library)
- **VIX, S&P 500, Oil Prices**: Additional market indicators

### Usage Example

```python
from skfin.datasets_ import load_kf_returns
from skfin.metrics import drawdown
from skfin.mv_estimators import Mbj
from skfin.plot import heatmap, line
import ruptures as rpt
from xgboost import XGBClassifier

# Load Fama-French factors
factors = load_kf_returns("F-F_Research_Data_Factors")

# Load FRED-MD from repo
url_data = "https://raw.githubusercontent.com/lxsd111/ML_PM/main/nbs/data/2025-09-MD.csv"
fred = pd.read_csv(url_data, index_col=0)

# Apply stationarity transformations (tcodes)
# ... see notebook for complete code
```

## Using the `skfin` Library

### Loading Ken French Data

```python
from skfin.datasets_ import load_kf_returns

# Load Fama-French factors
factors = load_kf_returns("F-F_Research_Data_Factors")
```

### Computing Metrics

```python
from skfin.metrics import sharpe_ratio, drawdown

# Compute Sharpe ratio
sr = sharpe_ratio(returns)

# Compute drawdown
dd = drawdown(returns)
```

### Portfolio Optimization

```python
from skfin.mv_estimators import Mbj

# MBJ estimator (Britten-Jones) - unconstrained mean-variance weights
mbj = Mbj()
mbj.fit(returns)
weights = mbj.coef_
```

### Visualization

```python
from skfin.plot import line, heatmap

# Plot cumulative returns with Sharpe ratio in legend
line(returns, cumsum=True)

# Plot correlation heatmap
heatmap(returns.corr())
```

## Main Dependencies

- **numpy / pandas**: Data manipulation
- **scikit-learn**: ML framework
- **xgboost**: Gradient boosting classifier
- **ruptures**: Change point detection
- **matplotlib / seaborn**: Visualization

## Authors

- **Alexis Dantzikian** - [alexis.dantzikian@gmail.com](mailto:alexis.dantzikian@gmail.com)
- **Antonin Devalland**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
