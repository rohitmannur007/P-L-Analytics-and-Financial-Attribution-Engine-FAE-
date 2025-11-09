# P&L Analytics and Financial Attribution Engine (FAE)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](https://github.com/rohitmannur007/P-L-Analytics-and-Financial-Attribution-Engine-FAE-/actions)

## Overview

The **P&L Analytics and Financial Attribution Engine (FAE)** is a modular Python library designed for quantitative finance professionals, portfolio managers, and analysts. It provides tools to decompose portfolio performance (Profit & Loss or P&L) into attributable components, such as market timing, stock selection, sector allocation, and factor exposures. Built on robust libraries like Pandas, NumPy, and SciPy, FAE supports classical methods like Brinson-Fachler attribution, factor-based models (e.g., Fama-French), risk decomposition, and transaction cost analysis.

Key features:
- **Performance Attribution**: Break down returns using Brinson, Information Ratio, or custom models.
- **Factor Models**: Integrate multi-factor models for equity, fixed income, and alternatives.
- **Risk Analytics**: Compute VaR, CVaR, tracking error, and attribution to risk factors.
- **P&L Visualization**: Interactive dashboards and reports for insights.
- **Modular & Extensible**: Easy to integrate into existing workflows.

FAE is ideal for hedge funds, asset managers, and research teams seeking transparent, reproducible attribution analysis.

## Project Structure

The repository is organized into core modules, utilities, tests, and supporting files. Below is a breakdown of each file and directory, with explanations of their purpose and key contents:

```
P-L-Analytics-and-Financial-Attribution-Engine-FAE/
├── attribution/                  # Core attribution logic
│   ├── __init__.py              # Package initializer; exposes main classes/functions for easy import (e.g., `from attribution import BrinsonAttribution`).
│   ├── brinson.py               # Implements Brinson-Hood-Beebower and Brinson-Fachler models. Key functions: `brinson_decomposition(returns, benchmarks)` – decomposes portfolio returns into allocation, selection, and interaction effects. Uses Pandas DataFrames for input.
│   ├── factor_attribution.py    # Factor-based attribution using linear regression. Includes `factor_exposure_attribution(portfolio_returns, factors)` to attribute P&L to market, size, value, etc. Supports custom factor libraries.
│   ├── risk_attribution.py      # Decomposes risk (e.g., volatility) into systematic and idiosyncratic components. Features `risk_budgeting(marginals, weights)` for risk parity analysis.
│   └── transaction_cost.py      # Models transaction costs' impact on P&L. Functions like `estimate_tc_impact(trades, prices)` calculate slippage, commissions, and market impact.
├── factor_models/               # Factor model definitions and utilities
│   ├── __init__.py              # Package initializer; imports model classes.
│   ├── models.py                # Core factor models: Fama-French 3/5-factor, Barra, or user-defined. Example: `FF3Model().fit(returns)` generates factor loadings and residuals.
│   └── utils.py                 # Helper functions for data prep, e.g., `fetch_factors(tickers, start_date)` pulls data from Yahoo Finance or local sources; normalization utilities.
├── risk/                        # Risk management tools
│   └── risk_manager.py          # Comprehensive risk engine. Computes metrics like VaR (`historical_var(returns, confidence=0.95)`), Tail Ratio, and Sharpe. Includes Monte Carlo simulation for stress testing.
├── tests/                       # Test suites for reliability
│   ├── unit/                    # Unit tests for individual functions (e.g., test_brinson.py validates decomposition math using mock data).
│   └── integration/             # End-to-end tests (e.g., test_full_pipeline.py simulates a portfolio workflow from data load to attribution report).
├── data/                        # Sample datasets (not committed; use .gitignore for large files)
│   ├── sample_portfolio.csv     # Example portfolio holdings and returns (hypothetical data for testing).
│   └── benchmarks.csv           # Benchmark indices (e.g., S&P 500, Russell 2000).
├── main.py                      # Entry point for CLI or demo scripts. Runs a sample attribution analysis: loads data, computes P&L breakdown, and exports to CSV/PDF.
├── requirements.txt             # Dependencies: Lists libraries like `pandas==2.0.3`, `numpy==1.24.3`, `scipy==1.10.1`, `matplotlib==3.7.1`, `yfinance==0.2.18`. Install with `pip install -r requirements.txt`.
├── .env                         # Environment variables template (e.g., API keys for data sources). Not committed; copy to .env.local for use.
├── .gitattributes               # Git config for handling line endings and large files (e.g., *.csv filter=lfs).
├── LICENSE                      # MIT License: Permissive open-source license allowing commercial use with attribution.
└── README.md                    # This file: Project documentation, installation, and usage guides.
```

**Note**: The `data/` directory contains placeholder CSV files for demonstration. Real-world usage requires loading your own datasets (e.g., via Pandas from SQL/Excel).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rohitmannur007/P-L-Analytics-and-Financial-Attribution-Engine-FAE-.git
   cd P-L-Analytics-and-Financial-Attribution-Engine-FAE-
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv fae_env
   source fae_env/bin/activate  # On Windows: fae_env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Set up environment variables in `.env` for external data sources.

## Quick Start

Run a basic Brinson attribution example:

```python
import pandas as pd
from attribution.brinson import BrinsonAttribution

# Load sample data
portfolio_returns = pd.read_csv('data/sample_portfolio.csv', index_col=0, parse_dates=True)
benchmark_returns = pd.read_csv('data/benchmarks.csv', index_col=0, parse_dates=True)

# Initialize and compute
attr = BrinsonAttribution(portfolio_returns, benchmark_returns)
decomposition = attr.decompose()

# View results
print(decomposition[['Allocation', 'Selection', 'Interaction']])
```

Output example (DataFrame snippet):
| Date       | Allocation | Selection | Interaction |
|------------|------------|-----------|-------------|
| 2023-01-01 | 0.015     | 0.008    | 0.002      |
| 2023-02-01 | -0.005    | 0.012    | -0.001     |

For full pipelines, execute `python main.py --input data/sample_portfolio.csv --output report.pdf`.

## Usage Examples

### 1. Factor Attribution
```python
from factor_models.models import FF3Model
from attribution.factor_attribution import factor_exposure_attribution

model = FF3Model()
loadings = model.fit(portfolio_returns)

attribution = factor_exposure_attribution(portfolio_returns, loadings)
print(attribution['Market'], attribution['SMB'], attribution['HML'])
```

### 2. Risk Decomposition
```python
from risk.risk_manager import RiskManager

rm = RiskManager(returns=portfolio_returns)
var_95 = rm.compute_var(confidence=0.95, method='historical')
print(f"95% VaR: {var_95:.4f}")
```

### 3. Transaction Cost Impact
```python
from attribution.transaction_cost import estimate_tc_impact

trades = pd.DataFrame({'symbol': ['AAPL', 'GOOG'], 'shares': [100, 50], 'price': [150, 2800]})
impact = estimate_tc_impact(trades, slippage=0.001, commission=0.005)
print(f"Total Cost Impact on P&L: ${impact:.2f}")
```

## Contributing

1. Fork the repo and create a feature branch (`git checkout -b feature/amazing-feature`).
2. Commit changes (`git commit -m 'Add amazing feature'`).
3. Push to the branch (`git push origin feature/amazing-feature`).
4. Open a Pull Request.

Run tests before submitting:
```
pytest tests/unit/ tests/integration/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with inspiration from Brinson's seminal work on performance attribution.
- Thanks to the open-source community for libraries like Pandas and SciPy.
- Contributions welcome! Contact: rohitmannur@gmail.com

## Support

If you encounter issues, open a GitHub issue or reach out via email. For enterprise features or custom models, consider sponsorship.

---

*Last updated: November 09, 2025*
