# xVA Calculation Engine 📊

[![CI Status](https://github.com/user/xva-project/workflows/CI/badge.svg)](https://github.com/user/xva-project/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade educational implementation** of xVA (CVA/DVA/FVA/MVA/KVA) calculations for counterparty credit risk, featuring Monte Carlo simulation, collateral management, and regulatory capital (SA-CCR).

![xVA Engine Demo](docs/demo.png)

---

## ✨ Features

- 🎲 **Monte Carlo Simulation**: Ornstein-Uhlenbeck short-rate and GBM FX models with correlation
- 📈 **Exposure Calculation**: Path-level MTM, netting sets, EPE/ENE/PFE profiles
- 🏦 **Collateral Management**: Variation margin (with MPR lag) and initial margin
- 💰 **xVA Metrics**: CVA, DVA, FVA, MVA, KVA with rigorous discrete approximations
- 🏛️ **Regulatory Capital**: Simplified SA-CCR (Basel III)
- 🖥️ **Interactive UI**: Streamlit app with real-time parameter adjustment
- 📊 **Rich Visualizations**: Plotly charts for exposure, xVA breakdown, SA-CCR
- 🧪 **Robust Testing**: 30+ unit/integration tests with >80% coverage
- 📦 **Clean Architecture**: Type-safe, well-documented, PEP-compliant

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/user/xva-project.git
cd xva-project

# Install with UI and dev dependencies
pip install -e ".[ui,dev]"
```

### Run Demo

```bash
# CLI demo (prints results + saves files)
python examples/run_demo.py

# Streamlit app
streamlit run xva_app/app.py

# Jupyter notebooks
jupyter notebook notebooks/
```

### Run Tests

```bash
pytest --cov=xva_core --cov-report=html
```

---

## 📐 Mathematical Framework

### Interest Rate Model (Ornstein-Uhlenbeck)

The short rate follows the SDE:

$$dr = \kappa(\theta - r) \, dt + \sigma \, dW$$

where $\kappa$ is mean reversion speed, $\theta$ is long-term mean, and $\sigma$ is volatility.

### FX Model (Geometric Brownian Motion)

Under the domestic risk-neutral measure:

$$\frac{dS}{S} = (r_d - r_f) \, dt + \sigma_S \, dW_S$$

### Exposure Metrics

**Expected Positive Exposure (EPE)**:

$$\text{EPE}(t) = \mathbb{E}\left[\max(V_t, 0)\right]$$

**Expected Negative Exposure (ENE)**:

$$\text{ENE}(t) = \mathbb{E}\left[\max(-V_t, 0)\right]$$

### Credit Valuation Adjustment (CVA)

$$\text{CVA} = \sum_{i=1}^{n} \text{DF}(t_i) \cdot \text{EPE}(t_i) \cdot \text{LGD} \cdot \Delta\text{PD}_{\text{cpty}}(t_i)$$

### Funding Valuation Adjustment (FVA)

$$\text{FVA} = \sum_{i=1}^{n} \text{DF}(t_i) \cdot \text{EPE}(t_i) \cdot s_f \cdot \Delta t$$

### SA-CCR Exposure at Default

$$\text{EAD} = \alpha \times (\text{RC} + \text{PFE})$$

where $\alpha = 1.4$ and $\text{PFE} = \text{multiplier} \times \text{AddOn}$.

---

## 🏗️ Architecture

```
xva_core/
├── config/       # YAML loading + Pydantic validation
├── market/       # Curves, OU/GBM models, correlations
├── instruments/  # IRS, FX Forward pricing
├── exposure/     # Monte Carlo, netting, EPE/ENE
├── collateral/   # VM (with MPR), IM
├── xva/          # CVA, DVA, FVA, MVA, KVA calculators
├── reg/          # SA-CCR approximations
└── reporting/    # Plots, tables, exports

xva_app/          # Streamlit application
notebooks/        # Educational Jupyter notebooks
tests/            # Comprehensive test suite
```

---

## 📊 Example Output

### Exposure Profiles

The engine generates exposure profiles showing how EPE and ENE evolve over the trade lifecycle, both with and without collateral:

| Metric | Uncollateralized | Collateralized |
|--------|------------------|----------------|
| Peak EPE | $2.5M | $1.2M |
| Peak ENE | $1.8M | $0.9M |
| Avg IM | - | $1.5M |

### xVA Breakdown

| Component | Value ($M) | bps of Notional |
|-----------|------------|-----------------|
| CVA | 1.23 | 12.3 |
| DVA (benefit) | -0.45 | -4.5 |
| FVA | 0.67 | 6.7 |
| MVA | 0.34 | 3.4 |
| KVA | 0.89 | 8.9 |
| **Total xVA** | **2.68** | **26.8** |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.11+, NumPy, Pandas, SciPy |
| **Validation** | Pydantic v2 |
| **Visualization** | Matplotlib, Plotly |
| **UI** | Streamlit |
| **Testing** | pytest, pytest-cov |
| **Linting** | Ruff, Black |
| **Type Checking** | mypy |
| **CI/CD** | GitHub Actions |

---

## 📁 Project Structure

```
xva-project/
├── pyproject.toml          # Modern packaging (PEP-621)
├── README.md               # This file
├── LICENSE                 # MIT License
├── data/                   # Configuration files
│   ├── curves.yaml         # Market data
│   ├── hazards.yaml        # Credit curves
│   └── portfolio.yaml      # Sample trades
├── xva_core/               # Main package
├── xva_app/                # Streamlit application
├── notebooks/              # Jupyter notebooks
├── examples/               # Demo scripts
└── tests/                  # Test suite
```

---

## 📚 Documentation

### Notebooks

1. **01_Introduction.ipynb** - Project overview and motivation
2. **02_Market_Models.ipynb** - OU and GBM model deep dive
3. **03_Exposure_Calc.ipynb** - EPE/ENE derivation and implementation
4. **04_xVA_Formulas.ipynb** - CVA/DVA/FVA/MVA/KVA theory
5. **05_Full_Demo.ipynb** - End-to-end example

### Configuration

Market data and portfolio are configured via YAML files in `data/`:

```yaml
# data/curves.yaml
market:
  ois_rate: 0.020
  domestic_rate_model:
    kappa: 0.10
    theta: 0.020
    sigma: 0.010
```

---

## 🧪 Testing

The project includes comprehensive tests covering:

- **Unit tests**: Deterministic checks (zero-vol convergence, known formulas)
- **Integration tests**: Stochastic with Monte Carlo tolerances
- **Edge cases**: Expired trades, exposure sign flips, extreme parameters

```bash
# Run all tests with coverage
pytest --cov=xva_core --cov-report=html

# Run specific test file
pytest tests/test_xva.py -v

# Run tests matching pattern
pytest -k "test_cva" -v
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Based on academic literature and industry practice:

- Gregory, J. (2015). *The xVA Challenge: Counterparty Credit Risk, Funding, Collateral, and Capital*
- Pykhtin, M. & Zhu, S. (2007). *A Guide to Modeling Counterparty Credit Risk*
- BCBS (2014). *The Standardised Approach for Measuring Counterparty Credit Risk Exposures*

---

## 📧 Contact

**Author**: Lucas
**Project**: Final-year Financial Engineering Portfolio
**Focus**: Counterparty Credit Risk, xVA, Regulatory Capital

---

*Built with ❤️ for financial engineering education*
