"""
xVA Calculation Engine - Core Package.

A production-grade educational implementation of xVA (CVA/DVA/FVA/MVA/KVA)
calculations for counterparty credit risk, featuring Monte Carlo simulation,
collateral management, and regulatory capital (SA-CCR).

Example
-------
>>> from xva_core import MonteCarloEngine, IRSwap, CVACalculator
>>> engine = MonteCarloEngine(n_paths=5000, horizon=5.0)
>>> swap = IRSwap(notional=1e7, fixed_rate=0.02, maturity=5.0)
>>> exposure = engine.simulate([swap])
>>> cva = CVACalculator().calculate(exposure, hazard_rate=0.01, lgd=0.6)
"""

__version__ = "1.0.0"
__author__ = "Lucas"

# Core types
from xva_core._types import FloatArray, IntArray, PathArray

# Configuration
from xva_core.config import MarketConfig, PortfolioConfig, SimulationConfig, load_config

# Market models
from xva_core.market import (
    CholeskyCorrelation,
    DiscountCurve,
    GBMFXModel,
    HazardCurve,
    OUShortRateModel,
)

# Instruments
from xva_core.instruments import FXForward, IRSwap, Instrument

# Exposure
from xva_core.exposure import (
    ExposureMetrics,
    MonteCarloEngine,
    NettingSet,
    calculate_ene,
    calculate_epe,
    calculate_pfe,
)

# Collateral
from xva_core.collateral import InitialMargin, VariationMargin

# xVA calculations
from xva_core.xva import (
    CVACalculator,
    DVACalculator,
    FVACalculator,
    KVACalculator,
    MVACalculator,
    XVAResult,
)

# Regulatory
from xva_core.reg import SACCRCalculator

# Reporting
from xva_core.reporting import (
    create_excel_report,
    create_exposure_plot,
    create_xva_breakdown_table,
    export_to_csv,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "FloatArray",
    "IntArray",
    "PathArray",
    # Config
    "MarketConfig",
    "PortfolioConfig",
    "SimulationConfig",
    "load_config",
    # Market
    "DiscountCurve",
    "HazardCurve",
    "OUShortRateModel",
    "GBMFXModel",
    "CholeskyCorrelation",
    # Instruments
    "Instrument",
    "IRSwap",
    "FXForward",
    # Exposure
    "MonteCarloEngine",
    "NettingSet",
    "ExposureMetrics",
    "calculate_epe",
    "calculate_ene",
    "calculate_pfe",
    # Collateral
    "VariationMargin",
    "InitialMargin",
    # xVA
    "CVACalculator",
    "DVACalculator",
    "FVACalculator",
    "MVACalculator",
    "KVACalculator",
    "XVAResult",
    # Regulatory
    "SACCRCalculator",
    # Reporting
    "create_exposure_plot",
    "create_xva_breakdown_table",
    "export_to_csv",
    "create_excel_report",
]
