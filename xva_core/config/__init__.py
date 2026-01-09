"""
Configuration module for xVA calculation engine.

Provides Pydantic-validated configuration models and YAML loading utilities
for market data, simulation parameters, and portfolio specifications.
"""

from xva_core.config.loader import load_config, load_market_config, load_portfolio_config
from xva_core.config.models import (
    CollateralConfig,
    CorrelationConfig,
    CreditConfig,
    FXForwardConfig,
    FXModelConfig,
    IRSwapConfig,
    MarketConfig,
    OUModelConfig,
    PortfolioConfig,
    SimulationConfig,
)

__all__ = [
    # Models
    "OUModelConfig",
    "FXModelConfig",
    "CorrelationConfig",
    "MarketConfig",
    "CollateralConfig",
    "CreditConfig",
    "SimulationConfig",
    "IRSwapConfig",
    "FXForwardConfig",
    "PortfolioConfig",
    # Loaders
    "load_config",
    "load_market_config",
    "load_portfolio_config",
]
