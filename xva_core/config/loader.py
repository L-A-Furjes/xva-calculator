"""
YAML configuration loading utilities.

Provides functions to load and validate configuration from YAML files,
returning properly typed Pydantic model instances.
"""

from pathlib import Path
from typing import Any

import yaml

from xva_core.config.models import (
    CollateralConfig,
    CreditConfig,
    MarketConfig,
    PortfolioConfig,
    SimulationConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path
        Path to the YAML file

    Returns
    -------
    dict[str, Any]
        Parsed YAML contents

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    yaml.YAMLError
        If the file contains invalid YAML
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_market_config(path: Path | str) -> MarketConfig:
    """
    Load market configuration from a YAML file.

    Parameters
    ----------
    path : Path | str
        Path to the market configuration YAML file

    Returns
    -------
    MarketConfig
        Validated market configuration

    Example
    -------
    >>> config = load_market_config("data/curves.yaml")
    >>> print(config.ois_rate)
    0.02
    """
    path = Path(path)
    data = _load_yaml(path)

    # Handle nested 'market' key if present
    if "market" in data:
        data = data["market"]

    return MarketConfig(**data)


def load_portfolio_config(path: Path | str) -> PortfolioConfig:
    """
    Load portfolio configuration from a YAML file.

    Parameters
    ----------
    path : Path | str
        Path to the portfolio configuration YAML file

    Returns
    -------
    PortfolioConfig
        Validated portfolio configuration

    Example
    -------
    >>> portfolio = load_portfolio_config("data/portfolio.yaml")
    >>> print(f"Loaded {portfolio.n_trades} trades")
    """
    path = Path(path)
    data = _load_yaml(path)

    # Handle nested 'portfolio' key if present
    if "portfolio" in data:
        data = data["portfolio"]

    return PortfolioConfig(**data)


def load_config(
    market_path: Path | str | None = None,
    portfolio_path: Path | str | None = None,
    simulation_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Load complete configuration from multiple YAML files.

    Parameters
    ----------
    market_path : Path | str | None
        Path to market configuration file
    portfolio_path : Path | str | None
        Path to portfolio configuration file
    simulation_path : Path | str | None
        Path to simulation configuration file (optional)

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'market': MarketConfig (if market_path provided)
        - 'portfolio': PortfolioConfig (if portfolio_path provided)
        - 'simulation': SimulationConfig (if simulation_path provided)
        - 'collateral': CollateralConfig (from simulation file)
        - 'credit': CreditConfig (from simulation file)

    Example
    -------
    >>> config = load_config(
    ...     market_path="data/curves.yaml",
    ...     portfolio_path="data/portfolio.yaml"
    ... )
    >>> print(config['market'].ois_rate)
    """
    result: dict[str, Any] = {}

    if market_path is not None:
        result["market"] = load_market_config(market_path)

    if portfolio_path is not None:
        result["portfolio"] = load_portfolio_config(portfolio_path)

    if simulation_path is not None:
        path = Path(simulation_path)
        data = _load_yaml(path)

        if "simulation" in data:
            result["simulation"] = SimulationConfig(**data["simulation"])
        if "collateral" in data:
            result["collateral"] = CollateralConfig(**data["collateral"])
        if "credit" in data:
            result["credit"] = CreditConfig(**data["credit"])

    return result


def create_default_market_config() -> MarketConfig:
    """
    Create a default market configuration with typical values.

    Returns
    -------
    MarketConfig
        Default market configuration suitable for testing
    """
    from xva_core.config.models import (
        CorrelationConfig,
        FXModelConfig,
        OUModelConfig,
    )

    return MarketConfig(
        ois_rate=0.02,
        funding_spread_bps=100,
        cost_of_capital=0.10,
        domestic_rate_model=OUModelConfig(
            kappa=0.10,
            theta=0.02,
            sigma=0.01,
            initial_rate=0.02,
        ),
        foreign_rate_model=OUModelConfig(
            kappa=0.08,
            theta=0.015,
            sigma=0.012,
            initial_rate=0.015,
        ),
        fx_model=FXModelConfig(
            initial_spot=1.10,
            volatility=0.12,
        ),
        correlations=CorrelationConfig(
            domestic_foreign=0.7,
            domestic_fx=-0.3,
            foreign_fx=0.4,
        ),
    )
