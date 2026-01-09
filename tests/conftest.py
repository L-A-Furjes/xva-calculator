"""
Pytest fixtures for xVA testing.

Provides reusable test fixtures for curves, models, instruments, and simulation results.
"""

import numpy as np
import pytest

from xva_core.config.models import (
    CorrelationConfig,
    FXModelConfig,
    MarketConfig,
    OUModelConfig,
)
from xva_core.instruments import FXForward, IRSwap
from xva_core.market import (
    CholeskyCorrelation,
    DiscountCurve,
    GBMFXModel,
    HazardCurve,
    OUShortRateModel,
)


@pytest.fixture
def time_grid() -> np.ndarray:
    """Standard 5-year quarterly time grid."""
    return np.linspace(0, 5, 21)


@pytest.fixture
def flat_discount_curve() -> DiscountCurve:
    """Flat 2% discount curve."""
    return DiscountCurve(rate=0.02)


@pytest.fixture
def hazard_curve() -> HazardCurve:
    """Standard hazard curve with 120bps hazard rate."""
    return HazardCurve(hazard_rate=0.012, recovery_rate=0.4)


@pytest.fixture
def ou_model() -> OUShortRateModel:
    """Standard OU short-rate model."""
    return OUShortRateModel(kappa=0.1, theta=0.02, sigma=0.01, r0=0.02)


@pytest.fixture
def fx_model() -> GBMFXModel:
    """Standard GBM FX model."""
    return GBMFXModel(S0=1.10, sigma=0.12)


@pytest.fixture
def correlation() -> CholeskyCorrelation:
    """Standard correlation structure."""
    return CholeskyCorrelation(rho_df=0.7, rho_dx=-0.3, rho_fx=0.4)


@pytest.fixture
def sample_swap() -> IRSwap:
    """Sample 5Y payer swap."""
    return IRSwap(
        notional=10_000_000,
        fixed_rate=0.02,
        maturity=5.0,
        pay_fixed=True,
        payment_freq=0.5,
    )


@pytest.fixture
def sample_receiver_swap() -> IRSwap:
    """Sample 3Y receiver swap."""
    return IRSwap(
        notional=15_000_000,
        fixed_rate=0.025,
        maturity=3.0,
        pay_fixed=False,
        payment_freq=0.5,
    )


@pytest.fixture
def sample_fx_forward() -> FXForward:
    """Sample 1Y FX forward."""
    return FXForward(
        notional_foreign=5_000_000,
        strike=1.10,
        maturity=1.0,
        buy_foreign=True,
    )


@pytest.fixture
def market_config() -> MarketConfig:
    """Standard market configuration."""
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


@pytest.fixture
def sample_exposure() -> np.ndarray:
    """Sample exposure matrix for testing."""
    np.random.seed(42)
    n_paths, n_steps = 1000, 21
    # Generate exposure that starts near zero and fans out
    t = np.linspace(0, 5, n_steps)
    base = np.sin(t) * 1e6  # Base profile
    noise = np.random.randn(n_paths, n_steps) * 5e5
    return base + noise


@pytest.fixture
def sample_discount_factors(time_grid: np.ndarray) -> np.ndarray:
    """Sample discount factors."""
    return np.exp(-0.02 * time_grid)
