"""
Pydantic configuration models for xVA calculation engine.

These models provide validation and type-safe configuration for:
- Market data (curves, volatilities, correlations)
- Simulation parameters (Monte Carlo settings)
- Portfolio specifications (trade definitions)
- Collateral and credit parameters
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class OUModelConfig(BaseModel):
    """
    Ornstein-Uhlenbeck short-rate model parameters.

    The OU process follows: dr = kappa * (theta - r) * dt + sigma * dW

    Attributes
    ----------
    kappa : float
        Mean reversion speed (typically 0.01 to 1.0)
    theta : float
        Long-term mean rate (e.g., 0.02 for 2%)
    sigma : float
        Volatility (e.g., 0.01 for 100bps)
    initial_rate : float
        Starting short rate

    Example
    -------
    >>> config = OUModelConfig(kappa=0.1, theta=0.02, sigma=0.01, initial_rate=0.02)
    """

    kappa: float = Field(gt=0, le=2.0, description="Mean reversion speed")
    theta: float = Field(ge=-0.02, le=0.20, description="Long-term mean rate")
    sigma: float = Field(ge=0, le=0.10, description="Volatility")
    initial_rate: float = Field(ge=-0.02, le=0.20, description="Starting short rate")

    @field_validator("kappa")
    @classmethod
    def kappa_realistic(cls, v: float) -> float:
        """Validate mean reversion speed is realistic."""
        if v > 1.0:
            import warnings

            warnings.warn(
                f"Mean reversion speed {v} > 1.0 is aggressive; "
                "typical values are 0.01-0.5",
                UserWarning,
                stacklevel=2,
            )
        return v


class FXModelConfig(BaseModel):
    """
    GBM FX model parameters.

    The FX rate follows: dS/S = (r_d - r_f) * dt + sigma * dW

    Attributes
    ----------
    initial_spot : float
        Starting FX spot rate (e.g., 1.10 for EUR/USD)
    volatility : float
        FX volatility (e.g., 0.12 for 12%)
    """

    initial_spot: float = Field(gt=0, description="Initial FX spot rate")
    volatility: float = Field(ge=0, le=1.0, description="FX volatility")


class CorrelationConfig(BaseModel):
    """
    Correlation parameters for multi-asset simulation.

    Attributes
    ----------
    domestic_foreign : float
        Correlation between domestic and foreign rates
    domestic_fx : float
        Correlation between domestic rate and FX
    foreign_fx : float
        Correlation between foreign rate and FX
    """

    domestic_foreign: float = Field(ge=-1, le=1, default=0.7)
    domestic_fx: float = Field(ge=-1, le=1, default=-0.3)
    foreign_fx: float = Field(ge=-1, le=1, default=0.4)

    @model_validator(mode="after")
    def validate_positive_definite(self) -> "CorrelationConfig":
        """Ensure correlation matrix is positive semi-definite."""
        import numpy as np

        corr_matrix = np.array(
            [
                [1.0, self.domestic_foreign, self.domestic_fx],
                [self.domestic_foreign, 1.0, self.foreign_fx],
                [self.domestic_fx, self.foreign_fx, 1.0],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"Correlation matrix is not positive semi-definite. "
                f"Eigenvalues: {eigenvalues}"
            )
        return self


class MarketConfig(BaseModel):
    """
    Complete market configuration.

    Attributes
    ----------
    ois_rate : float
        Flat OIS discount rate
    funding_spread_bps : float
        Funding spread over OIS in basis points
    cost_of_capital : float
        Cost of capital for KVA (e.g., 0.10 for 10%)
    domestic_rate_model : OUModelConfig
        Domestic short-rate model parameters
    foreign_rate_model : OUModelConfig
        Foreign short-rate model parameters
    fx_model : FXModelConfig
        FX model parameters
    correlations : CorrelationConfig
        Cross-asset correlations
    """

    ois_rate: float = Field(ge=-0.02, le=0.20, default=0.02)
    funding_spread_bps: float = Field(ge=0, le=500, default=100)
    cost_of_capital: float = Field(ge=0, le=0.30, default=0.10)
    domestic_rate_model: OUModelConfig
    foreign_rate_model: OUModelConfig
    fx_model: FXModelConfig
    correlations: CorrelationConfig = Field(default_factory=CorrelationConfig)

    @property
    def funding_spread(self) -> float:
        """Funding spread as decimal."""
        return self.funding_spread_bps / 10000


class CollateralConfig(BaseModel):
    """
    Collateral agreement parameters.

    Attributes
    ----------
    threshold : float
        Collateral threshold in base currency (e.g., 1e6 for $1M)
    mta : float
        Minimum transfer amount (e.g., 1e5 for $100K)
    mpr_days : int
        Margin period of risk in business days
    im_multiplier : float
        Initial margin as multiple of EPE
    """

    threshold: float = Field(ge=0, default=1e6)
    mta: float = Field(ge=0, default=1e5)
    mpr_days: int = Field(ge=0, le=30, default=10)
    im_multiplier: float = Field(ge=1.0, le=5.0, default=1.5)

    @model_validator(mode="after")
    def mta_less_than_threshold(self) -> "CollateralConfig":
        """Validate MTA is less than threshold."""
        if self.mta > self.threshold > 0:
            raise ValueError(
                f"MTA ({self.mta:,.0f}) cannot exceed threshold ({self.threshold:,.0f})"
            )
        return self


class CreditConfig(BaseModel):
    """
    Credit parameters for CVA/DVA calculations.

    Attributes
    ----------
    lgd_counterparty : float
        Loss given default for counterparty (0-1)
    lgd_own : float
        Loss given default for own default (0-1)
    hazard_rate_counterparty_bps : float
        Counterparty hazard rate in basis points (per annum)
    hazard_rate_own_bps : float
        Own hazard rate in basis points (per annum)
    """

    lgd_counterparty: float = Field(ge=0, le=1, default=0.60)
    lgd_own: float = Field(ge=0, le=1, default=0.60)
    hazard_rate_counterparty_bps: float = Field(ge=0, le=1000, default=120)
    hazard_rate_own_bps: float = Field(ge=0, le=1000, default=100)

    @property
    def hazard_rate_counterparty(self) -> float:
        """Counterparty hazard rate as decimal."""
        return self.hazard_rate_counterparty_bps / 10000

    @property
    def hazard_rate_own(self) -> float:
        """Own hazard rate as decimal."""
        return self.hazard_rate_own_bps / 10000


class SimulationConfig(BaseModel):
    """
    Monte Carlo simulation parameters.

    Attributes
    ----------
    n_paths : int
        Number of Monte Carlo paths
    horizon_years : float
        Simulation horizon in years
    time_step : str
        Time step frequency ('monthly' or 'quarterly')
    seed : int | None
        Random seed for reproducibility
    """

    n_paths: int = Field(ge=100, le=100000, default=5000)
    horizon_years: float = Field(ge=0.25, le=30, default=5.0)
    time_step: Literal["monthly", "quarterly"] = "quarterly"
    seed: int | None = Field(default=42)

    @property
    def dt(self) -> float:
        """Time step in years."""
        return 1 / 12 if self.time_step == "monthly" else 0.25

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.horizon_years / self.dt) + 1


class IRSwapConfig(BaseModel):
    """
    Interest rate swap configuration.

    Attributes
    ----------
    notional : float
        Notional amount in base currency
    fixed_rate : float
        Fixed rate (decimal, e.g., 0.02 for 2%)
    maturity_years : float
        Time to maturity in years
    pay_fixed : bool
        True for payer swap (pay fixed, receive float)
    payment_frequency : float
        Payment frequency in years (e.g., 0.5 for semi-annual)
    """

    notional: float = Field(gt=0)
    fixed_rate: float = Field(ge=-0.02, le=0.20)
    maturity_years: float = Field(gt=0, le=50)
    pay_fixed: bool = True
    payment_frequency: float = Field(gt=0, le=1, default=0.5)


class FXForwardConfig(BaseModel):
    """
    FX forward configuration.

    Attributes
    ----------
    notional_foreign : float
        Notional in foreign currency
    strike : float
        Forward strike rate (domestic per foreign)
    maturity_years : float
        Time to maturity in years
    buy_foreign : bool
        True if buying foreign currency
    """

    notional_foreign: float = Field(gt=0)
    strike: float = Field(gt=0)
    maturity_years: float = Field(gt=0, le=30)
    buy_foreign: bool = True


class PortfolioConfig(BaseModel):
    """
    Portfolio configuration with multiple trades.

    Attributes
    ----------
    irs_trades : list[IRSwapConfig]
        List of interest rate swap configurations
    fxf_trades : list[FXForwardConfig]
        List of FX forward configurations
    """

    irs_trades: list[IRSwapConfig] = Field(default_factory=list)
    fxf_trades: list[FXForwardConfig] = Field(default_factory=list)

    @property
    def total_notional(self) -> float:
        """Total notional across all trades."""
        irs_notional = sum(t.notional for t in self.irs_trades)
        fxf_notional = sum(t.notional_foreign for t in self.fxf_trades)
        return irs_notional + fxf_notional

    @property
    def n_trades(self) -> int:
        """Total number of trades."""
        return len(self.irs_trades) + len(self.fxf_trades)
