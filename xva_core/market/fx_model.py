"""
Geometric Brownian Motion FX model for currency simulation.

The GBM model captures the log-normal dynamics of exchange rates
under the domestic risk-neutral measure.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray, PathArray


@dataclass
class GBMFXModel:
    """
    Geometric Brownian Motion model for FX rates.

    Under the domestic risk-neutral measure:
        dS/S = (r_d - r_f) dt + σ dW

    where:
        r_d = domestic short rate
        r_f = foreign short rate
        σ = FX volatility

    Attributes
    ----------
    S0 : float
        Initial FX spot rate (domestic per foreign, e.g., 1.10 USD/EUR)
    sigma : float
        FX volatility (e.g., 0.12 for 12%)

    Example
    -------
    >>> model = GBMFXModel(S0=1.10, sigma=0.12)
    >>> time_grid = np.linspace(0, 5, 21)
    >>> r_d = np.full((1000, 21), 0.02)  # Flat rates for simplicity
    >>> r_f = np.full((1000, 21), 0.015)
    >>> spots = model.simulate(r_d, r_f, time_grid, seed=42)
    """

    S0: float = 1.10
    sigma: float = 0.12

    def __post_init__(self) -> None:
        """Validate model parameters."""
        if self.S0 <= 0:
            raise ValueError(f"Initial spot must be positive, got {self.S0}")
        if self.sigma < 0:
            raise ValueError(f"Volatility must be non-negative, got {self.sigma}")

    def simulate(
        self,
        r_domestic: PathArray,
        r_foreign: PathArray,
        time_grid: FloatArray,
        seed: int | None = None,
        z_correlated: PathArray | None = None,
    ) -> PathArray:
        """
        Simulate FX spot paths using log-normal dynamics.

        Parameters
        ----------
        r_domestic : PathArray
            Domestic short rate paths (n_paths, n_steps)
        r_foreign : PathArray
            Foreign short rate paths (n_paths, n_steps)
        time_grid : FloatArray
            Time points in years
        seed : int | None
            Random seed (ignored if z_correlated provided)
        z_correlated : PathArray | None
            Pre-generated correlated random numbers.
            If None, generates uncorrelated randoms.

        Returns
        -------
        PathArray
            FX spot paths of shape (n_paths, n_steps)

        Notes
        -----
        Uses log-Euler discretization:
            S(t+dt) = S(t) * exp[(r_d - r_f - 0.5σ²)dt + σ√dt * Z]
        """
        n_paths, n_steps = r_domestic.shape

        if z_correlated is not None:
            z = z_correlated
        else:
            if seed is not None:
                np.random.seed(seed)
            z = np.random.standard_normal((n_paths, n_steps - 1))

        # Initialize
        spots = np.zeros((n_paths, n_steps))
        spots[:, 0] = self.S0

        # Log-Euler simulation
        for i in range(n_steps - 1):
            dt = time_grid[i + 1] - time_grid[i]
            sqrt_dt = np.sqrt(dt)

            # Drift: (r_d - r_f - 0.5 * sigma^2)
            drift = (r_domestic[:, i] - r_foreign[:, i] - 0.5 * self.sigma**2) * dt

            # Diffusion
            diffusion = self.sigma * sqrt_dt * z[:, i]

            # Log-normal evolution
            spots[:, i + 1] = spots[:, i] * np.exp(drift + diffusion)

        return spots

    def forward_rate(
        self,
        t: float,
        T: float,
        r_d: float,
        r_f: float,
    ) -> float:
        """
        Calculate FX forward rate at time 0 for delivery at T.

        Parameters
        ----------
        t : float
            Current time
        T : float
            Delivery time
        r_d : float
            Domestic rate (continuously compounded)
        r_f : float
            Foreign rate (continuously compounded)

        Returns
        -------
        float
            Forward FX rate F(t, T)

        Notes
        -----
        F(0, T) = S(0) * exp((r_d - r_f) * T)
        """
        tau = T - t
        return self.S0 * np.exp((r_d - r_f) * tau)

    @classmethod
    def from_config(cls, config: "FXModelConfig") -> "GBMFXModel":  # type: ignore[name-defined]
        """
        Create model from configuration object.

        Parameters
        ----------
        config : FXModelConfig
            Configuration with model parameters

        Returns
        -------
        GBMFXModel
            Initialized model
        """
        return cls(
            S0=config.initial_spot,
            sigma=config.volatility,
        )
