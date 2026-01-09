"""
Ornstein-Uhlenbeck short-rate model for interest rate simulation.

The OU process is a mean-reverting stochastic process widely used
for modeling short-term interest rates.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray, PathArray


@dataclass
class OUShortRateModel:
    """
    Ornstein-Uhlenbeck (Vasicek) short-rate model.

    The short rate follows the SDE:
        dr = κ(θ - r) dt + σ dW

    where:
        κ = mean reversion speed
        θ = long-term mean rate
        σ = volatility
        r₀ = initial rate

    Attributes
    ----------
    kappa : float
        Mean reversion speed (higher = faster reversion)
    theta : float
        Long-term mean rate
    sigma : float
        Volatility of the short rate
    r0 : float
        Initial short rate at t=0

    Example
    -------
    >>> model = OUShortRateModel(kappa=0.1, theta=0.02, sigma=0.01, r0=0.02)
    >>> time_grid = np.linspace(0, 5, 21)  # 5Y quarterly
    >>> paths = model.simulate(n_paths=5000, time_grid=time_grid, seed=42)
    >>> print(f"Shape: {paths.shape}")  # (5000, 21)
    """

    kappa: float = 0.1
    theta: float = 0.02
    sigma: float = 0.01
    r0: float = 0.02

    def __post_init__(self) -> None:
        """Validate model parameters."""
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")

    def expected_rate(self, t: float) -> float:
        """
        Calculate expected rate at time t.

        Parameters
        ----------
        t : float
            Time in years

        Returns
        -------
        float
            E[r(t)] = θ + (r₀ - θ) * exp(-κt)
        """
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t: float) -> float:
        """
        Calculate variance of rate at time t.

        Parameters
        ----------
        t : float
            Time in years

        Returns
        -------
        float
            Var[r(t)] = (σ² / 2κ) * (1 - exp(-2κt))
        """
        return (self.sigma**2 / (2 * self.kappa)) * (
            1 - np.exp(-2 * self.kappa * t)
        )

    def long_term_variance(self) -> float:
        """
        Calculate long-term (stationary) variance.

        Returns
        -------
        float
            Var[r(∞)] = σ² / (2κ)
        """
        return self.sigma**2 / (2 * self.kappa)

    def simulate(
        self,
        n_paths: int,
        time_grid: FloatArray,
        seed: int | None = None,
        antithetic: bool = False,
    ) -> PathArray:
        """
        Simulate short rate paths using Euler-Maruyama discretization.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths
        time_grid : FloatArray
            Array of time points in years
        seed : int | None
            Random seed for reproducibility
        antithetic : bool
            If True, use antithetic variates (n_paths must be even)

        Returns
        -------
        PathArray
            Array of shape (n_paths, len(time_grid)) with simulated rates

        Notes
        -----
        Uses Euler-Maruyama discretization:
            r(t+dt) = r(t) + κ(θ - r(t))dt + σ√dt * Z
        where Z ~ N(0,1)

        Example
        -------
        >>> model = OUShortRateModel(kappa=0.1, theta=0.02, sigma=0.01, r0=0.015)
        >>> grid = np.linspace(0, 5, 21)
        >>> paths = model.simulate(5000, grid, seed=42)
        >>> print(f"Mean 5Y rate: {paths[:, -1].mean():.4f}")
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = len(time_grid)

        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError(
                    f"n_paths must be even for antithetic variates, got {n_paths}"
                )
            n_half = n_paths // 2
            z_half = np.random.standard_normal((n_half, n_steps - 1))
            z = np.vstack([z_half, -z_half])
        else:
            z = np.random.standard_normal((n_paths, n_steps - 1))

        # Initialize paths
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = self.r0

        # Euler-Maruyama simulation
        for i in range(n_steps - 1):
            dt = time_grid[i + 1] - time_grid[i]
            sqrt_dt = np.sqrt(dt)

            # dr = κ(θ - r)dt + σ√dt * Z
            drift = self.kappa * (self.theta - paths[:, i]) * dt
            diffusion = self.sigma * sqrt_dt * z[:, i]

            paths[:, i + 1] = paths[:, i] + drift + diffusion

        return paths

    def simulate_exact(
        self,
        n_paths: int,
        time_grid: FloatArray,
        seed: int | None = None,
    ) -> PathArray:
        """
        Simulate using exact transition density (more accurate).

        The OU process has a known Gaussian transition density,
        allowing exact sampling without discretization error.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths
        time_grid : FloatArray
            Array of time points in years
        seed : int | None
            Random seed

        Returns
        -------
        PathArray
            Simulated rate paths

        Notes
        -----
        r(t+dt) | r(t) ~ N(μ, σ²) where:
            μ = θ + (r(t) - θ) * exp(-κ*dt)
            σ² = (σ² / 2κ) * (1 - exp(-2κ*dt))
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = len(time_grid)
        z = np.random.standard_normal((n_paths, n_steps - 1))

        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = self.r0

        for i in range(n_steps - 1):
            dt = time_grid[i + 1] - time_grid[i]

            # Exact mean and variance
            exp_kdt = np.exp(-self.kappa * dt)
            mean = self.theta + (paths[:, i] - self.theta) * exp_kdt
            var = (self.sigma**2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa * dt))
            std = np.sqrt(var)

            paths[:, i + 1] = mean + std * z[:, i]

        return paths

    @classmethod
    def from_config(cls, config: "OUModelConfig") -> "OUShortRateModel":  # type: ignore[name-defined]
        """
        Create model from configuration object.

        Parameters
        ----------
        config : OUModelConfig
            Configuration with model parameters

        Returns
        -------
        OUShortRateModel
            Initialized model
        """
        return cls(
            kappa=config.kappa,
            theta=config.theta,
            sigma=config.sigma,
            r0=config.initial_rate,
        )
