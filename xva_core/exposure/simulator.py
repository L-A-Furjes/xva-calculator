"""
Monte Carlo simulation engine for exposure calculation.

Coordinates the joint simulation of interest rates and FX rates,
then computes instrument MTMs along each path.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from xva_core._types import FloatArray, PathArray
from xva_core.config.models import MarketConfig, SimulationConfig
from xva_core.instruments.base import Instrument
from xva_core.market.correlation import CholeskyCorrelation
from xva_core.market.curve import build_discount_factors_from_path
from xva_core.market.fx_model import GBMFXModel
from xva_core.market.ir_model import OUShortRateModel


@dataclass
class SimulationResult:
    """
    Container for Monte Carlo simulation results.

    Attributes
    ----------
    time_grid : FloatArray
        Time points in years, shape (n_steps,)
    r_domestic : PathArray
        Domestic short rate paths, shape (n_paths, n_steps)
    r_foreign : PathArray
        Foreign short rate paths, shape (n_paths, n_steps)
    fx_spot : PathArray
        FX spot paths, shape (n_paths, n_steps)
    df_domestic : PathArray
        Cumulative domestic discount factors, shape (n_paths, n_steps)
    df_foreign : PathArray
        Cumulative foreign discount factors, shape (n_paths, n_steps)
    mtm : PathArray
        Portfolio MTM paths, shape (n_paths, n_steps)
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int
        Number of time steps
    """

    time_grid: FloatArray
    r_domestic: PathArray
    r_foreign: PathArray
    fx_spot: PathArray
    df_domestic: PathArray
    df_foreign: PathArray
    mtm: PathArray
    n_paths: int = field(init=False)
    n_steps: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived attributes."""
        self.n_paths, self.n_steps = self.mtm.shape

    @property
    def paths_data(self) -> dict[str, PathArray]:
        """Return paths as dictionary for instrument pricing."""
        return {
            "r_domestic": self.r_domestic,
            "r_foreign": self.r_foreign,
            "fx_spot": self.fx_spot,
            "df_domestic": self.df_domestic,
            "df_foreign": self.df_foreign,
        }


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for xVA exposure calculation.

    Simulates correlated paths for domestic rates, foreign rates,
    and FX spot, then calculates portfolio MTM at each time step.

    Attributes
    ----------
    n_paths : int
        Number of simulation paths
    horizon : float
        Simulation horizon in years
    dt : float
        Time step size in years
    seed : int | None
        Random seed for reproducibility

    Example
    -------
    >>> from xva_core import MonteCarloEngine, IRSwap
    >>> engine = MonteCarloEngine(n_paths=5000, horizon=5.0)
    >>> swap = IRSwap(notional=1e7, fixed_rate=0.02, maturity=5.0)
    >>> result = engine.simulate(
    ...     instruments=[swap],
    ...     market_config=market_config
    ... )
    >>> print(f"Peak EPE: {result.mtm.clip(0).mean(axis=0).max():,.0f}")
    """

    def __init__(
        self,
        n_paths: int = 5000,
        horizon: float = 5.0,
        dt: float = 0.25,
        seed: int | None = 42,
    ) -> None:
        """
        Initialize Monte Carlo engine.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths (default 5000)
        horizon : float
            Simulation horizon in years (default 5.0)
        dt : float
            Time step in years (default 0.25 = quarterly)
        seed : int | None
            Random seed (default 42)
        """
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if dt <= 0 or dt > horizon:
            raise ValueError(f"dt must be in (0, horizon], got {dt}")

        self.n_paths = n_paths
        self.horizon = horizon
        self.dt = dt
        self.seed = seed

        # Build time grid
        self._n_steps = int(horizon / dt) + 1
        self._time_grid = np.linspace(0, horizon, self._n_steps)

    @property
    def time_grid(self) -> FloatArray:
        """Return the simulation time grid."""
        return self._time_grid.copy()

    @property
    def n_steps(self) -> int:
        """Number of time steps including t=0."""
        return self._n_steps

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "MonteCarloEngine":
        """
        Create engine from configuration.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration

        Returns
        -------
        MonteCarloEngine
            Configured engine
        """
        return cls(
            n_paths=config.n_paths,
            horizon=config.horizon_years,
            dt=config.dt,
            seed=config.seed,
        )

    def simulate(
        self,
        instruments: Sequence[Instrument],
        market_config: MarketConfig,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a portfolio of instruments.

        Parameters
        ----------
        instruments : Sequence[Instrument]
            List of instruments to price
        market_config : MarketConfig
            Market configuration with model parameters

        Returns
        -------
        SimulationResult
            Complete simulation results including paths and MTMs

        Example
        -------
        >>> result = engine.simulate([swap1, swap2, fxf1], market_config)
        >>> epe = result.mtm.clip(0).mean(axis=0)
        """
        # Initialize models
        ir_dom = OUShortRateModel.from_config(market_config.domestic_rate_model)
        ir_for = OUShortRateModel.from_config(market_config.foreign_rate_model)
        fx_model = GBMFXModel.from_config(market_config.fx_model)
        corr = CholeskyCorrelation.from_config(market_config.correlations)

        # Generate correlated random numbers
        z_dom, z_for, z_fx = corr.generate_correlated_samples(
            n_paths=self.n_paths,
            n_steps=self._n_steps - 1,  # n_steps - 1 innovations
            seed=self.seed,
        )

        # Simulate domestic rates (using correlated randoms)
        r_domestic = self._simulate_ou(ir_dom, z_dom)

        # Simulate foreign rates
        r_foreign = self._simulate_ou(ir_for, z_for)

        # Build discount factors
        df_domestic = build_discount_factors_from_path(r_domestic, self.dt)
        df_foreign = build_discount_factors_from_path(r_foreign, self.dt)

        # Simulate FX
        fx_spot = fx_model.simulate(
            r_domestic=r_domestic,
            r_foreign=r_foreign,
            time_grid=self._time_grid,
            z_correlated=z_fx,
        )

        # Package paths data for instrument pricing
        paths_data = {
            "r_domestic": r_domestic,
            "r_foreign": r_foreign,
            "fx_spot": fx_spot,
            "df_domestic": df_domestic,
            "df_foreign": df_foreign,
        }

        # Calculate portfolio MTM at each time step
        mtm = self._calculate_portfolio_mtm(instruments, paths_data)

        return SimulationResult(
            time_grid=self._time_grid,
            r_domestic=r_domestic,
            r_foreign=r_foreign,
            fx_spot=fx_spot,
            df_domestic=df_domestic,
            df_foreign=df_foreign,
            mtm=mtm,
        )

    def _simulate_ou(
        self,
        model: OUShortRateModel,
        z: PathArray,
    ) -> PathArray:
        """
        Simulate OU process using pre-generated random numbers.

        Parameters
        ----------
        model : OUShortRateModel
            OU model parameters
        z : PathArray
            Standard normal randoms, shape (n_paths, n_steps - 1)

        Returns
        -------
        PathArray
            Simulated rate paths, shape (n_paths, n_steps)
        """
        paths = np.zeros((self.n_paths, self._n_steps))
        paths[:, 0] = model.r0

        for i in range(self._n_steps - 1):
            dt = self._time_grid[i + 1] - self._time_grid[i]
            sqrt_dt = np.sqrt(dt)

            drift = model.kappa * (model.theta - paths[:, i]) * dt
            diffusion = model.sigma * sqrt_dt * z[:, i]

            paths[:, i + 1] = paths[:, i] + drift + diffusion

        return paths

    def _calculate_portfolio_mtm(
        self,
        instruments: Sequence[Instrument],
        paths_data: dict[str, PathArray],
    ) -> PathArray:
        """
        Calculate portfolio MTM at each time step.

        Parameters
        ----------
        instruments : Sequence[Instrument]
            Portfolio of instruments
        paths_data : dict
            Simulated market data paths

        Returns
        -------
        PathArray
            Portfolio MTM, shape (n_paths, n_steps)
        """
        mtm = np.zeros((self.n_paths, self._n_steps))

        for time_idx in range(self._n_steps):
            for instrument in instruments:
                inst_mtm = instrument.calculate_mtm(
                    time_idx=time_idx,
                    time_grid=self._time_grid,
                    paths_data=paths_data,
                )
                mtm[:, time_idx] += inst_mtm

        return mtm


def run_quick_simulation(
    instruments: Sequence[Instrument],
    n_paths: int = 1000,
    horizon: float = 5.0,
    seed: int = 42,
) -> SimulationResult:
    """
    Run a quick simulation with default market parameters.

    Convenience function for testing and demos.

    Parameters
    ----------
    instruments : Sequence[Instrument]
        Instruments to simulate
    n_paths : int
        Number of paths
    horizon : float
        Horizon in years
    seed : int
        Random seed

    Returns
    -------
    SimulationResult
        Simulation results
    """
    from xva_core.config.loader import create_default_market_config

    market_config = create_default_market_config()
    engine = MonteCarloEngine(n_paths=n_paths, horizon=horizon, seed=seed)
    return engine.simulate(instruments, market_config)
