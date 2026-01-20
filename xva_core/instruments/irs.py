"""
Interest Rate Swap instrument implementation.

Provides pricing for fixed-for-floating interest rate swaps
where one leg pays a fixed rate and the other pays floating
(linked to the short rate in our simplified model).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xva_core._types import FloatArray, PathArray
from xva_core.instruments.base import Instrument, InstrumentType


@dataclass
class IRSwap(Instrument):
    """
    Interest Rate Swap instrument.

    A payer swap pays fixed and receives floating.
    A receiver swap receives fixed and pays floating.

    In our simplified model:
    - Fixed leg: N × K × Σ τᵢ × P(t, Tᵢ)
    - Float leg: N × [1 - P(t, T_maturity)] (approximation)

    where:
        N = notional
        K = fixed rate
        τᵢ = accrual period
        P(t, T) = forward discount factor

    Attributes
    ----------
    notional : float
        Notional amount in base currency
    fixed_rate : float
        Fixed coupon rate (decimal, e.g., 0.02 for 2%)
    maturity : float
        Swap maturity in years
    pay_fixed : bool
        True for payer swap (pay fixed, receive float)
    payment_freq : float
        Payment frequency in years (0.5 = semi-annual)
    start : float
        Swap start date in years (default 0)

    Example
    -------
    >>> swap = IRSwap(
    ...     notional=10_000_000,
    ...     fixed_rate=0.02,
    ...     maturity=5.0,
    ...     pay_fixed=True
    ... )
    >>> # Calculate MTM at t=1Y
    >>> mtm = swap.calculate_mtm(4, time_grid, paths_data)
    """

    notional: float
    fixed_rate: float
    maturity: float
    pay_fixed: bool = True
    payment_freq: float = 0.5  # Semi-annual
    start: float = 0.0
    instrument_type: InstrumentType = field(
        default=InstrumentType.IRS, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate swap parameters."""
        if self.notional <= 0:
            raise ValueError(f"Notional must be positive, got {self.notional}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
        if self.payment_freq <= 0 or self.payment_freq > 1:
            raise ValueError(
                f"Payment frequency must be in (0, 1], got {self.payment_freq}"
            )
        if self.start < 0:
            raise ValueError(f"Start date must be non-negative, got {self.start}")
        if self.start >= self.maturity:
            raise ValueError(
                f"Start ({self.start}) must be before maturity ({self.maturity})"
            )

    def get_cash_flow_dates(self) -> FloatArray:
        """
        Get payment dates for the swap.

        Returns
        -------
        FloatArray
            Array of payment dates from start to maturity
        """
        n_payments = int((self.maturity - self.start) / self.payment_freq)
        dates = self.start + np.arange(1, n_payments + 1) * self.payment_freq
        # Ensure maturity is included
        if dates[-1] < self.maturity:
            dates = np.append(dates, self.maturity)
        return dates

    def calculate_mtm(
        self,
        time_idx: int,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> FloatArray:
        """
        Calculate swap MTM at a given time step.

        Parameters
        ----------
        time_idx : int
            Current time index
        time_grid : FloatArray
            Time grid in years
        paths_data : dict
            Must contain 'df_domestic' (cumulative discount factors)
            Optionally 'r0_domestic' for deterministic t=0 pricing

        Returns
        -------
        FloatArray
            MTM for each path (positive = in-the-money for us)
        """
        t = time_grid[time_idx]

        # If expired, return zero
        if t >= self.maturity:
            n_paths = paths_data["df_domestic"].shape[0]
            return np.zeros(n_paths)

        df_cumulative = paths_data["df_domestic"]
        n_paths = df_cumulative.shape[0]

        # Get remaining payment dates
        all_cf_dates = self.get_cash_flow_dates()
        remaining_dates = all_cf_dates[all_cf_dates > t]

        if len(remaining_dates) == 0:
            return np.zeros(n_paths)

        # At t=0: use DETERMINISTIC discount factors (today's curve is known)
        # At t>0: use path-dependent discount factors
        if time_idx == 0 and "r0_domestic" in paths_data:
            # Deterministic pricing at t=0 using flat curve
            r0 = paths_data["r0_domestic"]
            forward_df = np.exp(-r0 * remaining_dates)  # Shape: (n_payments,)
            forward_df = np.tile(forward_df, (n_paths, 1))  # Broadcast to all paths
            df_maturity_fwd = np.exp(-r0 * self.maturity)
            df_maturity_fwd = np.full(n_paths, df_maturity_fwd)
        else:
            # Path-dependent pricing for t>0
            cf_indices = np.searchsorted(time_grid, remaining_dates)
            cf_indices = np.clip(cf_indices, 0, len(time_grid) - 1)

            df_t = df_cumulative[:, time_idx : time_idx + 1]
            df_t = np.maximum(df_t, 1e-10)
            df_cf = df_cumulative[:, cf_indices]
            forward_df = df_cf / df_t

            maturity_idx = np.searchsorted(time_grid, self.maturity)
            maturity_idx = min(maturity_idx, len(time_grid) - 1)
            df_maturity_fwd = df_cumulative[:, maturity_idx] / df_t.flatten()

        # Fixed leg PV: N * K * Σ τ * DF(t, Tᵢ)
        tau = self.payment_freq
        pv_fixed = self.notional * self.fixed_rate * tau * forward_df.sum(axis=1)

        # Float leg PV (approximation): N * [1 - DF(t, T_maturity)]
        pv_float = self.notional * (1.0 - df_maturity_fwd)

        # MTM from our perspective
        if self.pay_fixed:
            mtm = pv_float - pv_fixed
        else:
            mtm = pv_fixed - pv_float

        return mtm

    def calculate_pv0_deterministic(self, r0: float) -> float:
        """
        Calculate deterministic PV at t=0 using flat curve.

        This is the true today's MTM, not a Monte Carlo expectation.

        Parameters
        ----------
        r0 : float
            Today's short rate (flat curve level)

        Returns
        -------
        float
            Deterministic PV at t=0
        """
        cf_dates = self.get_cash_flow_dates()
        discount_factors = np.exp(-r0 * cf_dates)

        # Fixed leg PV
        tau = self.payment_freq
        pv_fixed = self.notional * self.fixed_rate * tau * discount_factors.sum()

        # Float leg PV
        df_maturity = np.exp(-r0 * self.maturity)
        pv_float = self.notional * (1.0 - df_maturity)

        if self.pay_fixed:
            return float(pv_float - pv_fixed)
        else:
            return float(pv_fixed - pv_float)

    def calculate_par_rate_deterministic(self, r0: float) -> float:
        """
        Calculate par rate using deterministic flat curve.

        This gives K such that PV0(K) = 0 exactly.

        Parameters
        ----------
        r0 : float
            Today's short rate (flat curve level)

        Returns
        -------
        float
            Par swap rate
        """
        cf_dates = self.get_cash_flow_dates()
        discount_factors = np.exp(-r0 * cf_dates)

        # Par rate: K = (1 - DF(T)) / (τ * Σ DF(Ti))
        df_maturity = np.exp(-r0 * self.maturity)
        annuity = self.payment_freq * discount_factors.sum()

        if annuity < 1e-10:
            return r0

        return float((1.0 - df_maturity) / annuity)

    def par_rate(
        self,
        time_idx: int,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> FloatArray:
        """
        Calculate par swap rate at current time.

        The par rate is the fixed rate that makes the swap have zero MTM.

        Parameters
        ----------
        time_idx : int
            Current time index
        time_grid : FloatArray
            Time grid
        paths_data : dict
            Simulation paths data

        Returns
        -------
        FloatArray
            Par rate for each path
        """
        t = time_grid[time_idx]

        if t >= self.maturity:
            n_paths = paths_data["df_domestic"].shape[0]
            return np.zeros(n_paths)

        df_cumulative = paths_data["df_domestic"]
        all_cf_dates = self.get_cash_flow_dates()
        remaining_dates = all_cf_dates[all_cf_dates > t]

        if len(remaining_dates) == 0:
            return np.zeros(df_cumulative.shape[0])

        cf_indices = np.searchsorted(time_grid, remaining_dates)
        cf_indices = np.clip(cf_indices, 0, len(time_grid) - 1)

        df_t = df_cumulative[:, time_idx : time_idx + 1]
        df_t = np.maximum(df_t, 1e-10)
        df_cf = df_cumulative[:, cf_indices]
        forward_df = df_cf / df_t

        maturity_idx = np.searchsorted(time_grid, self.maturity)
        maturity_idx = min(maturity_idx, len(time_grid) - 1)
        df_maturity = df_cumulative[:, maturity_idx] / df_t.flatten()

        # Par rate K = (1 - DF(T)) / (τ * Σ DF(Tᵢ))
        annuity = self.payment_freq * forward_df.sum(axis=1)
        annuity = np.maximum(annuity, 1e-10)  # Avoid division by zero

        par_rate = (1.0 - df_maturity) / annuity

        return par_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "IRS",
            "notional": self.notional,
            "fixed_rate": self.fixed_rate,
            "maturity": self.maturity,
            "pay_fixed": self.pay_fixed,
            "payment_freq": self.payment_freq,
            "start": self.start,
        }

    @classmethod
    def from_config(cls, config: "IRSwapConfig") -> "IRSwap":  # noqa: F821
        """
        Create swap from configuration.

        Parameters
        ----------
        config : IRSwapConfig
            Swap configuration

        Returns
        -------
        IRSwap
            Configured swap instance
        """
        return cls(
            notional=config.notional,
            fixed_rate=config.fixed_rate,
            maturity=config.maturity_years,
            pay_fixed=config.pay_fixed,
            payment_freq=config.payment_frequency,
        )
