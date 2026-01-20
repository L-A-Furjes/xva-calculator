"""
FX Forward instrument implementation.

Provides pricing for foreign exchange forward contracts
where currencies are exchanged at a pre-agreed rate on a future date.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xva_core._types import FloatArray, PathArray
from xva_core.instruments.base import Instrument, InstrumentType


@dataclass
class FXForward(Instrument):
    """
    FX Forward contract instrument.

    An FX Forward is an agreement to exchange notional amounts of two
    currencies at a predetermined rate (strike) on a future date.

    PV at time t (in domestic currency):
        V(t) = N_f × S(t) × DF_f(t,T) - N_d × DF_d(t,T)

    where:
        N_f = notional in foreign currency
        N_d = K × N_f = notional in domestic currency
        S(t) = FX spot rate (domestic per foreign)
        DF_d = domestic discount factor
        DF_f = foreign discount factor
        K = forward strike

    Attributes
    ----------
    notional_foreign : float
        Notional in foreign currency
    strike : float
        Forward strike rate (domestic per foreign)
    maturity : float
        Settlement date in years
    buy_foreign : bool
        True if we're buying foreign currency (receiving foreign, paying domestic)

    Example
    -------
    >>> fxf = FXForward(
    ...     notional_foreign=1_000_000,  # 1M EUR
    ...     strike=1.10,                  # 1.10 USD/EUR
    ...     maturity=1.0,                 # 1 year
    ...     buy_foreign=True              # We buy EUR, sell USD
    ... )
    """

    notional_foreign: float
    strike: float
    maturity: float
    buy_foreign: bool = True
    instrument_type: InstrumentType = field(
        default=InstrumentType.FX_FORWARD, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate forward parameters."""
        if self.notional_foreign <= 0:
            raise ValueError(
                f"Foreign notional must be positive, got {self.notional_foreign}"
            )
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")

    @property
    def notional(self) -> float:
        """Return notional in domestic currency terms."""
        return self.notional_foreign * self.strike

    @property
    def notional_domestic(self) -> float:
        """Domestic currency notional at strike."""
        return self.notional_foreign * self.strike

    def get_cash_flow_dates(self) -> FloatArray:
        """Get cash flow dates (just maturity for a forward)."""
        return np.array([self.maturity])

    def calculate_mtm(
        self,
        time_idx: int,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> FloatArray:
        """
        Calculate forward MTM at a given time step.

        Parameters
        ----------
        time_idx : int
            Current time index
        time_grid : FloatArray
            Time grid in years
        paths_data : dict
            Must contain:
            - 'fx_spot': FX spot paths
            - 'df_domestic': Domestic cumulative DFs
            - 'df_foreign': Foreign cumulative DFs
            Optionally for deterministic t=0 pricing:
            - 'r0_domestic', 'r0_foreign', 'fx0'

        Returns
        -------
        FloatArray
            MTM in domestic currency for each path
        """
        t = time_grid[time_idx]

        # If expired, return zero
        if t >= self.maturity:
            n_paths = paths_data["fx_spot"].shape[0]
            return np.zeros(n_paths)

        df_dom_cumulative = paths_data["df_domestic"]
        df_for_cumulative = paths_data["df_foreign"]
        n_paths = df_dom_cumulative.shape[0]

        # At t=0: use DETERMINISTIC values (today's market is known)
        if time_idx == 0 and "r0_domestic" in paths_data:
            r0_d = paths_data["r0_domestic"]
            r0_f = paths_data["r0_foreign"]
            fx0 = paths_data["fx0"]

            # Deterministic DFs
            df_dom_forward = np.exp(-r0_d * self.maturity)
            df_for_forward = np.exp(-r0_f * self.maturity)

            # Deterministic FX spot at t=0
            fx_spot = np.full(n_paths, fx0)

            df_dom_forward = np.full(n_paths, df_dom_forward)
            df_for_forward = np.full(n_paths, df_for_forward)
        else:
            # Path-dependent pricing for t>0
            fx_spot = paths_data["fx_spot"][:, time_idx]

            mat_idx = np.searchsorted(time_grid, self.maturity)
            mat_idx = min(mat_idx, len(time_grid) - 1)

            df_dom_t = df_dom_cumulative[:, time_idx]
            df_dom_T = df_dom_cumulative[:, mat_idx]
            df_dom_t = np.maximum(df_dom_t, 1e-10)
            df_dom_forward = df_dom_T / df_dom_t

            df_for_t = df_for_cumulative[:, time_idx]
            df_for_T = df_for_cumulative[:, mat_idx]
            df_for_t = np.maximum(df_for_t, 1e-10)
            df_for_forward = df_for_T / df_for_t

        # PV = N_f × S(t) × DF_f(t,T) - K × N_f × DF_d(t,T)
        pv_foreign_leg = self.notional_foreign * fx_spot * df_for_forward
        pv_domestic_leg = self.notional_domestic * df_dom_forward

        if self.buy_foreign:
            mtm = pv_foreign_leg - pv_domestic_leg
        else:
            mtm = pv_domestic_leg - pv_foreign_leg

        return mtm

    def fair_forward_rate(
        self,
        S0: float,
        r_d: float,
        r_f: float,
        t: float = 0.0,
    ) -> float:
        """
        Calculate fair (at-the-money) forward rate.

        Parameters
        ----------
        S0 : float
            Current FX spot rate
        r_d : float
            Domestic interest rate (continuously compounded)
        r_f : float
            Foreign interest rate
        t : float
            Current time (for forward starting)

        Returns
        -------
        float
            ATM forward rate: F = S0 × exp((r_d - r_f) × (T - t))
        """
        tau = self.maturity - t
        if tau <= 0:
            return S0
        return S0 * np.exp((r_d - r_f) * tau)

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value at expiry.

        Parameters
        ----------
        spot : float
            Spot rate at maturity

        Returns
        -------
        float
            Payoff at maturity
        """
        if self.buy_foreign:
            # We buy foreign at K, sell at spot
            return self.notional_foreign * (spot - self.strike)
        else:
            # We sell foreign at K, buy at spot
            return self.notional_foreign * (self.strike - spot)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "FX_FORWARD",
            "notional_foreign": self.notional_foreign,
            "strike": self.strike,
            "maturity": self.maturity,
            "buy_foreign": self.buy_foreign,
        }

    @classmethod
    def from_config(cls, config: "FXForwardConfig") -> "FXForward":  # noqa: F821
        """
        Create forward from configuration.

        Parameters
        ----------
        config : FXForwardConfig
            Forward configuration

        Returns
        -------
        FXForward
            Configured forward instance
        """
        return cls(
            notional_foreign=config.notional_foreign,
            strike=config.strike,
            maturity=config.maturity_years,
            buy_foreign=config.buy_foreign,
        )

    def __repr__(self) -> str:
        """String representation."""
        direction = "Buy" if self.buy_foreign else "Sell"
        return (
            f"FXForward({direction} {self.notional_foreign:,.0f} foreign "
            f"@ {self.strike:.4f}, T={self.maturity:.2f}Y)"
        )
