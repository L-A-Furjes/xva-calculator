"""
Credit Valuation Adjustment (CVA) calculation.

CVA represents the market price of counterparty credit risk - the expected
loss due to counterparty default weighted by their probability of default.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray, PathArray
from xva_core.market.hazard import HazardCurve


@dataclass
class CVACalculator:
    """
    Calculator for Credit Valuation Adjustment.

    CVA = Σᵢ DF(tᵢ) × EPE(tᵢ) × LGD × ΔPD(tᵢ)

    where:
        DF = discount factor
        EPE = expected positive exposure
        LGD = loss given default
        ΔPD = incremental default probability

    Attributes
    ----------
    lgd : float
        Loss given default (1 - recovery rate)
    hazard_rate : float
        Constant hazard rate (per annum)

    Example
    -------
    >>> calc = CVACalculator(lgd=0.6, hazard_rate=0.012)
    >>> cva = calc.calculate(epe, discount_factors, time_grid)
    >>> print(f"CVA: ${cva:,.0f}")
    """

    lgd: float = 0.60
    hazard_rate: float = 0.012  # 120 bps

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0 <= self.lgd <= 1:
            raise ValueError(f"LGD must be in [0, 1], got {self.lgd}")
        if self.hazard_rate < 0:
            raise ValueError(f"Hazard rate must be non-negative, got {self.hazard_rate}")

    @property
    def hazard_curve(self) -> HazardCurve:
        """Create hazard curve from parameters."""
        return HazardCurve(hazard_rate=self.hazard_rate, recovery_rate=1 - self.lgd)

    def calculate(
        self,
        epe: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate CVA from EPE profile.

        Parameters
        ----------
        epe : FloatArray
            Expected positive exposure at each time, shape (n_steps,)
        discount_factors : FloatArray
            Discount factors at each time, shape (n_steps,)
        time_grid : FloatArray
            Time points in years, shape (n_steps,)

        Returns
        -------
        float
            CVA value in base currency

        Notes
        -----
        CVA = Σᵢ DF(tᵢ) × EPE(tᵢ) × LGD × ΔPD(tᵢ)
        """
        # Get incremental default probabilities
        hazard = self.hazard_curve
        inc_pd = hazard.incremental_default_probabilities(time_grid)

        # CVA calculation
        cva = np.sum(discount_factors * epe * self.lgd * inc_pd)

        return float(cva)

    def calculate_from_paths(
        self,
        exposure: PathArray,
        discount_factors: PathArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate CVA directly from exposure and DF paths.

        Parameters
        ----------
        exposure : PathArray
            Exposure paths, shape (n_paths, n_steps)
        discount_factors : PathArray
            Discount factor paths, shape (n_paths, n_steps)
        time_grid : FloatArray
            Time grid

        Returns
        -------
        float
            CVA value
        """
        # EPE: average positive exposure across paths
        epe = np.maximum(exposure, 0).mean(axis=0)

        # Average discount factors
        avg_df = discount_factors.mean(axis=0)

        return self.calculate(epe, avg_df, time_grid)

    def calculate_marginal_cva(
        self,
        epe_portfolio: FloatArray,
        epe_incremental: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate marginal CVA for an incremental trade.

        Marginal CVA measures the additional CVA from adding a trade
        to an existing portfolio.

        Parameters
        ----------
        epe_portfolio : FloatArray
            EPE of portfolio without new trade
        epe_incremental : FloatArray
            EPE of portfolio with new trade
        discount_factors : FloatArray
            Discount factors
        time_grid : FloatArray
            Time grid

        Returns
        -------
        float
            Marginal CVA
        """
        cva_before = self.calculate(epe_portfolio, discount_factors, time_grid)
        cva_after = self.calculate(epe_incremental, discount_factors, time_grid)
        return cva_after - cva_before

    def sensitivity_to_hazard_rate(
        self,
        epe: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
        bump: float = 0.0001,  # 1 bp
    ) -> float:
        """
        Calculate CVA sensitivity to hazard rate (CS01).

        Parameters
        ----------
        epe : FloatArray
            EPE profile
        discount_factors : FloatArray
            Discount factors
        time_grid : FloatArray
            Time grid
        bump : float
            Hazard rate bump size

        Returns
        -------
        float
            CVA change per 1bp hazard rate move
        """
        base_cva = self.calculate(epe, discount_factors, time_grid)

        bumped_calc = CVACalculator(lgd=self.lgd, hazard_rate=self.hazard_rate + bump)
        bumped_cva = bumped_calc.calculate(epe, discount_factors, time_grid)

        return (bumped_cva - base_cva) / (bump * 10000)  # Per 1bp


def calculate_unilateral_cva(
    epe: FloatArray,
    discount_factors: FloatArray,
    time_grid: FloatArray,
    lgd: float = 0.6,
    hazard_rate: float = 0.012,
) -> float:
    """
    Convenience function for unilateral CVA calculation.

    Parameters
    ----------
    epe : FloatArray
        Expected positive exposure
    discount_factors : FloatArray
        Discount factors
    time_grid : FloatArray
        Time grid
    lgd : float
        Loss given default
    hazard_rate : float
        Counterparty hazard rate

    Returns
    -------
    float
        CVA value
    """
    calc = CVACalculator(lgd=lgd, hazard_rate=hazard_rate)
    return calc.calculate(epe, discount_factors, time_grid)
