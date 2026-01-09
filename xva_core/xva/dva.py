"""
Debt Valuation Adjustment (DVA) calculation.

DVA represents the benefit from our own potential default - it's the
counterparty's CVA from their perspective, but a gain for us.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray, PathArray
from xva_core.market.hazard import HazardCurve


@dataclass
class DVACalculator:
    """
    Calculator for Debt Valuation Adjustment.

    DVA = Σᵢ DF(tᵢ) × ENE(tᵢ) × LGD_own × ΔPD_own(tᵢ)

    where:
        ENE = expected negative exposure (our liability)
        LGD_own = our loss given default
        ΔPD_own = our incremental default probability

    DVA is typically reported as a benefit (reduces xVA cost).

    Attributes
    ----------
    lgd : float
        Our loss given default
    hazard_rate : float
        Our hazard rate (per annum)

    Example
    -------
    >>> calc = DVACalculator(lgd=0.6, hazard_rate=0.01)
    >>> dva = calc.calculate(ene, discount_factors, time_grid)
    >>> print(f"DVA (benefit): ${dva:,.0f}")
    """

    lgd: float = 0.60
    hazard_rate: float = 0.010  # 100 bps

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
        ene: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate DVA from ENE profile.

        Parameters
        ----------
        ene : FloatArray
            Expected negative exposure at each time, shape (n_steps,)
        discount_factors : FloatArray
            Discount factors at each time, shape (n_steps,)
        time_grid : FloatArray
            Time points in years, shape (n_steps,)

        Returns
        -------
        float
            DVA value (positive = benefit)

        Notes
        -----
        DVA = Σᵢ DF(tᵢ) × ENE(tᵢ) × LGD × ΔPD_own(tᵢ)
        """
        hazard = self.hazard_curve
        inc_pd = hazard.incremental_default_probabilities(time_grid)

        dva = np.sum(discount_factors * ene * self.lgd * inc_pd)

        return float(dva)

    def calculate_from_paths(
        self,
        exposure: PathArray,
        discount_factors: PathArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate DVA directly from exposure paths.

        Parameters
        ----------
        exposure : PathArray
            Exposure paths (negative values are our liability)
        discount_factors : PathArray
            Discount factor paths
        time_grid : FloatArray
            Time grid

        Returns
        -------
        float
            DVA value
        """
        # ENE: average negative exposure (as positive value)
        ene = np.maximum(-exposure, 0).mean(axis=0)

        # Average discount factors
        avg_df = discount_factors.mean(axis=0)

        return self.calculate(ene, avg_df, time_grid)


def calculate_dva(
    ene: FloatArray,
    discount_factors: FloatArray,
    time_grid: FloatArray,
    lgd: float = 0.6,
    hazard_rate: float = 0.01,
) -> float:
    """
    Convenience function for DVA calculation.

    Parameters
    ----------
    ene : FloatArray
        Expected negative exposure
    discount_factors : FloatArray
        Discount factors
    time_grid : FloatArray
        Time grid
    lgd : float
        Our loss given default
    hazard_rate : float
        Our hazard rate

    Returns
    -------
    float
        DVA value (benefit)
    """
    calc = DVACalculator(lgd=lgd, hazard_rate=hazard_rate)
    return calc.calculate(ene, discount_factors, time_grid)
