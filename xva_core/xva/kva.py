"""
Capital Valuation Adjustment (KVA) calculation.

KVA represents the cost of regulatory capital required to support
counterparty credit risk. Banks must hold capital against potential
losses, and this capital has an opportunity cost.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray


@dataclass
class KVACalculator:
    """
    Calculator for Capital Valuation Adjustment.

    KVA = Σᵢ DF(tᵢ) × K(tᵢ) × CoC × Δt

    where:
        K = regulatory capital requirement
        CoC = cost of capital (hurdle rate)

    Attributes
    ----------
    cost_of_capital : float
        Hurdle rate for capital (e.g., 0.10 for 10%)
    capital_ratio : float
        Capital ratio applied to EAD (e.g., 0.08 for 8%)

    Example
    -------
    >>> calc = KVACalculator(cost_of_capital=0.10, capital_ratio=0.08)
    >>> kva = calc.calculate(ead_profile, discount_factors, time_grid)
    """

    cost_of_capital: float = 0.10  # 10%
    capital_ratio: float = 0.08  # 8% Tier 1

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.cost_of_capital < 0:
            raise ValueError(
                f"Cost of capital must be non-negative, got {self.cost_of_capital}"
            )
        if not 0 < self.capital_ratio <= 1:
            raise ValueError(
                f"Capital ratio must be in (0, 1], got {self.capital_ratio}"
            )

    def calculate(
        self,
        ead_profile: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate KVA from EAD profile.

        Parameters
        ----------
        ead_profile : FloatArray
            Exposure at default at each time step
        discount_factors : FloatArray
            Discount factors
        time_grid : FloatArray
            Time points in years

        Returns
        -------
        float
            KVA value (cost)

        Notes
        -----
        K(t) = β × EAD(t)
        KVA = Σᵢ DF(tᵢ) × K(tᵢ) × CoC × Δt
        """
        dt = np.diff(time_grid, prepend=0)

        # Capital requirement
        capital = self.capital_ratio * ead_profile

        # KVA calculation
        kva = np.sum(discount_factors * capital * self.cost_of_capital * dt)

        return float(kva)

    def calculate_from_epe(
        self,
        epe: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
        alpha: float = 1.4,
    ) -> float:
        """
        Calculate KVA using EPE as EAD proxy.

        Parameters
        ----------
        epe : FloatArray
            Expected positive exposure
        discount_factors : FloatArray
            Discount factors
        time_grid : FloatArray
            Time grid
        alpha : float
            Regulatory multiplier (1.4 for SA-CCR)

        Returns
        -------
        float
            KVA value
        """
        # Simple EAD proxy: alpha × EPE
        ead_profile = alpha * epe
        return self.calculate(ead_profile, discount_factors, time_grid)


def calculate_kva(
    ead_profile: FloatArray,
    discount_factors: FloatArray,
    time_grid: FloatArray,
    cost_of_capital: float = 0.10,
    capital_ratio: float = 0.08,
) -> float:
    """
    Convenience function for KVA calculation.

    Parameters
    ----------
    ead_profile : FloatArray
        EAD profile
    discount_factors : FloatArray
        Discount factors
    time_grid : FloatArray
        Time grid
    cost_of_capital : float
        Hurdle rate
    capital_ratio : float
        Capital requirement ratio

    Returns
    -------
    float
        KVA value
    """
    calc = KVACalculator(cost_of_capital=cost_of_capital, capital_ratio=capital_ratio)
    return calc.calculate(ead_profile, discount_factors, time_grid)
