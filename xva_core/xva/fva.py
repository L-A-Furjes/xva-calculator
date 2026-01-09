"""
Funding Valuation Adjustment (FVA) calculation.

FVA represents the cost of funding uncollateralized positive exposure.
When we have positive MTM with a counterparty, we need to fund that
exposure at our funding rate (above risk-free).
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray


@dataclass
class FVACalculator:
    """
    Calculator for Funding Valuation Adjustment.

    FVA = Σᵢ DF(tᵢ) × EPE(tᵢ) × s_f × Δt

    where:
        s_f = funding spread (over OIS)
        Δt = time step

    Attributes
    ----------
    funding_spread : float
        Funding spread over OIS (e.g., 0.01 for 100 bps)

    Example
    -------
    >>> calc = FVACalculator(funding_spread=0.01)  # 100 bps
    >>> fva = calc.calculate(epe, discount_factors, time_grid)
    """

    funding_spread: float = 0.01  # 100 bps

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.funding_spread < 0:
            raise ValueError(
                f"Funding spread must be non-negative, got {self.funding_spread}"
            )

    def calculate(
        self,
        epe: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate FVA from EPE profile.

        Parameters
        ----------
        epe : FloatArray
            Expected positive exposure at each time
        discount_factors : FloatArray
            Discount factors at each time
        time_grid : FloatArray
            Time points in years

        Returns
        -------
        float
            FVA value (cost)

        Notes
        -----
        FVA = Σᵢ DF(tᵢ) × EPE(tᵢ) × s_f × Δt

        This represents the cost of borrowing to fund positive exposure.
        """
        # Calculate time steps
        dt = np.diff(time_grid, prepend=0)

        # FVA calculation
        fva = np.sum(discount_factors * epe * self.funding_spread * dt)

        return float(fva)

    def calculate_symmetric(
        self,
        epe: FloatArray,
        ene: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
        lending_spread: float | None = None,
    ) -> float:
        """
        Calculate symmetric FVA including funding benefit from ENE.

        Parameters
        ----------
        epe : FloatArray
            Expected positive exposure
        ene : FloatArray
            Expected negative exposure
        discount_factors : FloatArray
            Discount factors
        time_grid : FloatArray
            Time grid
        lending_spread : float | None
            Spread for lending (if different from borrowing)

        Returns
        -------
        float
            Net FVA (FCA - FBA)
        """
        dt = np.diff(time_grid, prepend=0)

        # Funding Cost Adjustment (FCA) - cost to fund positive exposure
        fca = np.sum(discount_factors * epe * self.funding_spread * dt)

        # Funding Benefit Adjustment (FBA) - benefit from receiving funds
        lending = lending_spread if lending_spread is not None else self.funding_spread
        fba = np.sum(discount_factors * ene * lending * dt)

        return float(fca - fba)


def calculate_fva(
    epe: FloatArray,
    discount_factors: FloatArray,
    time_grid: FloatArray,
    funding_spread: float = 0.01,
) -> float:
    """
    Convenience function for FVA calculation.

    Parameters
    ----------
    epe : FloatArray
        Expected positive exposure
    discount_factors : FloatArray
        Discount factors
    time_grid : FloatArray
        Time grid
    funding_spread : float
        Funding spread in decimal

    Returns
    -------
    float
        FVA value
    """
    calc = FVACalculator(funding_spread=funding_spread)
    return calc.calculate(epe, discount_factors, time_grid)
