"""
Margin Valuation Adjustment (MVA) calculation.

MVA represents the cost of funding initial margin (IM) posted to CCPs
or bilateral counterparties. This is a relatively new xVA introduced
with mandatory margining requirements.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray


@dataclass
class MVACalculator:
    """
    Calculator for Margin Valuation Adjustment.

    MVA = Σᵢ DF(tᵢ) × IM(tᵢ) × s_f × Δt

    where:
        IM = initial margin
        s_f = funding spread

    Attributes
    ----------
    funding_spread : float
        Funding spread for IM funding (typically same as FVA)

    Example
    -------
    >>> calc = MVACalculator(funding_spread=0.01)
    >>> im_profile = calculate_im(exposure)
    >>> mva = calc.calculate(im_profile, discount_factors, time_grid)
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
        im_profile: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
    ) -> float:
        """
        Calculate MVA from IM profile.

        Parameters
        ----------
        im_profile : FloatArray
            Initial margin at each time step
        discount_factors : FloatArray
            Discount factors
        time_grid : FloatArray
            Time points in years

        Returns
        -------
        float
            MVA value (cost)

        Notes
        -----
        MVA = Σᵢ DF(tᵢ) × IM(tᵢ) × s_f × Δt

        IM must be funded at inception and adjusted over time.
        Unlike VM, we don't receive interest on posted IM.
        """
        dt = np.diff(time_grid, prepend=0)

        mva = np.sum(discount_factors * im_profile * self.funding_spread * dt)

        return float(mva)

    def calculate_from_epe(
        self,
        epe: FloatArray,
        discount_factors: FloatArray,
        time_grid: FloatArray,
        im_multiplier: float = 1.5,
    ) -> float:
        """
        Calculate MVA using simple IM approximation.

        Parameters
        ----------
        epe : FloatArray
            Expected positive exposure
        discount_factors : FloatArray
            Discount factors
        time_grid : FloatArray
            Time grid
        im_multiplier : float
            IM as multiple of EPE

        Returns
        -------
        float
            MVA value
        """
        im_profile = im_multiplier * epe
        return self.calculate(im_profile, discount_factors, time_grid)


def calculate_mva(
    im_profile: FloatArray,
    discount_factors: FloatArray,
    time_grid: FloatArray,
    funding_spread: float = 0.01,
) -> float:
    """
    Convenience function for MVA calculation.

    Parameters
    ----------
    im_profile : FloatArray
        Initial margin profile
    discount_factors : FloatArray
        Discount factors
    time_grid : FloatArray
        Time grid
    funding_spread : float
        Funding spread

    Returns
    -------
    float
        MVA value
    """
    calc = MVACalculator(funding_spread=funding_spread)
    return calc.calculate(im_profile, discount_factors, time_grid)
