"""
Exposure metrics calculation for xVA.

Provides functions to compute:
- EPE (Expected Positive Exposure)
- ENE (Expected Negative Exposure)
- PFE (Potential Future Exposure)
- EE (Expected Exposure)
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from xva_core._types import FloatArray, PathArray


class ExposureProfile(NamedTuple):
    """Container for exposure profile data."""

    time_grid: FloatArray
    values: FloatArray


@dataclass
class ExposureMetrics:
    """
    Complete exposure metrics for a netting set.

    Attributes
    ----------
    time_grid : FloatArray
        Time points in years
    epe : FloatArray
        Expected positive exposure at each time
    ene : FloatArray
        Expected negative exposure at each time
    pfe_95 : FloatArray
        95% PFE at each time
    pfe_99 : FloatArray
        99% PFE at each time
    expected_exposure : FloatArray
        Expected exposure (unsigned) at each time
    peak_epe : float
        Maximum EPE across all times
    peak_ene : float
        Maximum ENE across all times
    average_epe : float
        Time-weighted average EPE
    average_ene : float
        Time-weighted average ENE
    """

    time_grid: FloatArray
    epe: FloatArray
    ene: FloatArray
    pfe_95: FloatArray
    pfe_99: FloatArray
    expected_exposure: FloatArray
    peak_epe: float
    peak_ene: float
    average_epe: float
    average_ene: float

    @classmethod
    def from_mtm(cls, mtm: PathArray, time_grid: FloatArray) -> "ExposureMetrics":
        """
        Calculate all exposure metrics from MTM matrix.

        Parameters
        ----------
        mtm : PathArray
            Portfolio MTM, shape (n_paths, n_steps)
        time_grid : FloatArray
            Time points in years

        Returns
        -------
        ExposureMetrics
            Complete exposure metrics
        """
        epe = calculate_epe(mtm)
        ene = calculate_ene(mtm)
        pfe_95 = calculate_pfe(mtm, quantile=0.95)
        pfe_99 = calculate_pfe(mtm, quantile=0.99)
        ee = calculate_expected_exposure(mtm)

        peak_epe = float(np.max(epe))
        peak_ene = float(np.max(ene))

        # Time-weighted averages
        dt = np.diff(time_grid, prepend=0)
        total_time = time_grid[-1] if time_grid[-1] > 0 else 1.0
        average_epe = float(np.sum(epe * dt) / total_time)
        average_ene = float(np.sum(ene * dt) / total_time)

        return cls(
            time_grid=time_grid,
            epe=epe,
            ene=ene,
            pfe_95=pfe_95,
            pfe_99=pfe_99,
            expected_exposure=ee,
            peak_epe=peak_epe,
            peak_ene=peak_ene,
            average_epe=average_epe,
            average_ene=average_ene,
        )


def calculate_epe(mtm: PathArray) -> FloatArray:
    """
    Calculate Expected Positive Exposure at each time step.

    EPE(t) = E[max(V(t), 0)]

    Parameters
    ----------
    mtm : PathArray
        MTM values, shape (n_paths, n_steps)

    Returns
    -------
    FloatArray
        EPE at each time step, shape (n_steps,)

    Example
    -------
    >>> mtm = np.random.randn(1000, 20) * 1e6
    >>> epe = calculate_epe(mtm)
    >>> print(f"Peak EPE: ${epe.max():,.0f}")
    """
    positive_exposure = np.maximum(mtm, 0)
    return positive_exposure.mean(axis=0)


def calculate_ene(mtm: PathArray) -> FloatArray:
    """
    Calculate Expected Negative Exposure at each time step.

    ENE(t) = E[max(-V(t), 0)]

    This represents the counterparty's exposure to us.

    Parameters
    ----------
    mtm : PathArray
        MTM values, shape (n_paths, n_steps)

    Returns
    -------
    FloatArray
        ENE at each time step, shape (n_steps,)
    """
    negative_exposure = np.maximum(-mtm, 0)
    return negative_exposure.mean(axis=0)


def calculate_pfe(mtm: PathArray, quantile: float = 0.95) -> FloatArray:
    """
    Calculate Potential Future Exposure at each time step.

    PFE(t, α) = Quantile_α(max(V(t), 0))

    Parameters
    ----------
    mtm : PathArray
        MTM values, shape (n_paths, n_steps)
    quantile : float
        Quantile level (default 0.95 for 95% PFE)

    Returns
    -------
    FloatArray
        PFE at each time step, shape (n_steps,)

    Example
    -------
    >>> pfe_95 = calculate_pfe(mtm, quantile=0.95)
    >>> pfe_99 = calculate_pfe(mtm, quantile=0.99)
    """
    if not 0 < quantile < 1:
        raise ValueError(f"Quantile must be in (0, 1), got {quantile}")

    positive_exposure = np.maximum(mtm, 0)
    return np.quantile(positive_exposure, quantile, axis=0)


def calculate_expected_exposure(mtm: PathArray) -> FloatArray:
    """
    Calculate Expected Exposure (unsigned) at each time step.

    EE(t) = E[|V(t)|]

    Parameters
    ----------
    mtm : PathArray
        MTM values, shape (n_paths, n_steps)

    Returns
    -------
    FloatArray
        EE at each time step, shape (n_steps,)
    """
    return np.abs(mtm).mean(axis=0)


def calculate_effective_epe(epe: FloatArray) -> FloatArray:
    """
    Calculate Effective EPE (non-decreasing EPE).

    Effective EPE at time t is the maximum of EPE from 0 to t.
    This is used in regulatory capital calculations.

    Parameters
    ----------
    epe : FloatArray
        EPE profile, shape (n_steps,)

    Returns
    -------
    FloatArray
        Effective EPE, shape (n_steps,)
    """
    return np.maximum.accumulate(epe)


def calculate_effective_expected_exposure(
    mtm: PathArray,
) -> FloatArray:
    """
    Calculate Effective Expected Exposure.

    EEE(t) = max_{s <= t} E[max(V(s), 0)]

    Parameters
    ----------
    mtm : PathArray
        MTM values

    Returns
    -------
    FloatArray
        EEE profile
    """
    epe = calculate_epe(mtm)
    return calculate_effective_epe(epe)


def calculate_exposure_at_default(
    epe: FloatArray,
    time_grid: FloatArray,
    maturity_weighted: bool = True,
) -> float:
    """
    Calculate Exposure at Default (EAD) for internal models.

    EAD = Integral of effective EPE, or simplified as average EPE × maturity

    Parameters
    ----------
    epe : FloatArray
        EPE profile
    time_grid : FloatArray
        Time grid in years
    maturity_weighted : bool
        If True, use maturity-weighted average

    Returns
    -------
    float
        EAD estimate
    """
    effective_epe = calculate_effective_epe(epe)

    if maturity_weighted:
        # Time-weighted integral
        dt = np.diff(time_grid, prepend=0)
        return float(np.sum(effective_epe * dt))
    else:
        # Simple average × horizon
        return float(effective_epe.mean() * time_grid[-1])


def collateral_benefit_ratio(
    epe_uncoll: FloatArray,
    epe_coll: FloatArray,
) -> float:
    """
    Calculate the reduction in exposure from collateralization.

    Parameters
    ----------
    epe_uncoll : FloatArray
        Uncollateralized EPE profile
    epe_coll : FloatArray
        Collateralized EPE profile

    Returns
    -------
    float
        Reduction ratio: (1 - peak_coll / peak_uncoll) × 100
    """
    peak_uncoll = np.max(epe_uncoll)
    peak_coll = np.max(epe_coll)

    if peak_uncoll < 1e-10:
        return 0.0

    return (1 - peak_coll / peak_uncoll) * 100
