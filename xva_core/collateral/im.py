"""
Initial Margin (IM) calculation for MVA.

IM is posted at trade inception and adjusted periodically to cover
potential future exposure over a short risk horizon (typically 10 days).
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray, PathArray


@dataclass
class InitialMargin:
    """
    Initial Margin calculator.

    Two approaches are supported:
    1. Simple: IM = multiplier × EPE (quick approximation)
    2. Quantile-based: IM = 99% quantile of max exposure over MPR

    Attributes
    ----------
    multiplier : float
        Multiplier for simple IM (default 1.5)
    quantile : float
        Quantile level for SIMM-like calculation (default 0.99)
    mpr_steps : int
        Number of steps in the MPR window (default 1)
    method : str
        Calculation method: 'simple' or 'quantile'

    Example
    -------
    >>> im = InitialMargin(multiplier=1.5)
    >>> im_profile = im.calculate(exposure, time_grid)
    """

    multiplier: float = 1.5
    quantile: float = 0.99
    mpr_steps: int = 1
    method: str = "simple"

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.multiplier < 1.0:
            raise ValueError(f"IM multiplier must be >= 1.0, got {self.multiplier}")
        if not 0 < self.quantile < 1:
            raise ValueError(f"Quantile must be in (0, 1), got {self.quantile}")
        if self.method not in ("simple", "quantile"):
            raise ValueError(f"Method must be 'simple' or 'quantile', got {self.method}")

    def calculate(
        self,
        exposure: PathArray,
        time_grid: FloatArray | None = None,
    ) -> FloatArray:
        """
        Calculate IM profile over time.

        Parameters
        ----------
        exposure : PathArray
            Exposure paths, shape (n_paths, n_steps)
        time_grid : FloatArray | None
            Time grid (not used currently, for API consistency)

        Returns
        -------
        FloatArray
            IM at each time step, shape (n_steps,)

        Notes
        -----
        Simple method: IM(t) = multiplier × EPE(t)
        Quantile method: IM(t) = quantile of max exposure over [t, t+MPR]
        """
        if self.method == "simple":
            return self._calculate_simple(exposure)
        else:
            return self._calculate_quantile(exposure)

    def _calculate_simple(self, exposure: PathArray) -> FloatArray:
        """
        Simple IM calculation as multiple of EPE.

        Parameters
        ----------
        exposure : PathArray
            Exposure paths

        Returns
        -------
        FloatArray
            IM profile
        """
        epe = np.maximum(exposure, 0).mean(axis=0)
        return self.multiplier * epe

    def _calculate_quantile(self, exposure: PathArray) -> FloatArray:
        """
        Quantile-based IM calculation.

        For each time t, compute the quantile of the maximum positive
        exposure over the MPR window [t, t + mpr_steps].

        Parameters
        ----------
        exposure : PathArray
            Exposure paths

        Returns
        -------
        FloatArray
            IM profile
        """
        n_paths, n_steps = exposure.shape
        im_profile = np.zeros(n_steps)

        for t in range(n_steps):
            end_t = min(t + self.mpr_steps + 1, n_steps)
            window = exposure[:, t:end_t]
            max_exposure = np.maximum(window.max(axis=1), 0)
            im_profile[t] = np.quantile(max_exposure, self.quantile)

        return im_profile

    def calculate_with_netting(
        self,
        exposure: PathArray,
        net_to_gross_ratio: float = 1.0,
    ) -> FloatArray:
        """
        Calculate IM with netting benefit.

        Parameters
        ----------
        exposure : PathArray
            Exposure paths
        net_to_gross_ratio : float
            NGR for the netting set (1.0 = no benefit)

        Returns
        -------
        FloatArray
            IM profile with netting adjustment
        """
        base_im = self.calculate(exposure)
        # Apply NGR adjustment (simplified)
        return base_im * np.sqrt(net_to_gross_ratio)


def calculate_im_from_epe(
    epe: FloatArray,
    multiplier: float = 1.5,
) -> FloatArray:
    """
    Quick IM calculation from EPE profile.

    Parameters
    ----------
    epe : FloatArray
        EPE profile
    multiplier : float
        IM multiplier

    Returns
    -------
    FloatArray
        IM profile
    """
    return multiplier * epe


def calculate_simm_approximation(
    exposure: PathArray,
    delta_sensitivity: float | None = None,
    vega_sensitivity: float | None = None,
    curvature_sensitivity: float | None = None,
) -> FloatArray:
    """
    Simplified SIMM-like IM calculation.

    This is a very simplified approximation. Real SIMM uses:
    - Delta sensitivities per risk factor
    - Vega sensitivities for options
    - Curvature charges
    - Cross-bucket correlations

    Parameters
    ----------
    exposure : PathArray
        Exposure paths
    delta_sensitivity : float | None
        Delta sensitivity (if provided, overrides exposure-based)
    vega_sensitivity : float | None
        Vega sensitivity
    curvature_sensitivity : float | None
        Curvature sensitivity

    Returns
    -------
    FloatArray
        SIMM-like IM profile
    """
    n_steps = exposure.shape[1]

    if delta_sensitivity is not None:
        # Use provided sensitivities
        delta_charge = abs(delta_sensitivity) * 0.004  # ~40bps for IR
        vega_charge = abs(vega_sensitivity or 0) * 0.55  # 55% for vega
        curv_charge = abs(curvature_sensitivity or 0) * 0.5

        total_im = np.sqrt(delta_charge**2 + vega_charge**2 + curv_charge**2)
        return np.full(n_steps, total_im)

    # Fallback: use exposure-based approximation
    epe = np.maximum(exposure, 0).mean(axis=0)
    pfe_99 = np.quantile(np.maximum(exposure, 0), 0.99, axis=0)

    # SIMM-like: blend of EPE and tail risk
    return 0.5 * epe + 0.5 * pfe_99
