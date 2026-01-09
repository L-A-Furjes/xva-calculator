"""
Variation Margin (VM) implementation with margin period of risk.

VM is posted/received based on current exposure, with a lag representing
the margin period of risk (MPR) - the time to close out a defaulted position.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray, PathArray


@dataclass
class VariationMargin:
    """
    Variation Margin engine with threshold, MTA, and MPR.

    Models the exchange of collateral based on net exposure,
    accounting for operational delays (margin period of risk).

    Attributes
    ----------
    threshold : float
        Collateral threshold below which no VM is posted
    mta : float
        Minimum transfer amount
    mpr_days : int
        Margin period of risk in business days
    days_per_step : float
        Number of days per simulation time step

    Example
    -------
    >>> vm = VariationMargin(threshold=1e6, mta=1e5, mpr_days=10)
    >>> collat, coll_exposure = vm.apply(exposure, time_grid)
    >>> print(f"Peak collateralized EPE: {coll_exposure.clip(0).mean(0).max():,.0f}")
    """

    threshold: float = 1_000_000  # $1M
    mta: float = 100_000  # $100K
    mpr_days: int = 10  # 10 business days
    days_per_step: float = 91.25  # Quarterly = 365/4

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {self.threshold}")
        if self.mta < 0:
            raise ValueError(f"MTA must be non-negative, got {self.mta}")
        if self.mpr_days < 0:
            raise ValueError(f"MPR days must be non-negative, got {self.mpr_days}")

    @property
    def lag_steps(self) -> int:
        """Number of time steps corresponding to MPR."""
        if self.days_per_step <= 0:
            return 0
        return max(1, int(np.ceil(self.mpr_days / self.days_per_step)))

    def apply(
        self,
        exposure: PathArray,
        time_grid: FloatArray | None = None,
    ) -> tuple[PathArray, PathArray]:
        """
        Apply variation margin to exposure paths.

        Parameters
        ----------
        exposure : PathArray
            Uncollateralized exposure, shape (n_paths, n_steps)
        time_grid : FloatArray | None
            Time grid (used to compute time step size if provided)

        Returns
        -------
        tuple[PathArray, PathArray]
            (collateral_balance, collateralized_exposure)
            Both of shape (n_paths, n_steps)

        Notes
        -----
        VM logic at each time step:
        1. If |exposure| > threshold + MTA: post/receive collateral
        2. Collateral is effective after MPR lag
        3. Collateralized exposure = exposure - collateral
        """
        n_paths, n_steps = exposure.shape

        # Update days_per_step if time_grid provided
        if time_grid is not None and len(time_grid) > 1:
            dt_years = time_grid[1] - time_grid[0]
            self.days_per_step = dt_years * 365

        lag = self.lag_steps
        collateral = np.zeros_like(exposure)
        pending_calls = np.zeros((n_paths, n_steps + lag))

        for t in range(n_steps):
            # Current exposure
            exp_t = exposure[:, t]

            # Determine required collateral
            call_amount = self._calculate_margin_call(exp_t, collateral[:, t])

            # Apply call with lag
            effective_t = min(t + lag, n_steps - 1)
            pending_calls[:, effective_t] += call_amount

            # Apply pending calls that have settled
            if t < n_steps:
                collateral[:, t] += pending_calls[:, t]
                # Carry forward collateral balance
                if t < n_steps - 1:
                    collateral[:, t + 1] = collateral[:, t]

        # Calculate collateralized exposure
        coll_exposure = exposure - collateral

        return collateral, coll_exposure

    def _calculate_margin_call(
        self,
        exposure: FloatArray,
        current_collateral: FloatArray,
    ) -> FloatArray:
        """
        Calculate margin call amount.

        Parameters
        ----------
        exposure : FloatArray
            Current exposure, shape (n_paths,)
        current_collateral : FloatArray
            Current collateral balance, shape (n_paths,)

        Returns
        -------
        FloatArray
            Margin call amount (positive = post, negative = return)
        """
        n_paths = len(exposure)
        call = np.zeros(n_paths)

        # Uncollateralized portion
        uncoll = exposure - current_collateral

        # Positive exposure - we receive collateral
        pos_mask = uncoll > self.threshold + self.mta
        call[pos_mask] = uncoll[pos_mask] - self.threshold

        # Negative exposure - we post collateral
        neg_mask = uncoll < -(self.threshold + self.mta)
        call[neg_mask] = uncoll[neg_mask] + self.threshold

        # Check for collateral return (exposure reduced significantly)
        return_threshold = max(self.threshold - self.mta, 0)

        # If we were receiving collateral but exposure dropped
        return_pos = (current_collateral > 0) & (uncoll < return_threshold)
        call[return_pos] = np.minimum(uncoll[return_pos] - self.threshold, 0)

        # If we were posting collateral but exposure improved
        return_neg = (current_collateral < 0) & (uncoll > -return_threshold)
        call[return_neg] = np.maximum(uncoll[return_neg] + self.threshold, 0)

        return call


def apply_vm_to_exposure(
    exposure: PathArray,
    threshold: float = 1_000_000,
    mta: float = 100_000,
    mpr_days: int = 10,
    dt_years: float = 0.25,
) -> tuple[PathArray, PathArray]:
    """
    Convenience function to apply VM to exposure.

    Parameters
    ----------
    exposure : PathArray
        Uncollateralized exposure
    threshold : float
        Collateral threshold
    mta : float
        Minimum transfer amount
    mpr_days : int
        Margin period of risk
    dt_years : float
        Time step in years

    Returns
    -------
    tuple[PathArray, PathArray]
        (collateral, collateralized_exposure)

    Example
    -------
    >>> collat, coll_exp = apply_vm_to_exposure(
    ...     exposure,
    ...     threshold=1e6,
    ...     mta=1e5,
    ...     mpr_days=10
    ... )
    """
    vm = VariationMargin(
        threshold=threshold,
        mta=mta,
        mpr_days=mpr_days,
        days_per_step=dt_years * 365,
    )
    return vm.apply(exposure)


def calculate_collateral_profile(
    exposure: PathArray,
    threshold: float = 0.0,
    mta: float = 0.0,
) -> PathArray:
    """
    Calculate simplified collateral profile (no MPR lag).

    Parameters
    ----------
    exposure : PathArray
        Exposure paths
    threshold : float
        Collateral threshold
    mta : float
        Minimum transfer amount

    Returns
    -------
    PathArray
        Collateral balance at each time
    """
    # Simple: collateral = max(exposure - threshold, 0) for positive
    #         collateral = min(exposure + threshold, 0) for negative
    collateral = np.zeros_like(exposure)

    pos_mask = exposure > threshold
    collateral[pos_mask] = exposure[pos_mask] - threshold

    neg_mask = exposure < -threshold
    collateral[neg_mask] = exposure[neg_mask] + threshold

    return collateral
