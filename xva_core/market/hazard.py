"""
Hazard rate curves for credit modeling.

Provides survival probability and default probability calculations
used in CVA and DVA computations.
"""

from dataclasses import dataclass

import numpy as np

from xva_core._types import FloatArray, Year


@dataclass
class HazardCurve:
    """
    Hazard rate curve for modeling default probabilities.

    The hazard rate λ(t) determines the instantaneous probability of default.
    Survival probability is S(t) = exp(-∫₀ᵗ λ(u) du).

    For a flat hazard rate: S(t) = exp(-λ * t)

    Attributes
    ----------
    hazard_rate : float
        Constant hazard rate (per annum)
    recovery_rate : float
        Recovery rate in case of default (0-1)

    Example
    -------
    >>> curve = HazardCurve(hazard_rate=0.012, recovery_rate=0.4)
    >>> surv_prob = curve.survival_probability(5.0)
    >>> print(f"5Y survival: {surv_prob:.2%}")
    5Y survival: 94.18%
    """

    hazard_rate: float = 0.01
    recovery_rate: float = 0.4

    def __post_init__(self) -> None:
        """Validate inputs."""
        if self.hazard_rate < 0:
            raise ValueError(
                f"Hazard rate must be non-negative, got {self.hazard_rate}"
            )
        if not 0 <= self.recovery_rate <= 1:
            raise ValueError(
                f"Recovery rate must be in [0, 1], got {self.recovery_rate}"
            )

    @property
    def lgd(self) -> float:
        """Loss given default (1 - recovery rate)."""
        return 1.0 - self.recovery_rate

    def survival_probability(self, t: Year | FloatArray) -> float | FloatArray:
        """
        Calculate survival probability to time t.

        Parameters
        ----------
        t : float | FloatArray
            Time(s) in years

        Returns
        -------
        float | FloatArray
            Survival probability S(t) = exp(-λ * t)

        Example
        -------
        >>> curve = HazardCurve(hazard_rate=0.02)
        >>> probs = curve.survival_probability(np.array([1, 3, 5]))
        """
        t_arr = np.asarray(t)
        return np.exp(-self.hazard_rate * t_arr)  # type: ignore[return-value]

    def default_probability(
        self, t1: Year, t2: Year | None = None
    ) -> float | FloatArray:
        """
        Calculate probability of default in interval [t1, t2].

        If t2 is None, calculates probability of default in [0, t1].

        Parameters
        ----------
        t1 : float
            Start time (or end time if t2 is None)
        t2 : float | None
            End time

        Returns
        -------
        float
            Probability of default PD(t1, t2) = S(t1) - S(t2)

        Notes
        -----
        This is the incremental default probability, representing
        the probability of defaulting in the period [t1, t2] given
        survival to t1.
        """
        if t2 is None:
            # PD from 0 to t1
            return float(1.0 - self.survival_probability(t1))

        s1 = self.survival_probability(t1)
        s2 = self.survival_probability(t2)
        return float(s1 - s2)

    def incremental_default_probabilities(self, time_grid: FloatArray) -> FloatArray:
        """
        Calculate incremental default probabilities for each period.

        Parameters
        ----------
        time_grid : FloatArray
            Array of time points in years

        Returns
        -------
        FloatArray
            Array of shape (len(time_grid),) with incremental PDs.
            First element is PD(0, t_0), then PD(t_i-1, t_i).

        Example
        -------
        >>> curve = HazardCurve(hazard_rate=0.01)
        >>> grid = np.array([0.25, 0.5, 0.75, 1.0])
        >>> inc_pd = curve.incremental_default_probabilities(grid)
        >>> print(f"Total 1Y PD: {inc_pd.sum():.4f}")
        """
        survival_probs = self.survival_probability(time_grid)

        # First period: from 0 to t[0]
        inc_pd = np.zeros_like(time_grid)
        inc_pd[0] = 1.0 - survival_probs[0]

        # Subsequent periods: S(t_{i-1}) - S(t_i)
        inc_pd[1:] = survival_probs[:-1] - survival_probs[1:]

        return inc_pd

    def implied_spread(self) -> float:
        """
        Calculate implied CDS spread (approximation).

        Returns
        -------
        float
            Approximate CDS spread in decimal form

        Notes
        -----
        CDS spread ≈ λ × LGD for short maturities
        """
        return self.hazard_rate * self.lgd

    @classmethod
    def from_cds_spread(
        cls, spread: float, recovery_rate: float = 0.4
    ) -> "HazardCurve":
        """
        Create hazard curve from CDS spread.

        Parameters
        ----------
        spread : float
            CDS spread in decimal (e.g., 0.01 for 100bps)
        recovery_rate : float
            Recovery rate assumption

        Returns
        -------
        HazardCurve
            Hazard curve implied by the spread

        Example
        -------
        >>> curve = HazardCurve.from_cds_spread(spread=0.01, recovery_rate=0.4)
        >>> print(f"Implied hazard rate: {curve.hazard_rate:.4f}")
        """
        lgd = 1.0 - recovery_rate
        if lgd <= 0:
            raise ValueError("LGD must be positive")

        hazard_rate = spread / lgd
        return cls(hazard_rate=hazard_rate, recovery_rate=recovery_rate)
