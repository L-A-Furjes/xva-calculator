"""
Discount curve implementation for risk-free discounting.

Provides flat and piecewise constant discount curves with
efficient vectorized discount factor calculation.
"""

from dataclasses import dataclass, field

import numpy as np

from xva_core._types import FloatArray, Year


@dataclass
class DiscountCurve:
    """
    Discount curve for calculating present values.

    Supports flat curves (single rate) and piecewise constant curves
    (different rates for different tenors).

    Attributes
    ----------
    rate : float
        Flat rate for simple curves (continuously compounded)
    tenors : FloatArray | None
        Tenor points for piecewise curves (in years)
    rates : FloatArray | None
        Rates corresponding to each tenor

    Example
    -------
    >>> # Flat curve at 2%
    >>> curve = DiscountCurve(rate=0.02)
    >>> df = curve.discount_factor(1.0)
    >>> print(f"1Y DF: {df:.4f}")
    1Y DF: 0.9802

    >>> # Piecewise curve
    >>> curve = DiscountCurve(
    ...     tenors=np.array([1.0, 2.0, 5.0]),
    ...     rates=np.array([0.02, 0.025, 0.03])
    ... )
    """

    rate: float = 0.02
    tenors: FloatArray | None = field(default=None)
    rates: FloatArray | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate curve inputs."""
        if self.tenors is not None and self.rates is not None:
            if len(self.tenors) != len(self.rates):
                raise ValueError(
                    f"Tenors and rates must have same length, "
                    f"got {len(self.tenors)} and {len(self.rates)}"
                )
            if not np.all(np.diff(self.tenors) > 0):
                raise ValueError("Tenors must be strictly increasing")

    def _get_rate(self, t: float | FloatArray) -> float | FloatArray:
        """
        Get the applicable rate for a given time point.

        Uses flat rate if no piecewise curve defined, otherwise
        interpolates (flat extrapolation at boundaries).
        """
        if self.tenors is None or self.rates is None:
            return self.rate

        # Piecewise constant (step function) interpolation
        return np.interp(t, self.tenors, self.rates)  # type: ignore[return-value]

    def discount_factor(
        self, t: Year | FloatArray, t_start: Year = 0.0
    ) -> float | FloatArray:
        """
        Calculate discount factor from t_start to t.

        Parameters
        ----------
        t : float | FloatArray
            End time(s) in years
        t_start : float
            Start time in years (default 0)

        Returns
        -------
        float | FloatArray
            Discount factor(s) P(t_start, t)

        Notes
        -----
        DF(t_start, t) = exp(-r * (t - t_start))
        For t <= t_start, returns 1.0
        """
        dt = np.maximum(np.asarray(t) - t_start, 0.0)
        r = self._get_rate(t)
        return np.exp(-r * dt)  # type: ignore[return-value]

    def forward_rate(self, t1: Year, t2: Year) -> float:
        """
        Calculate continuously compounded forward rate.

        Parameters
        ----------
        t1 : float
            Start time in years
        t2 : float
            End time in years

        Returns
        -------
        float
            Forward rate f(t1, t2)

        Notes
        -----
        f(t1, t2) = -[ln(P(0,t2)) - ln(P(0,t1))] / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1})")

        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)

        return float(-np.log(df2 / df1) / (t2 - t1))

    def zero_rate(self, t: Year) -> float:
        """
        Calculate zero rate to time t.

        Parameters
        ----------
        t : float
            Time in years

        Returns
        -------
        float
            Zero rate z(t) such that DF(t) = exp(-z(t) * t)
        """
        if t <= 0:
            return float(self._get_rate(0.01))

        df = self.discount_factor(t)
        return float(-np.log(df) / t)


def build_discount_factors_from_path(
    rates: FloatArray, dt: float, cumulative: bool = True
) -> FloatArray:
    """
    Build discount factors from a simulated rate path.

    Parameters
    ----------
    rates : FloatArray
        Array of shape (n_paths, n_steps) with short rates
    dt : float
        Time step size in years
    cumulative : bool
        If True, return cumulative DF from t=0 to each point.
        If False, return period-by-period DF.

    Returns
    -------
    FloatArray
        Discount factors of same shape as input

    Example
    -------
    >>> rates = np.array([[0.02, 0.025, 0.03]])
    >>> df = build_discount_factors_from_path(rates, dt=0.25)
    >>> # df[0, i] = exp(-sum(r[0, 0:i] * dt))
    """
    # Period DFs: exp(-r * dt)
    period_df = np.exp(-rates * dt)

    if cumulative:
        # Cumulative product from left
        # DF(0, t_i) = prod_{j=0}^{i-1} exp(-r_j * dt)
        cumulative_df = np.cumprod(period_df, axis=1)
        # Shift right and set first column to 1.0 (DF at t=0)
        result = np.ones_like(cumulative_df)
        result[:, 1:] = cumulative_df[:, :-1]
        return result

    return period_df
