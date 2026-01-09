"""
xVA result container and combined calculation.

Provides a unified interface for calculating and storing all xVA metrics.
"""

from dataclasses import dataclass

from xva_core._types import FloatArray
from xva_core.xva.cva import CVACalculator
from xva_core.xva.dva import DVACalculator
from xva_core.xva.fva import FVACalculator
from xva_core.xva.kva import KVACalculator
from xva_core.xva.mva import MVACalculator


@dataclass
class XVAResult:
    """
    Container for all xVA calculation results.

    Attributes
    ----------
    cva : float
        Credit Valuation Adjustment (cost)
    dva : float
        Debt Valuation Adjustment (benefit)
    fva : float
        Funding Valuation Adjustment (cost)
    mva : float
        Margin Valuation Adjustment (cost)
    kva : float
        Capital Valuation Adjustment (cost)
    total : float
        Total xVA = CVA - DVA + FVA + MVA + KVA

    Example
    -------
    >>> result = calculate_all_xva(exposure, discount_factors, time_grid, params)
    >>> print(f"Total xVA: ${result.total:,.0f}")
    >>> print(result.to_dict())
    """

    cva: float
    dva: float
    fva: float
    mva: float
    kva: float

    @property
    def total(self) -> float:
        """
        Total xVA cost.

        Total = CVA - DVA + FVA + MVA + KVA

        DVA is subtracted as it represents a benefit.
        """
        return self.cva - self.dva + self.fva + self.mva + self.kva

    @property
    def total_cost(self) -> float:
        """Alias for total."""
        return self.total

    @property
    def bilateral_cva(self) -> float:
        """Bilateral CVA = CVA - DVA."""
        return self.cva - self.dva

    def to_dict(self) -> dict[str, float]:
        """
        Convert to dictionary.

        Returns
        -------
        dict[str, float]
            Dictionary with all xVA components
        """
        return {
            "cva": self.cva,
            "dva": self.dva,
            "fva": self.fva,
            "mva": self.mva,
            "kva": self.kva,
            "total": self.total,
            "bilateral_cva": self.bilateral_cva,
        }

    def to_bps(self, notional: float) -> dict[str, float]:
        """
        Convert to basis points of notional.

        Parameters
        ----------
        notional : float
            Total notional

        Returns
        -------
        dict[str, float]
            xVA components in basis points
        """
        if notional <= 0:
            return dict.fromkeys(self.to_dict(), 0.0)

        multiplier = 10000 / notional  # Convert to bps
        return {k: v * multiplier for k, v in self.to_dict().items()}

    def summary(self, notional: float | None = None) -> str:
        """
        Generate formatted summary string.

        Parameters
        ----------
        notional : float | None
            Notional for bps calculation

        Returns
        -------
        str
            Formatted summary
        """
        lines = [
            "xVA Summary",
            "=" * 40,
            f"CVA:            ${self.cva:>15,.0f}",
            f"DVA (benefit):  ${-self.dva:>15,.0f}",
            f"FVA:            ${self.fva:>15,.0f}",
            f"MVA:            ${self.mva:>15,.0f}",
            f"KVA:            ${self.kva:>15,.0f}",
            "-" * 40,
            f"Total xVA:      ${self.total:>15,.0f}",
        ]

        if notional is not None and notional > 0:
            bps = self.to_bps(notional)
            lines.extend(
                [
                    "",
                    f"Total (bps):    {bps['total']:>15.1f}",
                ]
            )

        return "\n".join(lines)


@dataclass
class XVAParams:
    """
    Parameters for xVA calculation.

    Attributes
    ----------
    lgd_counterparty : float
        Counterparty loss given default
    lgd_own : float
        Own loss given default
    hazard_rate_counterparty : float
        Counterparty hazard rate
    hazard_rate_own : float
        Own hazard rate
    funding_spread : float
        Funding spread over OIS
    cost_of_capital : float
        Hurdle rate for capital
    capital_ratio : float
        Capital ratio for KVA
    im_multiplier : float
        IM as multiple of EPE
    """

    lgd_counterparty: float = 0.60
    lgd_own: float = 0.60
    hazard_rate_counterparty: float = 0.012
    hazard_rate_own: float = 0.010
    funding_spread: float = 0.01
    cost_of_capital: float = 0.10
    capital_ratio: float = 0.08
    im_multiplier: float = 1.5


def calculate_all_xva(
    epe: FloatArray,
    ene: FloatArray,
    discount_factors: FloatArray,
    time_grid: FloatArray,
    params: XVAParams | None = None,
    ead_profile: FloatArray | None = None,
    im_profile: FloatArray | None = None,
) -> XVAResult:
    """
    Calculate all xVA metrics.

    Parameters
    ----------
    epe : FloatArray
        Expected positive exposure profile
    ene : FloatArray
        Expected negative exposure profile
    discount_factors : FloatArray
        Discount factors
    time_grid : FloatArray
        Time grid in years
    params : XVAParams | None
        Calculation parameters (uses defaults if None)
    ead_profile : FloatArray | None
        EAD profile for KVA (computed from EPE if None)
    im_profile : FloatArray | None
        IM profile for MVA (computed from EPE if None)

    Returns
    -------
    XVAResult
        All xVA components

    Example
    -------
    >>> params = XVAParams(lgd_counterparty=0.6, hazard_rate_counterparty=0.012)
    >>> result = calculate_all_xva(epe, ene, df, time_grid, params)
    >>> print(f"Total xVA: ${result.total:,.0f}")
    """
    if params is None:
        params = XVAParams()

    # CVA
    cva_calc = CVACalculator(
        lgd=params.lgd_counterparty,
        hazard_rate=params.hazard_rate_counterparty,
    )
    cva = cva_calc.calculate(epe, discount_factors, time_grid)

    # DVA
    dva_calc = DVACalculator(
        lgd=params.lgd_own,
        hazard_rate=params.hazard_rate_own,
    )
    dva = dva_calc.calculate(ene, discount_factors, time_grid)

    # FVA
    fva_calc = FVACalculator(funding_spread=params.funding_spread)
    fva = fva_calc.calculate(epe, discount_factors, time_grid)

    # MVA
    if im_profile is None:
        im_profile = params.im_multiplier * epe
    mva_calc = MVACalculator(funding_spread=params.funding_spread)
    mva = mva_calc.calculate(im_profile, discount_factors, time_grid)

    # KVA
    if ead_profile is None:
        ead_profile = 1.4 * epe  # Alpha = 1.4 for SA-CCR
    kva_calc = KVACalculator(
        cost_of_capital=params.cost_of_capital,
        capital_ratio=params.capital_ratio,
    )
    kva = kva_calc.calculate(ead_profile, discount_factors, time_grid)

    return XVAResult(cva=cva, dva=dva, fva=fva, mva=mva, kva=kva)
