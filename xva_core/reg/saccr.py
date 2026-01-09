"""
SA-CCR (Standardized Approach for Counterparty Credit Risk) implementation.

This is a simplified implementation of the Basel III SA-CCR framework
for calculating regulatory Exposure at Default (EAD).

EAD = α × (RC + PFE)

where:
    α = 1.4 (regulatory multiplier)
    RC = replacement cost (current MTM, floored at 0)
    PFE = potential future exposure = multiplier × AddOn
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from xva_core._types import FloatArray
from xva_core.instruments.base import Instrument, InstrumentType


class AssetClass(Enum):
    """SA-CCR asset classes."""

    INTEREST_RATE = "IR"
    FX = "FX"
    CREDIT = "CR"
    EQUITY = "EQ"
    COMMODITY = "CO"


# Supervisory factors by asset class
SUPERVISORY_FACTORS = {
    AssetClass.INTEREST_RATE: 0.005,  # 0.5%
    AssetClass.FX: 0.04,  # 4%
    AssetClass.CREDIT: 0.05,  # 5% (default)
    AssetClass.EQUITY: 0.32,  # 32%
    AssetClass.COMMODITY: 0.18,  # 18%
}


@dataclass
class TradeAddOn:
    """Add-on calculation for a single trade."""

    trade_id: str
    asset_class: AssetClass
    notional: float
    maturity: float
    addon: float
    supervisory_factor: float


@dataclass
class SACCRResult:
    """
    Container for SA-CCR calculation results.

    Attributes
    ----------
    replacement_cost : float
        Current MTM (floored at 0)
    aggregate_addon : float
        Total add-on across all trades
    multiplier : float
        PFE multiplier (accounts for excess collateral)
    pfe : float
        Potential Future Exposure = multiplier × addon
    ead : float
        Exposure at Default = α × (RC + PFE)
    trade_addons : list[TradeAddOn]
        Per-trade add-on breakdown
    """

    replacement_cost: float
    aggregate_addon: float
    multiplier: float
    pfe: float
    ead: float
    trade_addons: list[TradeAddOn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "replacement_cost": self.replacement_cost,
            "aggregate_addon": self.aggregate_addon,
            "multiplier": self.multiplier,
            "pfe": self.pfe,
            "ead": self.ead,
            "n_trades": len(self.trade_addons),
        }

    def summary(self) -> str:
        """Generate formatted summary."""
        lines = [
            "SA-CCR Summary",
            "=" * 40,
            f"Replacement Cost (RC):  ${self.replacement_cost:>12,.0f}",
            f"Aggregate Add-On:       ${self.aggregate_addon:>12,.0f}",
            f"Multiplier:             {self.multiplier:>12.4f}",
            f"PFE:                    ${self.pfe:>12,.0f}",
            "-" * 40,
            f"EAD (α=1.4):            ${self.ead:>12,.0f}",
        ]
        return "\n".join(lines)


class SACCRCalculator:
    """
    SA-CCR EAD calculator.

    Implements the simplified Basel III Standardized Approach for
    Counterparty Credit Risk.

    Attributes
    ----------
    alpha : float
        Regulatory multiplier (default 1.4)

    Example
    -------
    >>> calc = SACCRCalculator()
    >>> result = calc.calculate(instruments, current_mtm=1_000_000)
    >>> print(f"EAD: ${result.ead:,.0f}")
    """

    def __init__(self, alpha: float = 1.4) -> None:
        """
        Initialize SA-CCR calculator.

        Parameters
        ----------
        alpha : float
            Regulatory multiplier (default 1.4)
        """
        if alpha < 1.0:
            raise ValueError(f"Alpha must be >= 1.0, got {alpha}")
        self.alpha = alpha

    def calculate(
        self,
        instruments: Sequence[Instrument],
        current_mtm: float,
        collateral: float = 0.0,
    ) -> SACCRResult:
        """
        Calculate SA-CCR EAD for a netting set.

        Parameters
        ----------
        instruments : Sequence[Instrument]
            Portfolio of instruments
        current_mtm : float
            Current MTM of the netting set
        collateral : float
            Collateral held (reduces RC)

        Returns
        -------
        SACCRResult
            Complete SA-CCR calculation results
        """
        # Replacement cost (floored at 0)
        rc = max(current_mtm - collateral, 0)

        # Calculate add-ons by trade
        trade_addons = []
        addon_by_class: dict[AssetClass, float] = {}

        for i, instrument in enumerate(instruments):
            asset_class = self._get_asset_class(instrument)
            addon = self._calculate_trade_addon(instrument, asset_class)

            trade_addons.append(
                TradeAddOn(
                    trade_id=f"Trade_{i+1}",
                    asset_class=asset_class,
                    notional=instrument.notional,
                    maturity=instrument.maturity,
                    addon=addon,
                    supervisory_factor=SUPERVISORY_FACTORS[asset_class],
                )
            )

            # Aggregate by asset class
            if asset_class not in addon_by_class:
                addon_by_class[asset_class] = 0.0
            addon_by_class[asset_class] += addon**2  # For geometric aggregation

        # Aggregate add-ons within each asset class (geometric)
        class_addons = {
            ac: np.sqrt(val) for ac, val in addon_by_class.items()
        }

        # Total add-on across classes (simple sum for this implementation)
        aggregate_addon = sum(class_addons.values())

        # Multiplier (simplified)
        multiplier = self._calculate_multiplier(current_mtm, collateral, aggregate_addon)

        # PFE
        pfe = multiplier * aggregate_addon

        # EAD
        ead = self.alpha * (rc + pfe)

        return SACCRResult(
            replacement_cost=rc,
            aggregate_addon=aggregate_addon,
            multiplier=multiplier,
            pfe=pfe,
            ead=ead,
            trade_addons=trade_addons,
        )

    def _get_asset_class(self, instrument: Instrument) -> AssetClass:
        """Determine asset class for an instrument."""
        if instrument.instrument_type == InstrumentType.IRS:
            return AssetClass.INTEREST_RATE
        elif instrument.instrument_type == InstrumentType.FX_FORWARD:
            return AssetClass.FX
        else:
            return AssetClass.INTEREST_RATE  # Default

    def _calculate_trade_addon(
        self,
        instrument: Instrument,
        asset_class: AssetClass,
    ) -> float:
        """
        Calculate add-on for a single trade.

        Parameters
        ----------
        instrument : Instrument
            The instrument
        asset_class : AssetClass
            Asset class for SF lookup

        Returns
        -------
        float
            Trade-level add-on
        """
        sf = SUPERVISORY_FACTORS[asset_class]
        notional = instrument.notional
        maturity = instrument.maturity

        if asset_class == AssetClass.INTEREST_RATE:
            # IR add-on includes maturity factor
            maturity_factor = self._maturity_factor(maturity)
            return notional * sf * maturity_factor

        elif asset_class == AssetClass.FX:
            # FX add-on is simpler
            return notional * sf

        else:
            # Default
            return notional * sf

    def _maturity_factor(self, maturity: float) -> float:
        """
        Calculate maturity factor for IR add-on.

        MF = sqrt(min(M, 1) / 1Y) for M < 1Y
        MF = sqrt(M / 1Y) for M >= 1Y, capped at sqrt(5)
        """
        if maturity < 1.0:
            return np.sqrt(max(maturity, 0.01))
        else:
            return np.sqrt(min(maturity, 5.0))

    def _calculate_multiplier(
        self,
        mtm: float,
        collateral: float,
        addon: float,
    ) -> float:
        """
        Calculate PFE multiplier.

        The multiplier reduces PFE when there is excess collateral
        or when MTM is negative.

        multiplier = min(1, floor + (1-floor) × exp(numerator / (2 × (1-floor) × addon)))

        where:
            floor = 0.05
            numerator = V - C (net MTM after collateral)
        """
        floor = 0.05

        if addon < 1e-10:
            return 1.0

        v_minus_c = mtm - collateral
        if v_minus_c >= 0:
            return 1.0

        numerator = v_minus_c
        denominator = 2 * (1 - floor) * addon

        if abs(denominator) < 1e-10:
            return 1.0

        multiplier = floor + (1 - floor) * np.exp(numerator / denominator)
        return min(max(multiplier, floor), 1.0)

    def calculate_ead_profile(
        self,
        instruments: Sequence[Instrument],
        mtm_path: FloatArray,
    ) -> FloatArray:
        """
        Calculate EAD profile over time from MTM path.

        Parameters
        ----------
        instruments : Sequence[Instrument]
            Portfolio instruments
        mtm_path : FloatArray
            MTM at each time step, shape (n_steps,)

        Returns
        -------
        FloatArray
            EAD at each time step
        """
        n_steps = len(mtm_path)
        ead_profile = np.zeros(n_steps)

        for t in range(n_steps):
            result = self.calculate(instruments, current_mtm=mtm_path[t])
            ead_profile[t] = result.ead

        return ead_profile


def calculate_ir_addon(
    notional: float,
    maturity: float,
    supervisory_factor: float = 0.005,
) -> float:
    """
    Calculate IR add-on for a single trade.

    Parameters
    ----------
    notional : float
        Trade notional
    maturity : float
        Time to maturity in years
    supervisory_factor : float
        SF for IR (default 0.5%)

    Returns
    -------
    float
        IR add-on
    """
    mf = np.sqrt(min(maturity, 5.0)) if maturity >= 1.0 else np.sqrt(max(maturity, 0.01))
    return notional * supervisory_factor * mf


def calculate_fx_addon(
    notional_domestic: float,
    supervisory_factor: float = 0.04,
) -> float:
    """
    Calculate FX add-on for a single trade.

    Parameters
    ----------
    notional_domestic : float
        Notional in domestic currency
    supervisory_factor : float
        SF for FX (default 4%)

    Returns
    -------
    float
        FX add-on
    """
    return notional_domestic * supervisory_factor


def calculate_saccr_ead(
    instruments: Sequence[Instrument],
    current_mtm: float,
    collateral: float = 0.0,
    alpha: float = 1.4,
) -> float:
    """
    Convenience function for SA-CCR EAD calculation.

    Parameters
    ----------
    instruments : Sequence[Instrument]
        Portfolio instruments
    current_mtm : float
        Current portfolio MTM
    collateral : float
        Collateral held
    alpha : float
        Regulatory multiplier

    Returns
    -------
    float
        EAD value
    """
    calc = SACCRCalculator(alpha=alpha)
    result = calc.calculate(instruments, current_mtm, collateral)
    return result.ead
