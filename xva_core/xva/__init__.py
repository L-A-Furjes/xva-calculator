"""
xVA calculation module.

Provides calculators for all valuation adjustments:
- CVA: Credit Valuation Adjustment
- DVA: Debt Valuation Adjustment
- FVA: Funding Valuation Adjustment
- MVA: Margin Valuation Adjustment
- KVA: Capital Valuation Adjustment
"""

from xva_core.xva.cva import CVACalculator
from xva_core.xva.dva import DVACalculator
from xva_core.xva.fva import FVACalculator
from xva_core.xva.kva import KVACalculator
from xva_core.xva.mva import MVACalculator
from xva_core.xva.result import XVAResult, calculate_all_xva

__all__ = [
    "CVACalculator",
    "DVACalculator",
    "FVACalculator",
    "MVACalculator",
    "KVACalculator",
    "XVAResult",
    "calculate_all_xva",
]
