"""
Regulatory capital calculation module.

Provides simplified SA-CCR (Standardized Approach for Counterparty Credit Risk)
calculations as specified in Basel III.
"""

from xva_core.reg.saccr import (
    SACCRCalculator,
    SACCRResult,
    calculate_fx_addon,
    calculate_ir_addon,
    calculate_saccr_ead,
)

__all__ = [
    "SACCRCalculator",
    "SACCRResult",
    "calculate_saccr_ead",
    "calculate_ir_addon",
    "calculate_fx_addon",
]
