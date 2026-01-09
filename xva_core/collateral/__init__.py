"""
Collateral management module for xVA calculations.

Provides variation margin (VM) and initial margin (IM) calculations
with support for:
- Collateral thresholds and minimum transfer amounts
- Margin period of risk (MPR) lag
- Two-way collateral posting
"""

from xva_core.collateral.im import InitialMargin
from xva_core.collateral.vm import VariationMargin, apply_vm_to_exposure

__all__ = [
    "VariationMargin",
    "InitialMargin",
    "apply_vm_to_exposure",
]
