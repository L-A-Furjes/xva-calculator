"""
Financial instruments module for xVA calculations.

Provides pricing functions for:
- Interest Rate Swaps (IRS)
- FX Forwards

Each instrument can calculate its mark-to-market value at any point
along a simulated path, enabling exposure calculation.
"""

from xva_core.instruments.base import Instrument, InstrumentType
from xva_core.instruments.fxforward import FXForward
from xva_core.instruments.irs import IRSwap

__all__ = [
    "Instrument",
    "InstrumentType",
    "IRSwap",
    "FXForward",
]
