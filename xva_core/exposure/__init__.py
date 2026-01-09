"""
Exposure calculation module for xVA.

Provides Monte Carlo simulation engine, netting set aggregation,
and exposure metrics (EPE, ENE, PFE).
"""

from xva_core.exposure.metrics import (
    ExposureMetrics,
    calculate_ene,
    calculate_epe,
    calculate_expected_exposure,
    calculate_pfe,
)
from xva_core.exposure.netting import NettingSet
from xva_core.exposure.simulator import MonteCarloEngine, SimulationResult

__all__ = [
    "MonteCarloEngine",
    "SimulationResult",
    "NettingSet",
    "ExposureMetrics",
    "calculate_epe",
    "calculate_ene",
    "calculate_pfe",
    "calculate_expected_exposure",
]
