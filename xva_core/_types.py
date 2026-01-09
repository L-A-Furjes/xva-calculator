"""
Common type aliases used throughout the xVA calculation engine.

This module defines type aliases for numpy arrays and other common types
to improve code readability and enable better static type checking.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# Array type aliases
FloatArray: TypeAlias = npt.NDArray[np.float64]
"""1D or 2D array of 64-bit floats."""

IntArray: TypeAlias = npt.NDArray[np.int64]
"""1D or 2D array of 64-bit integers."""

PathArray: TypeAlias = npt.NDArray[np.float64]
"""
2D array of shape (n_paths, n_timesteps) representing Monte Carlo paths.

Each row is a single simulation path, and each column is a time step.
"""

TimeGrid: TypeAlias = npt.NDArray[np.float64]
"""1D array of time points in years."""

# Scalar type aliases
Rate: TypeAlias = float
"""Interest rate or spread as a decimal (e.g., 0.02 for 2%)."""

Notional: TypeAlias = float
"""Notional amount in base currency units."""

Year: TypeAlias = float
"""Time measured in years (e.g., 0.25 for quarterly)."""
