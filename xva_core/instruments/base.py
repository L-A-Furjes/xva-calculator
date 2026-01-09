"""
Base classes for financial instruments.

Provides the abstract interface that all instruments must implement
for exposure calculation.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np

from xva_core._types import FloatArray, PathArray


class InstrumentType(Enum):
    """Enumeration of supported instrument types."""

    IRS = "interest_rate_swap"
    FX_FORWARD = "fx_forward"
    FX_OPTION = "fx_option"  # Future extension


class Instrument(ABC):
    """
    Abstract base class for all financial instruments.

    All instruments must implement the `calculate_mtm` method to
    compute mark-to-market values along simulated paths.

    Attributes
    ----------
    notional : float
        Notional amount
    maturity : float
        Time to maturity in years
    instrument_type : InstrumentType
        Type of instrument for classification

    Methods
    -------
    calculate_mtm(time_idx, paths_data)
        Calculate MTM at a specific time step for all paths
    is_expired(t)
        Check if instrument has expired at time t
    """

    notional: float
    maturity: float
    instrument_type: InstrumentType

    @abstractmethod
    def calculate_mtm(
        self,
        time_idx: int,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> FloatArray:
        """
        Calculate mark-to-market value at a time step.

        Parameters
        ----------
        time_idx : int
            Index into the time grid
        time_grid : FloatArray
            Array of time points in years
        paths_data : dict[str, PathArray]
            Dictionary containing simulated paths:
            - 'r_domestic': Domestic short rate paths (n_paths, n_steps)
            - 'r_foreign': Foreign short rate paths (n_paths, n_steps)
            - 'fx_spot': FX spot paths (n_paths, n_steps)
            - 'df_domestic': Domestic discount factors (n_paths, n_steps)
            - 'df_foreign': Foreign discount factors (n_paths, n_steps)

        Returns
        -------
        FloatArray
            MTM values for each path at the specified time, shape (n_paths,)
        """
        pass

    def is_expired(self, t: float) -> bool:
        """
        Check if instrument has expired at time t.

        Parameters
        ----------
        t : float
            Current time in years

        Returns
        -------
        bool
            True if t >= maturity
        """
        return t >= self.maturity

    @abstractmethod
    def get_cash_flow_dates(self) -> FloatArray:
        """
        Get all cash flow dates for the instrument.

        Returns
        -------
        FloatArray
            Array of cash flow dates in years
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Convert instrument to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the instrument
        """
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"notional={self.notional:,.0f}, "
            f"maturity={self.maturity:.2f}Y)"
        )


def compute_forward_discount_factors(
    cumulative_df: PathArray,
    start_idx: int,
    end_indices: FloatArray | list[int],
) -> PathArray:
    """
    Compute forward discount factors from cumulative DFs.

    DF(t1, t2) = DF(0, t2) / DF(0, t1)

    Parameters
    ----------
    cumulative_df : PathArray
        Cumulative discount factors from t=0, shape (n_paths, n_steps)
    start_idx : int
        Index of the start time
    end_indices : array-like
        Indices of end times

    Returns
    -------
    PathArray
        Forward discount factors, shape (n_paths, len(end_indices))
    """
    df_start = cumulative_df[:, start_idx : start_idx + 1]  # Keep 2D
    df_ends = cumulative_df[:, end_indices]

    # Avoid division by zero
    df_start = np.maximum(df_start, 1e-10)

    return df_ends / df_start
