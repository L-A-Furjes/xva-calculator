"""
Netting set management for portfolio exposure calculation.

A netting set is a group of trades that can be netted in case of
counterparty default under a master agreement (e.g., ISDA).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from xva_core._types import FloatArray, PathArray
from xva_core.instruments.base import Instrument


@dataclass
class NettingSet:
    """
    A netting set containing trades that can be netted.

    Under bilateral netting agreements, if a counterparty defaults,
    all trades in the netting set are valued and only the net amount
    is owed (positive or negative).

    Attributes
    ----------
    instruments : list[Instrument]
        List of instruments in the netting set
    name : str
        Name/identifier for the netting set

    Example
    -------
    >>> swap1 = IRSwap(notional=1e7, fixed_rate=0.02, maturity=5.0)
    >>> swap2 = IRSwap(notional=1e7, fixed_rate=0.025, maturity=3.0, pay_fixed=False)
    >>> netting_set = NettingSet(instruments=[swap1, swap2], name="CPTY_A")
    """

    instruments: list[Instrument] = field(default_factory=list)
    name: str = "Default"

    def add_instrument(self, instrument: Instrument) -> None:
        """
        Add an instrument to the netting set.

        Parameters
        ----------
        instrument : Instrument
            Instrument to add
        """
        self.instruments.append(instrument)

    def remove_instrument(self, instrument: Instrument) -> bool:
        """
        Remove an instrument from the netting set.

        Parameters
        ----------
        instrument : Instrument
            Instrument to remove

        Returns
        -------
        bool
            True if instrument was found and removed
        """
        try:
            self.instruments.remove(instrument)
            return True
        except ValueError:
            return False

    @property
    def n_instruments(self) -> int:
        """Number of instruments in the netting set."""
        return len(self.instruments)

    @property
    def total_notional(self) -> float:
        """Total notional across all instruments."""
        return sum(inst.notional for inst in self.instruments)

    @property
    def max_maturity(self) -> float:
        """Maximum maturity across all instruments."""
        if not self.instruments:
            return 0.0
        return max(inst.maturity for inst in self.instruments)

    def calculate_net_mtm(
        self,
        time_idx: int,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> FloatArray:
        """
        Calculate net MTM for the netting set at a time step.

        Parameters
        ----------
        time_idx : int
            Time index
        time_grid : FloatArray
            Time grid in years
        paths_data : dict
            Market data paths

        Returns
        -------
        FloatArray
            Net MTM for each path, shape (n_paths,)
        """
        n_paths = paths_data["df_domestic"].shape[0]
        net_mtm = np.zeros(n_paths)

        for instrument in self.instruments:
            mtm = instrument.calculate_mtm(time_idx, time_grid, paths_data)
            net_mtm += mtm

        return net_mtm

    def calculate_mtm_matrix(
        self,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> PathArray:
        """
        Calculate net MTM for all time steps.

        Parameters
        ----------
        time_grid : FloatArray
            Time grid
        paths_data : dict
            Market data paths

        Returns
        -------
        PathArray
            Net MTM matrix, shape (n_paths, n_steps)
        """
        n_paths = paths_data["df_domestic"].shape[0]
        n_steps = len(time_grid)
        mtm_matrix = np.zeros((n_paths, n_steps))

        for time_idx in range(n_steps):
            mtm_matrix[:, time_idx] = self.calculate_net_mtm(
                time_idx, time_grid, paths_data
            )

        return mtm_matrix

    def gross_mtm(
        self,
        time_idx: int,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> tuple[FloatArray, FloatArray]:
        """
        Calculate gross positive and negative MTM.

        Parameters
        ----------
        time_idx : int
            Time index
        time_grid : FloatArray
            Time grid
        paths_data : dict
            Market data

        Returns
        -------
        tuple[FloatArray, FloatArray]
            (gross_positive, gross_negative) MTM for each path
        """
        n_paths = paths_data["df_domestic"].shape[0]
        gross_positive = np.zeros(n_paths)
        gross_negative = np.zeros(n_paths)

        for instrument in self.instruments:
            mtm = instrument.calculate_mtm(time_idx, time_grid, paths_data)
            gross_positive += np.maximum(mtm, 0)
            gross_negative += np.maximum(-mtm, 0)

        return gross_positive, gross_negative

    def net_to_gross_ratio(
        self,
        time_idx: int,
        time_grid: FloatArray,
        paths_data: dict[str, PathArray],
    ) -> float:
        """
        Calculate average net-to-gross ratio.

        NGR = Net exposure / Gross positive exposure

        This measures the benefit of netting. NGR = 1 means no benefit
        (single trade), NGR < 1 means exposure reduced by netting.

        Parameters
        ----------
        time_idx : int
            Time index
        time_grid : FloatArray
            Time grid
        paths_data : dict
            Market data

        Returns
        -------
        float
            Average NGR across paths
        """
        net_mtm = self.calculate_net_mtm(time_idx, time_grid, paths_data)
        gross_pos, _ = self.gross_mtm(time_idx, time_grid, paths_data)

        net_exposure = np.maximum(net_mtm, 0).mean()
        gross_exposure = gross_pos.mean()

        if gross_exposure < 1e-10:
            return 1.0

        return net_exposure / gross_exposure

    @classmethod
    def from_instruments(
        cls,
        instruments: Sequence[Instrument],
        name: str = "Default",
    ) -> "NettingSet":
        """
        Create netting set from a sequence of instruments.

        Parameters
        ----------
        instruments : Sequence[Instrument]
            Instruments to include
        name : str
            Name for the netting set

        Returns
        -------
        NettingSet
            New netting set
        """
        return cls(instruments=list(instruments), name=name)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NettingSet(name='{self.name}', "
            f"n_instruments={self.n_instruments}, "
            f"total_notional={self.total_notional:,.0f})"
        )
