"""
Tests for exposure calculation module.
"""

import numpy as np
import pytest

from xva_core.exposure import (
    ExposureMetrics,
    MonteCarloEngine,
    NettingSet,
    calculate_ene,
    calculate_epe,
    calculate_pfe,
)
from xva_core.instruments import IRSwap


class TestExposureMetrics:
    """Tests for exposure metric calculations."""

    def test_epe_positive(self, sample_exposure: np.ndarray) -> None:
        """EPE should be non-negative."""
        epe = calculate_epe(sample_exposure)
        assert np.all(epe >= 0)

    def test_ene_positive(self, sample_exposure: np.ndarray) -> None:
        """ENE should be non-negative."""
        ene = calculate_ene(sample_exposure)
        assert np.all(ene >= 0)

    def test_pfe_higher_than_epe(self, sample_exposure: np.ndarray) -> None:
        """95% PFE should be >= EPE at each time."""
        epe = calculate_epe(sample_exposure)
        pfe_95 = calculate_pfe(sample_exposure, quantile=0.95)
        assert np.all(pfe_95 >= epe)

    def test_pfe_99_higher_than_95(self, sample_exposure: np.ndarray) -> None:
        """99% PFE should be >= 95% PFE."""
        pfe_95 = calculate_pfe(sample_exposure, quantile=0.95)
        pfe_99 = calculate_pfe(sample_exposure, quantile=0.99)
        assert np.all(pfe_99 >= pfe_95)

    def test_exposure_metrics_from_mtm(
        self, sample_exposure: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """ExposureMetrics.from_mtm should calculate all metrics."""
        metrics = ExposureMetrics.from_mtm(sample_exposure, time_grid)

        assert len(metrics.epe) == len(time_grid)
        assert len(metrics.ene) == len(time_grid)
        assert metrics.peak_epe >= 0
        assert metrics.average_epe >= 0


class TestNettingSet:
    """Tests for netting set functionality."""

    def test_netting_set_creation(
        self, sample_swap: IRSwap, sample_receiver_swap: IRSwap
    ) -> None:
        """Test netting set initialization."""
        ns = NettingSet(instruments=[sample_swap, sample_receiver_swap], name="Test")
        assert ns.n_instruments == 2
        assert ns.name == "Test"

    def test_total_notional(
        self, sample_swap: IRSwap, sample_receiver_swap: IRSwap
    ) -> None:
        """Total notional should be sum of instrument notionals."""
        ns = NettingSet(instruments=[sample_swap, sample_receiver_swap])
        expected = sample_swap.notional + sample_receiver_swap.notional
        assert ns.total_notional == expected

    def test_max_maturity(
        self, sample_swap: IRSwap, sample_receiver_swap: IRSwap
    ) -> None:
        """Max maturity should be the longest trade."""
        ns = NettingSet(instruments=[sample_swap, sample_receiver_swap])
        assert ns.max_maturity == max(
            sample_swap.maturity, sample_receiver_swap.maturity
        )

    def test_empty_netting_set(self) -> None:
        """Empty netting set should have zero notional."""
        ns = NettingSet()
        assert ns.n_instruments == 0
        assert ns.total_notional == 0
        assert ns.max_maturity == 0


class TestMonteCarloEngine:
    """Tests for Monte Carlo simulation engine."""

    def test_engine_creation(self) -> None:
        """Test engine initialization."""
        engine = MonteCarloEngine(n_paths=1000, horizon=5.0, dt=0.25)
        assert engine.n_paths == 1000
        assert engine.horizon == 5.0
        assert engine.n_steps == 21  # 5/0.25 + 1

    def test_time_grid_correct(self) -> None:
        """Time grid should span [0, horizon]."""
        engine = MonteCarloEngine(n_paths=100, horizon=5.0, dt=0.25)
        assert engine.time_grid[0] == 0.0
        assert engine.time_grid[-1] == 5.0

    def test_simulation_output_shape(self, market_config, sample_swap: IRSwap) -> None:
        """Simulation should return correct shapes."""
        engine = MonteCarloEngine(n_paths=100, horizon=5.0, dt=0.25, seed=42)
        result = engine.simulate([sample_swap], market_config)

        assert result.n_paths == 100
        assert result.n_steps == 21
        assert result.mtm.shape == (100, 21)
        assert result.r_domestic.shape == (100, 21)
        assert result.fx_spot.shape == (100, 21)

    def test_simulation_reproducible(self, market_config, sample_swap: IRSwap) -> None:
        """Same seed should give same results."""
        engine = MonteCarloEngine(n_paths=100, horizon=5.0, dt=0.25, seed=42)

        result1 = engine.simulate([sample_swap], market_config)
        result2 = engine.simulate([sample_swap], market_config)

        # Need to reset seed for second run
        engine2 = MonteCarloEngine(n_paths=100, horizon=5.0, dt=0.25, seed=42)
        result3 = engine2.simulate([sample_swap], market_config)

        assert np.allclose(result1.mtm, result3.mtm)

    def test_invalid_paths_raises(self) -> None:
        """Invalid n_paths should raise error."""
        with pytest.raises(ValueError):
            MonteCarloEngine(n_paths=0, horizon=5.0)

    def test_invalid_horizon_raises(self) -> None:
        """Invalid horizon should raise error."""
        with pytest.raises(ValueError):
            MonteCarloEngine(n_paths=100, horizon=-1.0)
