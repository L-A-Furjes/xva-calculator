"""
Tests for market module: curves, models, and correlation.
"""

import numpy as np
import pytest

from xva_core.market import (
    CholeskyCorrelation,
    DiscountCurve,
    GBMFXModel,
    HazardCurve,
    OUShortRateModel,
)


class TestDiscountCurve:
    """Tests for DiscountCurve class."""

    def test_flat_curve_discount_factor(
        self, flat_discount_curve: DiscountCurve
    ) -> None:
        """Test flat curve discount factor calculation."""
        df_1y = flat_discount_curve.discount_factor(1.0)
        expected = np.exp(-0.02 * 1.0)
        assert np.isclose(df_1y, expected, rtol=1e-10)

    def test_discount_factor_at_zero(self, flat_discount_curve: DiscountCurve) -> None:
        """Discount factor at t=0 should be 1."""
        assert flat_discount_curve.discount_factor(0.0) == 1.0

    def test_discount_factor_decreasing(
        self, flat_discount_curve: DiscountCurve
    ) -> None:
        """Discount factors should decrease with time."""
        df_1y = flat_discount_curve.discount_factor(1.0)
        df_5y = flat_discount_curve.discount_factor(5.0)
        assert df_5y < df_1y

    def test_forward_rate(self, flat_discount_curve: DiscountCurve) -> None:
        """Forward rate for flat curve should equal spot rate."""
        fwd = flat_discount_curve.forward_rate(1.0, 2.0)
        assert np.isclose(fwd, 0.02, rtol=1e-6)


class TestHazardCurve:
    """Tests for HazardCurve class."""

    def test_survival_probability(self, hazard_curve: HazardCurve) -> None:
        """Test survival probability calculation."""
        surv_5y = hazard_curve.survival_probability(5.0)
        expected = np.exp(-0.012 * 5.0)
        assert np.isclose(surv_5y, expected, rtol=1e-10)

    def test_survival_at_zero(self, hazard_curve: HazardCurve) -> None:
        """Survival probability at t=0 should be 1."""
        assert hazard_curve.survival_probability(0.0) == 1.0

    def test_incremental_default_probabilities_sum(
        self, hazard_curve: HazardCurve, time_grid: np.ndarray
    ) -> None:
        """Sum of incremental PDs should equal total PD."""
        inc_pd = hazard_curve.incremental_default_probabilities(time_grid)
        total_pd = 1.0 - hazard_curve.survival_probability(time_grid[-1])
        assert np.isclose(inc_pd.sum(), total_pd, rtol=1e-6)

    def test_lgd_property(self, hazard_curve: HazardCurve) -> None:
        """LGD should be 1 - recovery rate."""
        assert hazard_curve.lgd == 0.6


class TestOUShortRateModel:
    """Tests for Ornstein-Uhlenbeck short-rate model."""

    def test_expected_rate_convergence(self, ou_model: OUShortRateModel) -> None:
        """Expected rate should converge to theta."""
        expected_10y = ou_model.expected_rate(10.0)
        expected_50y = ou_model.expected_rate(50.0)
        assert np.isclose(expected_50y, ou_model.theta, rtol=1e-3)

    def test_simulation_shape(
        self, ou_model: OUShortRateModel, time_grid: np.ndarray
    ) -> None:
        """Simulation output should have correct shape."""
        paths = ou_model.simulate(n_paths=100, time_grid=time_grid, seed=42)
        assert paths.shape == (100, len(time_grid))

    def test_simulation_starts_at_r0(
        self, ou_model: OUShortRateModel, time_grid: np.ndarray
    ) -> None:
        """All paths should start at r0."""
        paths = ou_model.simulate(n_paths=100, time_grid=time_grid, seed=42)
        assert np.all(paths[:, 0] == ou_model.r0)

    def test_zero_vol_converges_deterministically(self) -> None:
        """With sigma=0, paths should be deterministic."""
        model = OUShortRateModel(kappa=0.5, theta=0.03, sigma=0.0, r0=0.01)
        time_grid = np.linspace(0, 10, 100)
        paths = model.simulate(n_paths=10, time_grid=time_grid, seed=42)

        # All paths should be identical
        assert np.allclose(paths[0], paths[-1])

        # Should converge to theta
        assert np.isclose(paths[0, -1], 0.03, rtol=1e-3)


class TestGBMFXModel:
    """Tests for GBM FX model."""

    def test_simulation_starts_at_S0(self, fx_model: GBMFXModel) -> None:
        """All paths should start at S0."""
        n_paths, n_steps = 100, 21
        r_d = np.full((n_paths, n_steps), 0.02)
        r_f = np.full((n_paths, n_steps), 0.015)
        time_grid = np.linspace(0, 5, n_steps)

        spots = fx_model.simulate(r_d, r_f, time_grid, seed=42)
        assert np.all(spots[:, 0] == fx_model.S0)

    def test_simulation_positive(self, fx_model: GBMFXModel) -> None:
        """FX rates should always be positive."""
        n_paths, n_steps = 100, 21
        r_d = np.full((n_paths, n_steps), 0.02)
        r_f = np.full((n_paths, n_steps), 0.015)
        time_grid = np.linspace(0, 5, n_steps)

        spots = fx_model.simulate(r_d, r_f, time_grid, seed=42)
        assert np.all(spots > 0)


class TestCholeskyCorrelation:
    """Tests for correlation handling."""

    def test_correlation_matrix_symmetric(
        self, correlation: CholeskyCorrelation
    ) -> None:
        """Correlation matrix should be symmetric."""
        corr_mat = correlation.correlation_matrix
        assert np.allclose(corr_mat, corr_mat.T)

    def test_correlation_diagonal_ones(self, correlation: CholeskyCorrelation) -> None:
        """Diagonal of correlation matrix should be 1."""
        corr_mat = correlation.correlation_matrix
        assert np.allclose(np.diag(corr_mat), 1.0)

    def test_correlated_samples_empirical(
        self, correlation: CholeskyCorrelation
    ) -> None:
        """Empirical correlation should match target."""
        z_d, z_f, z_x = correlation.generate_correlated_samples(
            n_paths=50000, n_steps=1, seed=42
        )

        empirical_corr = np.corrcoef(z_d[:, 0], z_f[:, 0])[0, 1]
        assert np.isclose(empirical_corr, correlation.rho_df, atol=0.05)

    def test_invalid_correlation_raises(self) -> None:
        """Invalid correlation matrix should raise error."""
        with pytest.raises(ValueError):
            # These correlations don't form a valid positive-definite matrix
            CholeskyCorrelation(rho_df=0.99, rho_dx=0.99, rho_fx=-0.99)
