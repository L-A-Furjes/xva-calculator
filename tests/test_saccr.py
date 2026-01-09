"""
Tests for SA-CCR regulatory capital calculation.
"""

import numpy as np
import pytest

from xva_core.instruments import FXForward, IRSwap
from xva_core.reg.saccr import (
    SACCRCalculator,
    calculate_fx_addon,
    calculate_ir_addon,
    calculate_saccr_ead,
)


class TestSACCRCalculator:
    """Tests for SA-CCR EAD calculation."""

    def test_saccr_creation(self) -> None:
        """Test calculator initialization."""
        calc = SACCRCalculator(alpha=1.4)
        assert calc.alpha == 1.4

    def test_ead_positive(self, sample_swap: IRSwap) -> None:
        """EAD should be positive."""
        calc = SACCRCalculator()
        result = calc.calculate([sample_swap], current_mtm=1e6)
        assert result.ead > 0

    def test_ead_includes_rc(self, sample_swap: IRSwap) -> None:
        """EAD should include replacement cost."""
        calc = SACCRCalculator()
        current_mtm = 2e6

        result = calc.calculate([sample_swap], current_mtm=current_mtm)

        # RC should be floored at 0
        assert result.replacement_cost == max(current_mtm, 0)

    def test_negative_mtm_zero_rc(self, sample_swap: IRSwap) -> None:
        """Negative MTM should give RC = 0."""
        calc = SACCRCalculator()
        result = calc.calculate([sample_swap], current_mtm=-1e6)
        assert result.replacement_cost == 0

    def test_collateral_reduces_rc(self, sample_swap: IRSwap) -> None:
        """Collateral should reduce RC."""
        calc = SACCRCalculator()
        current_mtm = 2e6
        collateral = 1e6

        result_no_coll = calc.calculate([sample_swap], current_mtm=current_mtm, collateral=0)
        result_coll = calc.calculate(
            [sample_swap], current_mtm=current_mtm, collateral=collateral
        )

        assert result_coll.replacement_cost < result_no_coll.replacement_cost

    def test_alpha_multiplier(self, sample_swap: IRSwap) -> None:
        """EAD should scale with alpha."""
        calc_14 = SACCRCalculator(alpha=1.4)
        calc_20 = SACCRCalculator(alpha=2.0)

        result_14 = calc_14.calculate([sample_swap], current_mtm=1e6)
        result_20 = calc_20.calculate([sample_swap], current_mtm=1e6)

        # Both should have same RC and PFE, so EAD ratio = alpha ratio
        expected_ratio = 2.0 / 1.4
        actual_ratio = result_20.ead / result_14.ead
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-6)

    def test_trade_addon_breakdown(
        self, sample_swap: IRSwap, sample_fx_forward: FXForward
    ) -> None:
        """Should have per-trade add-on breakdown."""
        calc = SACCRCalculator()
        result = calc.calculate([sample_swap, sample_fx_forward], current_mtm=0)

        assert len(result.trade_addons) == 2
        assert result.trade_addons[0].asset_class.value in ["IR", "FX"]

    def test_multiplier_bounds(self, sample_swap: IRSwap) -> None:
        """Multiplier should be in [floor, 1]."""
        calc = SACCRCalculator()

        # Positive MTM -> multiplier = 1
        result_pos = calc.calculate([sample_swap], current_mtm=1e6)
        assert result_pos.multiplier == 1.0

        # Very negative MTM -> multiplier approaches floor (0.05)
        result_neg = calc.calculate([sample_swap], current_mtm=-100e6)
        assert 0.05 <= result_neg.multiplier <= 1.0


class TestAddOnCalculations:
    """Tests for individual add-on calculations."""

    def test_ir_addon_scales_with_notional(self) -> None:
        """IR add-on should scale with notional."""
        addon_10m = calculate_ir_addon(notional=10e6, maturity=5.0)
        addon_20m = calculate_ir_addon(notional=20e6, maturity=5.0)

        assert np.isclose(addon_20m / addon_10m, 2.0, rtol=1e-6)

    def test_ir_addon_maturity_factor(self) -> None:
        """Longer maturity should give higher add-on."""
        addon_1y = calculate_ir_addon(notional=10e6, maturity=1.0)
        addon_5y = calculate_ir_addon(notional=10e6, maturity=5.0)

        assert addon_5y > addon_1y

    def test_fx_addon_scales_with_notional(self) -> None:
        """FX add-on should scale with notional."""
        addon_10m = calculate_fx_addon(notional_domestic=10e6)
        addon_20m = calculate_fx_addon(notional_domestic=20e6)

        assert np.isclose(addon_20m / addon_10m, 2.0, rtol=1e-6)

    def test_fx_addon_supervisory_factor(self) -> None:
        """FX add-on should use correct SF (4%)."""
        notional = 10e6
        addon = calculate_fx_addon(notional_domestic=notional)
        expected = notional * 0.04
        assert np.isclose(addon, expected, rtol=1e-6)


class TestConvenienceFunction:
    """Tests for calculate_saccr_ead convenience function."""

    def test_returns_ead(self, sample_swap: IRSwap) -> None:
        """Should return EAD value."""
        ead = calculate_saccr_ead([sample_swap], current_mtm=1e6)
        assert isinstance(ead, float)
        assert ead > 0

    def test_matches_calculator(self, sample_swap: IRSwap) -> None:
        """Should match SACCRCalculator result."""
        ead = calculate_saccr_ead([sample_swap], current_mtm=1e6, alpha=1.4)

        calc = SACCRCalculator(alpha=1.4)
        result = calc.calculate([sample_swap], current_mtm=1e6)

        assert np.isclose(ead, result.ead, rtol=1e-6)
