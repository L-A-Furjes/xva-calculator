"""
Tests for xVA calculations: CVA, DVA, FVA, MVA, KVA.
"""

import numpy as np

from xva_core.xva import (
    CVACalculator,
    DVACalculator,
    FVACalculator,
    KVACalculator,
    MVACalculator,
    XVAResult,
)
from xva_core.xva.result import XVAParams, calculate_all_xva


class TestCVACalculator:
    """Tests for CVA calculation."""

    def test_cva_positive(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """CVA should be positive for positive EPE."""
        epe = np.ones(len(time_grid)) * 1e6  # $1M constant EPE
        calc = CVACalculator(lgd=0.6, hazard_rate=0.012)
        cva = calc.calculate(epe, sample_discount_factors, time_grid)
        assert cva > 0

    def test_cva_scales_with_lgd(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """CVA should scale linearly with LGD."""
        epe = np.ones(len(time_grid)) * 1e6

        calc_60 = CVACalculator(lgd=0.60, hazard_rate=0.012)
        calc_40 = CVACalculator(lgd=0.40, hazard_rate=0.012)

        cva_60 = calc_60.calculate(epe, sample_discount_factors, time_grid)
        cva_40 = calc_40.calculate(epe, sample_discount_factors, time_grid)

        # Ratio should be 60/40 = 1.5
        ratio = cva_60 / cva_40
        assert np.isclose(ratio, 1.5, rtol=1e-6)

    def test_cva_zero_for_zero_epe(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """CVA should be zero when EPE is zero."""
        epe = np.zeros(len(time_grid))
        calc = CVACalculator(lgd=0.6, hazard_rate=0.012)
        cva = calc.calculate(epe, sample_discount_factors, time_grid)
        assert cva == 0

    def test_cva_scales_with_hazard_rate(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """Higher hazard rate should give higher CVA."""
        epe = np.ones(len(time_grid)) * 1e6

        calc_low = CVACalculator(lgd=0.6, hazard_rate=0.01)
        calc_high = CVACalculator(lgd=0.6, hazard_rate=0.02)

        cva_low = calc_low.calculate(epe, sample_discount_factors, time_grid)
        cva_high = calc_high.calculate(epe, sample_discount_factors, time_grid)

        assert cva_high > cva_low


class TestDVACalculator:
    """Tests for DVA calculation."""

    def test_dva_positive(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """DVA should be positive for positive ENE."""
        ene = np.ones(len(time_grid)) * 1e6
        calc = DVACalculator(lgd=0.6, hazard_rate=0.01)
        dva = calc.calculate(ene, sample_discount_factors, time_grid)
        assert dva > 0

    def test_dva_zero_for_zero_ene(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """DVA should be zero when ENE is zero."""
        ene = np.zeros(len(time_grid))
        calc = DVACalculator(lgd=0.6, hazard_rate=0.01)
        dva = calc.calculate(ene, sample_discount_factors, time_grid)
        assert dva == 0


class TestFVACalculator:
    """Tests for FVA calculation."""

    def test_fva_positive(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """FVA should be positive for positive EPE."""
        epe = np.ones(len(time_grid)) * 1e6
        calc = FVACalculator(funding_spread=0.01)
        fva = calc.calculate(epe, sample_discount_factors, time_grid)
        assert fva > 0

    def test_fva_scales_with_spread(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """FVA should scale linearly with funding spread."""
        epe = np.ones(len(time_grid)) * 1e6

        calc_100 = FVACalculator(funding_spread=0.01)  # 100 bps
        calc_50 = FVACalculator(funding_spread=0.005)  # 50 bps

        fva_100 = calc_100.calculate(epe, sample_discount_factors, time_grid)
        fva_50 = calc_50.calculate(epe, sample_discount_factors, time_grid)

        ratio = fva_100 / fva_50
        assert np.isclose(ratio, 2.0, rtol=1e-6)


class TestMVACalculator:
    """Tests for MVA calculation."""

    def test_mva_positive(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """MVA should be positive for positive IM."""
        im = np.ones(len(time_grid)) * 1e6
        calc = MVACalculator(funding_spread=0.01)
        mva = calc.calculate(im, sample_discount_factors, time_grid)
        assert mva > 0


class TestKVACalculator:
    """Tests for KVA calculation."""

    def test_kva_positive(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """KVA should be positive for positive EAD."""
        ead = np.ones(len(time_grid)) * 1e7
        calc = KVACalculator(cost_of_capital=0.10, capital_ratio=0.08)
        kva = calc.calculate(ead, sample_discount_factors, time_grid)
        assert kva > 0

    def test_kva_scales_with_capital_ratio(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """KVA should scale with capital ratio."""
        ead = np.ones(len(time_grid)) * 1e7

        calc_8 = KVACalculator(cost_of_capital=0.10, capital_ratio=0.08)
        calc_4 = KVACalculator(cost_of_capital=0.10, capital_ratio=0.04)

        kva_8 = calc_8.calculate(ead, sample_discount_factors, time_grid)
        kva_4 = calc_4.calculate(ead, sample_discount_factors, time_grid)

        ratio = kva_8 / kva_4
        assert np.isclose(ratio, 2.0, rtol=1e-6)


class TestXVAResult:
    """Tests for XVAResult container."""

    def test_total_calculation(self) -> None:
        """Total xVA should be CVA - DVA + FVA + MVA + KVA."""
        result = XVAResult(cva=100, dva=30, fva=50, mva=20, kva=40)
        expected = 100 - 30 + 50 + 20 + 40  # 180
        assert result.total == expected

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        result = XVAResult(cva=100, dva=30, fva=50, mva=20, kva=40)
        d = result.to_dict()
        assert d["cva"] == 100
        assert d["total"] == 180

    def test_to_bps(self) -> None:
        """Test basis points conversion."""
        result = XVAResult(cva=1e6, dva=0, fva=0, mva=0, kva=0)
        bps = result.to_bps(notional=1e9)
        assert np.isclose(bps["cva"], 10.0)  # 1M / 1B * 10000 = 10 bps


class TestCalculateAllXVA:
    """Tests for combined xVA calculation."""

    def test_all_xva_returns_result(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """calculate_all_xva should return XVAResult."""
        epe = np.ones(len(time_grid)) * 1e6
        ene = np.ones(len(time_grid)) * 5e5

        result = calculate_all_xva(
            epe=epe,
            ene=ene,
            discount_factors=sample_discount_factors,
            time_grid=time_grid,
        )

        assert isinstance(result, XVAResult)
        assert result.cva > 0
        assert result.dva > 0
        assert result.fva > 0

    def test_custom_params(
        self, sample_discount_factors: np.ndarray, time_grid: np.ndarray
    ) -> None:
        """Custom parameters should affect results."""
        epe = np.ones(len(time_grid)) * 1e6
        ene = np.ones(len(time_grid)) * 5e5

        params_high = XVAParams(lgd_counterparty=0.8, hazard_rate_counterparty=0.02)
        params_low = XVAParams(lgd_counterparty=0.4, hazard_rate_counterparty=0.01)

        result_high = calculate_all_xva(
            epe, ene, sample_discount_factors, time_grid, params_high
        )
        result_low = calculate_all_xva(
            epe, ene, sample_discount_factors, time_grid, params_low
        )

        assert result_high.cva > result_low.cva
