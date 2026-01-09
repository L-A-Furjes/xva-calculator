"""
Tests for instruments module: IRS and FX Forward pricing.
"""

import numpy as np
import pytest

from xva_core.instruments import FXForward, IRSwap


class TestIRSwap:
    """Tests for Interest Rate Swap."""

    def test_swap_creation(self, sample_swap: IRSwap) -> None:
        """Test swap initialization."""
        assert sample_swap.notional == 10_000_000
        assert sample_swap.fixed_rate == 0.02
        assert sample_swap.maturity == 5.0
        assert sample_swap.pay_fixed is True

    def test_swap_cash_flow_dates(self, sample_swap: IRSwap) -> None:
        """Test cash flow date generation."""
        dates = sample_swap.get_cash_flow_dates()
        # Semi-annual payments from 0.5 to 5.0
        expected = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        assert np.allclose(dates, expected)

    def test_expired_swap_returns_zero(self, sample_swap: IRSwap) -> None:
        """Expired swap should have zero MTM."""
        time_grid = np.array([0.0, 5.0, 6.0])
        # Create mock paths data
        paths_data = {
            "df_domestic": np.exp(-0.02 * time_grid).reshape(1, -1).repeat(10, axis=0)
        }

        # At t=6, swap is expired
        mtm = sample_swap.calculate_mtm(2, time_grid, paths_data)
        assert np.all(mtm == 0)

    def test_payer_receiver_opposite_sign(
        self, sample_swap: IRSwap, sample_receiver_swap: IRSwap
    ) -> None:
        """Payer and receiver swaps should have opposite signs at same rate."""
        time_grid = np.linspace(0, 5, 21)
        n_paths = 100
        paths_data = {
            "df_domestic": np.exp(-0.02 * time_grid)
            .reshape(1, -1)
            .repeat(n_paths, axis=0)
        }

        # Create matching swaps
        payer = IRSwap(notional=1e7, fixed_rate=0.02, maturity=5.0, pay_fixed=True)
        receiver = IRSwap(notional=1e7, fixed_rate=0.02, maturity=5.0, pay_fixed=False)

        mtm_payer = payer.calculate_mtm(5, time_grid, paths_data)
        mtm_receiver = receiver.calculate_mtm(5, time_grid, paths_data)

        # Should be opposite signs (approximately equal magnitude)
        assert np.allclose(mtm_payer, -mtm_receiver, rtol=1e-6)

    def test_swap_to_dict(self, sample_swap: IRSwap) -> None:
        """Test serialization to dict."""
        d = sample_swap.to_dict()
        assert d["type"] == "IRS"
        assert d["notional"] == 10_000_000
        assert d["fixed_rate"] == 0.02

    def test_invalid_notional_raises(self) -> None:
        """Negative notional should raise error."""
        with pytest.raises(ValueError):
            IRSwap(notional=-1e7, fixed_rate=0.02, maturity=5.0)

    def test_invalid_maturity_raises(self) -> None:
        """Zero or negative maturity should raise error."""
        with pytest.raises(ValueError):
            IRSwap(notional=1e7, fixed_rate=0.02, maturity=0)


class TestFXForward:
    """Tests for FX Forward."""

    def test_forward_creation(self, sample_fx_forward: FXForward) -> None:
        """Test forward initialization."""
        assert sample_fx_forward.notional_foreign == 5_000_000
        assert sample_fx_forward.strike == 1.10
        assert sample_fx_forward.maturity == 1.0

    def test_notional_domestic(self, sample_fx_forward: FXForward) -> None:
        """Test domestic notional calculation."""
        expected = 5_000_000 * 1.10
        assert sample_fx_forward.notional_domestic == expected

    def test_expired_forward_returns_zero(self, sample_fx_forward: FXForward) -> None:
        """Expired forward should have zero MTM."""
        time_grid = np.array([0.0, 1.0, 2.0])
        n_paths = 10
        paths_data = {
            "fx_spot": np.full((n_paths, 3), 1.10),
            "df_domestic": np.exp(-0.02 * time_grid)
            .reshape(1, -1)
            .repeat(n_paths, axis=0),
            "df_foreign": np.exp(-0.015 * time_grid)
            .reshape(1, -1)
            .repeat(n_paths, axis=0),
        }

        # At t=2, forward is expired
        mtm = sample_fx_forward.calculate_mtm(2, time_grid, paths_data)
        assert np.all(mtm == 0)

    def test_atm_forward_near_zero_at_inception(self) -> None:
        """ATM forward should have near-zero MTM at inception."""
        # Calculate ATM forward rate
        S0, r_d, r_f, T = 1.10, 0.02, 0.015, 1.0
        atm_strike = S0 * np.exp((r_d - r_f) * T)

        fwd = FXForward(
            notional_foreign=1_000_000,
            strike=atm_strike,
            maturity=T,
            buy_foreign=True,
        )

        time_grid = np.array([0.0, 1.0])
        n_paths = 100
        paths_data = {
            "fx_spot": np.full((n_paths, 2), S0),
            "df_domestic": np.exp(-r_d * time_grid)
            .reshape(1, -1)
            .repeat(n_paths, axis=0),
            "df_foreign": np.exp(-r_f * time_grid)
            .reshape(1, -1)
            .repeat(n_paths, axis=0),
        }

        mtm = fwd.calculate_mtm(0, time_grid, paths_data)
        # Should be close to zero for ATM forward
        assert np.abs(mtm.mean()) < 100  # Less than $100 for $1M notional

    def test_buy_sell_opposite_sign(self) -> None:
        """Buy and sell forwards should have opposite MTM."""
        time_grid = np.linspace(0, 1, 5)
        n_paths = 100
        paths_data = {
            "fx_spot": np.full((n_paths, 5), 1.15),  # Spot above strike
            "df_domestic": np.exp(-0.02 * time_grid)
            .reshape(1, -1)
            .repeat(n_paths, axis=0),
            "df_foreign": np.exp(-0.015 * time_grid)
            .reshape(1, -1)
            .repeat(n_paths, axis=0),
        }

        buy_fwd = FXForward(
            notional_foreign=1e6, strike=1.10, maturity=1.0, buy_foreign=True
        )
        sell_fwd = FXForward(
            notional_foreign=1e6, strike=1.10, maturity=1.0, buy_foreign=False
        )

        mtm_buy = buy_fwd.calculate_mtm(2, time_grid, paths_data)
        mtm_sell = sell_fwd.calculate_mtm(2, time_grid, paths_data)

        # If spot > strike, buy is positive, sell is negative
        assert mtm_buy.mean() > 0
        assert mtm_sell.mean() < 0
        assert np.allclose(mtm_buy, -mtm_sell, rtol=1e-6)
