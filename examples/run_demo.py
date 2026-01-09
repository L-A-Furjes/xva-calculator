#!/usr/bin/env python3
"""
xVA Calculation Engine - Demo Script

This script demonstrates the complete xVA calculation workflow:
1. Define a portfolio of IRS and FX Forward trades
2. Run Monte Carlo simulation
3. Calculate exposure metrics (EPE, ENE)
4. Apply collateral (VM with MPR lag)
5. Calculate all xVA metrics
6. Compute SA-CCR regulatory capital
7. Export results

Usage:
    python examples/run_demo.py
"""

from pathlib import Path

import numpy as np

# Import xva_core components
from xva_core import (
    FXForward,
    IRSwap,
    MonteCarloEngine,
    SACCRCalculator,
)
from xva_core.collateral import InitialMargin, VariationMargin
from xva_core.config.loader import create_default_market_config
from xva_core.exposure.metrics import ExposureMetrics
from xva_core.reporting import create_summary_report
from xva_core.xva.result import XVAParams, calculate_all_xva


def main() -> None:
    """Run the xVA demo."""
    print("=" * 60)
    print("xVA Calculation Engine - Demo")
    print("=" * 60)
    print()

    # =========================================================================
    # 1. Define Portfolio
    # =========================================================================
    print("1. Defining portfolio...")

    instruments = [
        # IRS trades
        IRSwap(
            notional=10_000_000,
            fixed_rate=0.025,
            maturity=5.0,
            pay_fixed=True,
            payment_freq=0.5,
        ),
        IRSwap(
            notional=15_000_000,
            fixed_rate=0.020,
            maturity=3.0,
            pay_fixed=False,
            payment_freq=0.5,
        ),
        IRSwap(
            notional=8_000_000,
            fixed_rate=0.030,
            maturity=7.0,
            pay_fixed=True,
            payment_freq=0.25,
        ),
        # FX Forward trades
        FXForward(
            notional_foreign=5_000_000,
            strike=1.12,
            maturity=1.0,
            buy_foreign=True,
        ),
        FXForward(
            notional_foreign=3_000_000,
            strike=1.08,
            maturity=2.0,
            buy_foreign=False,
        ),
    ]

    total_notional = sum(inst.notional for inst in instruments)
    print(f"   Portfolio: {len(instruments)} trades")
    print(f"   Total notional: ${total_notional:,.0f}")
    print()

    # =========================================================================
    # 2. Run Monte Carlo Simulation
    # =========================================================================
    print("2. Running Monte Carlo simulation...")

    market_config = create_default_market_config()

    engine = MonteCarloEngine(
        n_paths=5000,
        horizon=7.0,  # Max maturity in portfolio
        dt=0.25,  # Quarterly
        seed=42,
    )

    result = engine.simulate(instruments, market_config)

    print(f"   Paths: {result.n_paths}")
    print(f"   Time steps: {result.n_steps}")
    print(f"   Horizon: {result.time_grid[-1]:.1f} years")
    print()

    # =========================================================================
    # 3. Calculate Exposure Metrics
    # =========================================================================
    print("3. Calculating exposure metrics...")

    metrics = ExposureMetrics.from_mtm(result.mtm, result.time_grid)

    print(f"   Peak EPE (uncoll): ${metrics.peak_epe:,.0f}")
    print(f"   Peak ENE (uncoll): ${metrics.peak_ene:,.0f}")
    print(f"   Average EPE: ${metrics.average_epe:,.0f}")
    print()

    # =========================================================================
    # 4. Apply Collateral
    # =========================================================================
    print("4. Applying collateral (VM + IM)...")

    vm = VariationMargin(
        threshold=1_000_000,  # $1M
        mta=100_000,  # $100K
        mpr_days=10,  # 10 business days
        days_per_step=91.25,  # Quarterly = ~91 days
    )

    collateral, coll_exposure = vm.apply(result.mtm, result.time_grid)

    epe_coll = np.maximum(coll_exposure, 0).mean(axis=0)
    ene_coll = np.maximum(-coll_exposure, 0).mean(axis=0)

    im = InitialMargin(multiplier=1.5)
    im_profile = im.calculate(coll_exposure)

    peak_epe_coll = epe_coll.max()
    collateral_benefit = (1 - peak_epe_coll / metrics.peak_epe) * 100

    print(f"   Peak EPE (coll): ${peak_epe_coll:,.0f}")
    print(f"   Collateral benefit: {collateral_benefit:.1f}%")
    print(f"   Average IM: ${im_profile.mean():,.0f}")
    print()

    # =========================================================================
    # 5. Calculate xVA Metrics
    # =========================================================================
    print("5. Calculating xVA metrics...")

    avg_df = result.df_domestic.mean(axis=0)

    xva_params = XVAParams(
        lgd_counterparty=0.60,
        lgd_own=0.60,
        hazard_rate_counterparty=0.012,  # 120 bps
        hazard_rate_own=0.010,  # 100 bps
        funding_spread=0.01,  # 100 bps
        cost_of_capital=0.10,  # 10%
        capital_ratio=0.08,  # 8%
        im_multiplier=1.5,
    )

    xva_result = calculate_all_xva(
        epe=epe_coll,
        ene=ene_coll,
        discount_factors=avg_df,
        time_grid=result.time_grid,
        params=xva_params,
        im_profile=im_profile,
    )

    print()
    print(xva_result.summary(notional=total_notional))
    print()

    # =========================================================================
    # 6. Calculate SA-CCR
    # =========================================================================
    print("6. Calculating SA-CCR regulatory capital...")

    saccr_calc = SACCRCalculator(alpha=1.4)
    current_mtm = result.mtm[:, 0].mean()
    saccr_result = saccr_calc.calculate(instruments, current_mtm=current_mtm)

    print()
    print(saccr_result.summary())
    print()

    # Comparison with internal model
    avg_epe = epe_coll.mean()
    ead_ratio = saccr_result.ead / avg_epe if avg_epe > 0 else 0

    print(f"   Internal Avg EPE: ${avg_epe:,.0f}")
    print(f"   EAD/EPE Ratio: {ead_ratio:.2f}x")
    print()

    # =========================================================================
    # 7. Export Results
    # =========================================================================
    print("7. Exporting results...")

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    files = create_summary_report(
        time_grid=result.time_grid,
        epe=epe_coll,
        ene=ene_coll,
        xva_result=xva_result,
        notional=total_notional,
        output_dir=output_dir,
        prefix="demo",
    )

    for name, path in files.items():
        print(f"   Saved: {path}")

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
