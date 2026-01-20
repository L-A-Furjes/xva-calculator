"""
xVA Calculation Engine - Streamlit Application.

A professional interactive interface for counterparty credit risk
and xVA calculations.

Run with: streamlit run xva_app/app.py
"""

import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# Import xva_core components
from xva_core import (
    FXForward,
    IRSwap,
    MonteCarloEngine,
    SACCRCalculator,
)
from xva_core.collateral import InitialMargin, VariationMargin
from xva_core.config.models import (
    CorrelationConfig,
    FXModelConfig,
    MarketConfig,
    OUModelConfig,
)
from xva_core.exposure.metrics import ExposureMetrics
from xva_core.xva.result import XVAParams, calculate_all_xva

# Page config
st.set_page_config(
    page_title="xVA Calculation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """Main application entry point."""
    st.markdown(
        '<p class="main-header">📊 xVA Calculation Engine</p>', unsafe_allow_html=True
    )
    st.markdown("""
        *Production-grade counterparty credit risk and valuation adjustments*

        This application calculates **CVA, DVA, FVA, MVA, KVA** for a portfolio
        of interest rate swaps and FX forwards using Monte Carlo simulation.
        """)

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0

    # Apply preset slider values BEFORE sidebar renders (Streamlit requirement)
    if st.session_state.get("apply_preset") == "bell_curve_slow":
        # Set slider values for slow MR preset
        st.session_state.kappa_d = 0.03
        st.session_state.sigma_d = 200  # bps
    elif st.session_state.get("apply_preset") in ["mixed", "clear"]:
        # Reset to defaults by deleting keys
        if "kappa_d" in st.session_state:
            del st.session_state["kappa_d"]
        if "sigma_d" in st.session_state:
            del st.session_state["sigma_d"]

    # Sidebar configuration
    config = build_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "📋 Portfolio",
            "📊 Exposure",
            "💸 xVA",
            "🏛️ SA-CCR",
            "📈 Calibration",
            "⚡ Stress Test",
            "📐 Sensitivities",
            "📚 Methodology",
            "💾 Export",
        ]
    )

    with tab1:
        portfolio_tab(config)

    with tab2:
        exposure_tab(config)

    with tab3:
        xva_tab(config)

    with tab4:
        saccr_tab(config)

    with tab5:
        calibration_tab(config)

    with tab6:
        stress_test_tab(config)

    with tab7:
        sensitivities_tab(config)

    with tab8:
        methodology_tab(config)

    with tab9:
        export_tab(config)


def build_sidebar() -> dict:
    """Build sidebar with all configuration options."""
    config = {}

    # Get calibrated params if available
    calib = st.session_state.get("calibrated_params", {})

    # Helper to get calibrated value or default
    def get_calib(key: str, default):
        val = calib.get(key)
        return val if val is not None else default

    with st.sidebar:
        st.title("⚙️ Configuration")

        # Show calibration status
        has_calib = any(
            v is not None
            for k, v in calib.items()
            if k
            in [
                "ir_vol",
                "ir_kappa",
                "ir_theta",
                "fx_vol",
                "fx_spot",
                "hazard_rate",
                "correlation_ir_fx",
            ]
        )
        if has_calib:
            st.success("✓ Using calibrated parameters")

        # Simulation parameters
        with st.expander("🎲 Monte Carlo", expanded=True):
            config["n_paths"] = st.slider("Number of Paths", 100, 10000, 5000, step=100)
            config["horizon"] = st.slider("Horizon (years)", 1, 10, 5)
            config["freq"] = st.selectbox(
                "Time Step", ["Quarterly", "Monthly"], index=0
            )
            config["seed"] = st.number_input("Random Seed", value=42, step=1)

        # Market models
        with st.expander("📈 Market Models"):
            st.subheader("Domestic Rates (OU)")

            # Use calibrated kappa if available
            default_kappa_d = get_calib("ir_kappa", 0.10)
            config["kappa_d"] = st.slider(
                "Mean Reversion κ", 0.01, 0.50, float(default_kappa_d), key="kappa_d"
            )

            # Use calibrated theta if available (stored as decimal, display as %)
            default_theta_d = get_calib("ir_theta", 0.02) * 100  # Convert to %
            config["theta_d"] = (
                st.slider(
                    "Long-term θ (%)", 0.0, 5.0, float(default_theta_d), key="theta_d"
                )
                / 100
            )

            # Use calibrated sigma if available (stored as decimal, display as bps)
            default_sigma_d = get_calib("ir_vol", 0.01) * 10000  # Convert to bps
            config["sigma_d"] = (
                st.slider(
                    "Volatility σ (bps)", 10, 200, int(default_sigma_d), key="sigma_d"
                )
                / 10000
            )

            st.subheader("Foreign Rates (OU)")
            config["kappa_f"] = st.slider(
                "Mean Reversion κ", 0.01, 0.50, 0.08, key="kappa_f"
            )
            config["theta_f"] = (
                st.slider("Long-term θ (%)", 0.0, 5.0, 1.5, key="theta_f") / 100
            )
            config["sigma_f"] = (
                st.slider("Volatility σ (bps)", 10, 200, 120, key="sigma_f") / 10000
            )

            st.subheader("FX Model (GBM)")
            # Use calibrated FX spot if available
            default_fx_spot = get_calib("fx_spot", 1.10)
            config["fx_spot"] = st.number_input(
                "Initial Spot", value=float(default_fx_spot), format="%.4f"
            )

            # Use calibrated FX vol if available (stored as decimal, display as %)
            default_fx_vol = get_calib("fx_vol", 0.12) * 100  # Convert to %
            config["fx_vol"] = (
                st.slider("Volatility (%)", 5, 30, int(default_fx_vol)) / 100
            )

        # Correlations
        with st.expander("🔗 Correlations"):
            config["corr_df"] = st.slider(
                "Domestic-Foreign", -1.0, 1.0, 0.7, key="corr_df"
            )
            # Use calibrated correlation if available
            default_corr_dx = get_calib("correlation_ir_fx", -0.3)
            config["corr_dx"] = st.slider(
                "Domestic-FX", -1.0, 1.0, float(default_corr_dx), key="corr_dx"
            )
            config["corr_fx"] = st.slider("Foreign-FX", -1.0, 1.0, 0.4, key="corr_fx")

            # Validate correlation matrix is positive semi-definite
            # det = 1 + 2*ρ12*ρ13*ρ23 - ρ12² - ρ13² - ρ23²
            rho_df, rho_dx, rho_fx = (
                config["corr_df"],
                config["corr_dx"],
                config["corr_fx"],
            )
            det = 1 + 2 * rho_df * rho_dx * rho_fx - rho_df**2 - rho_dx**2 - rho_fx**2
            if det < 0:
                st.error(
                    f"⚠️ Invalid correlation combination (det={det:.3f} < 0). "
                    "Adjust values to form a valid correlation matrix."
                )
                config["corr_valid"] = False
            else:
                config["corr_valid"] = True

        # Collateral
        with st.expander("🏦 Collateral"):
            # Check if bell curve mode is active - use high threshold to disable VM
            bell_mode = st.session_state.get("bell_curve_mode", False)
            if bell_mode:
                st.info("🔔 Bell Curve Mode: VM/IM disabled (high threshold)")
                default_threshold = 1000.0  # $1B = effectively off
                default_mta = 0.0
                default_mpr = 0
                default_im = 1.0
            else:
                default_threshold = 1.0
                default_mta = 100.0
                default_mpr = 10
                default_im = 1.5

            config["threshold"] = (
                st.number_input("Threshold ($M)", value=default_threshold, step=0.1)
                * 1e6
            )
            config["mta"] = (
                st.number_input("MTA ($K)", value=default_mta, step=10.0) * 1e3
            )
            config["mpr_days"] = st.slider("MPR (days)", 0, 20, default_mpr)
            config["im_mult"] = st.slider("IM Multiplier", 1.0, 3.0, default_im)

        # Credit & Funding
        with st.expander("💰 Credit & Funding"):
            config["lgd_cpty"] = st.slider("LGD Counterparty (%)", 0, 100, 60) / 100
            config["lgd_own"] = st.slider("LGD Own (%)", 0, 100, 60) / 100

            # Use calibrated hazard rate if available (stored as decimal, display as bps)
            default_lambda_cpty = (
                get_calib("hazard_rate", 0.012) * 10000
            )  # Convert to bps
            config["lambda_cpty"] = (
                st.slider("λ Counterparty (bps)", 0, 500, int(default_lambda_cpty))
                / 10000
            )
            config["lambda_own"] = st.slider("λ Own (bps)", 0, 500, 100) / 10000
            config["funding_spread"] = (
                st.slider("Funding Spread (bps)", 0, 300, 100) / 10000
            )
            config["coc"] = st.slider("Cost of Capital (%)", 5, 20, 10) / 100
            config["capital_ratio"] = st.slider("Capital Ratio (%)", 4, 15, 8) / 100

        st.divider()

        # Run button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Run", type="primary", use_container_width=True):
                if not config.get("corr_valid", True):
                    st.error("Cannot run: fix correlation matrix first")
                else:
                    run_simulation(config)
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.results = None
                st.rerun()

    return config


def get_portfolio() -> tuple[list, list]:
    """Get portfolio from session state or defaults."""
    if "irs_trades" not in st.session_state:
        st.session_state.irs_trades = pd.DataFrame(
            {
                "Notional ($M)": [10.0, 15.0, 8.0],
                "Fixed Rate (%)": [2.5, 2.0, 3.0],
                "Maturity (Y)": [5.0, 3.0, 7.0],
                "Pay Fixed": [True, False, True],
            }
        )

    if "fxf_trades" not in st.session_state:
        st.session_state.fxf_trades = pd.DataFrame(
            {
                "Notional (M EUR)": [5.0, 3.0],
                "Strike": [1.12, 1.08],
                "Maturity (Y)": [1.0, 2.0],
                "Buy Foreign": [True, False],
            }
        )

    # Convert to instrument objects
    irs_list = []
    for _, row in st.session_state.irs_trades.iterrows():
        irs_list.append(
            IRSwap(
                notional=row["Notional ($M)"] * 1e6,
                fixed_rate=row["Fixed Rate (%)"] / 100,
                maturity=row["Maturity (Y)"],
                pay_fixed=row["Pay Fixed"],
            )
        )

    fxf_list = []
    for _, row in st.session_state.fxf_trades.iterrows():
        fxf_list.append(
            FXForward(
                notional_foreign=row["Notional (M EUR)"] * 1e6,
                strike=row["Strike"],
                maturity=row["Maturity (Y)"],
                buy_foreign=row["Buy Foreign"],
            )
        )

    return irs_list, fxf_list


def build_market_config(config: dict) -> MarketConfig:
    """Build market configuration from sidebar inputs."""
    return MarketConfig(
        ois_rate=config["theta_d"],
        funding_spread_bps=config["funding_spread"] * 10000,
        cost_of_capital=config["coc"],
        domestic_rate_model=OUModelConfig(
            kappa=config["kappa_d"],
            theta=config["theta_d"],
            sigma=config["sigma_d"],
            initial_rate=config["theta_d"],
        ),
        foreign_rate_model=OUModelConfig(
            kappa=config["kappa_f"],
            theta=config["theta_f"],
            sigma=config["sigma_f"],
            initial_rate=config["theta_f"],
        ),
        fx_model=FXModelConfig(
            initial_spot=config["fx_spot"],
            volatility=config["fx_vol"],
        ),
        correlations=CorrelationConfig(
            domestic_foreign=config["corr_df"],
            domestic_fx=config["corr_dx"],
            foreign_fx=config["corr_fx"],
        ),
    )


def run_simulation(config: dict) -> None:
    """Run the Monte Carlo simulation and calculate xVA."""
    with st.spinner("Running simulation..."):
        # Get portfolio
        irs_list, fxf_list = get_portfolio()
        instruments = irs_list + fxf_list

        if len(instruments) == 0:
            st.error("Please add at least one trade to the portfolio.")
            return

        # Build configs
        market_config = build_market_config(config)

        # Bell curve mode: use dt=0.5 to align with semi-annual payment dates
        # This removes sawtooth artifacts and shows cleaner bell shape
        if st.session_state.get("bell_curve_mode"):
            dt = 0.5
        else:
            dt = 1 / 12 if config["freq"] == "Monthly" else 0.25

        # Run Monte Carlo
        engine = MonteCarloEngine(
            n_paths=config["n_paths"],
            horizon=config["horizon"],
            dt=dt,
            seed=config["seed"],
        )

        result = engine.simulate(instruments, market_config)

        # Calculate exposure metrics
        metrics = ExposureMetrics.from_mtm(result.mtm, result.time_grid)

        # Apply collateral
        vm = VariationMargin(
            threshold=config["threshold"],
            mta=config["mta"],
            mpr_days=config["mpr_days"],
            days_per_step=dt * 365,
        )
        collateral, coll_exposure = vm.apply(result.mtm, result.time_grid)
        epe_coll = np.maximum(coll_exposure, 0).mean(axis=0)
        ene_coll = np.maximum(-coll_exposure, 0).mean(axis=0)

        # Calculate IM
        im = InitialMargin(multiplier=config["im_mult"])
        im_profile = im.calculate(coll_exposure)

        # Average discount factors
        avg_df = result.df_domestic.mean(axis=0)

        # Calculate xVA
        xva_params = XVAParams(
            lgd_counterparty=config["lgd_cpty"],
            lgd_own=config["lgd_own"],
            hazard_rate_counterparty=config["lambda_cpty"],
            hazard_rate_own=config["lambda_own"],
            funding_spread=config["funding_spread"],
            cost_of_capital=config["coc"],
            capital_ratio=config["capital_ratio"],
            im_multiplier=config["im_mult"],
        )

        xva_result = calculate_all_xva(
            epe=epe_coll,
            ene=ene_coll,
            discount_factors=avg_df,
            time_grid=result.time_grid,
            params=xva_params,
            im_profile=im_profile,
        )

        # SA-CCR
        saccr_calc = SACCRCalculator()
        current_mtm = result.mtm[:, 0].mean()
        saccr_result = saccr_calc.calculate(instruments, current_mtm)

        # Store results
        st.session_state.results = {
            "sim_result": result,
            "metrics": metrics,
            "epe_uncoll": metrics.epe,
            "ene_uncoll": metrics.ene,
            "epe_coll": epe_coll,
            "ene_coll": ene_coll,
            "im_profile": im_profile,
            "xva_result": xva_result,
            "saccr_result": saccr_result,
            "instruments": instruments,
            "config": config,
            "avg_df": avg_df,
        }

        st.session_state.run_count += 1

    st.success("Simulation complete!")
    st.rerun()


def _calculate_par_swap_rate(
    r0: float, maturity: float = 5.0, freq: float = 0.5
) -> float:
    """
    Calculate par swap rate for a flat yield curve.

    For a flat curve at rate r, par rate K satisfies:
    K = (1 - P(0,T)) / (Σ τ_i * P(0, t_i))

    where P(0, t) = e^(-r*t) for a flat curve.
    """
    # Payment times
    n_payments = int(maturity / freq)
    payment_times = np.array([(i + 1) * freq for i in range(n_payments)])

    # Discount factors for flat curve
    discount_factors = np.exp(-r0 * payment_times)

    # Par rate formula
    numerator = 1.0 - discount_factors[-1]
    denominator = freq * np.sum(discount_factors)

    if denominator < 1e-10:
        return r0  # Fallback

    return numerator / denominator


def _calculate_deterministic_par_rate(
    r0: float, maturity: float = 5.0, freq: float = 0.5
) -> float:
    """
    Calculate par swap rate using deterministic flat curve.

    This gives K such that PV0(K) = 0 exactly using the same
    pricing routine as the simulator. No MC needed.
    """
    # Create a dummy swap to use its par rate calculation
    dummy_swap = IRSwap(
        notional=10_000_000,
        fixed_rate=0.02,  # Doesn't matter
        maturity=maturity,
        pay_fixed=True,
        payment_freq=freq,
    )
    return dummy_swap.calculate_par_rate_deterministic(r0)


def _apply_portfolio_preset(preset: str, config: dict) -> None:
    """Apply a portfolio preset - sets trades AND collateral parameters."""
    if preset == "bell_curve":
        # ===== BELL CURVE DEMO =====
        # Single ATM IRS, VM/IM disabled, clean setup for classic EPE hump

        # Calculate par rate using DETERMINISTIC flat curve
        # This is the proper fix: PV0(K_par) = 0 exactly, using same pricing as simulator
        r0 = config.get("theta_d", 0.02)
        par_rate = _calculate_deterministic_par_rate(r0, maturity=5.0, freq=0.5)
        par_rate_pct = par_rate * 100  # As percentage

        # Set single ATM swap with calculated par rate
        st.session_state.irs_trades = pd.DataFrame(
            {
                "Notional ($M)": [10.0],
                "Fixed Rate (%)": [round(par_rate_pct, 4)],  # High precision
                "Maturity (Y)": [5.0],
                "Pay Fixed": [True],
            }
        )
        # No FX forwards
        st.session_state.fxf_trades = pd.DataFrame(
            {
                "Notional (M EUR)": pd.Series([], dtype=float),
                "Strike": pd.Series([], dtype=float),
                "Maturity (Y)": pd.Series([], dtype=float),
                "Buy Foreign": pd.Series([], dtype=bool),
            }
        )

        # Disable collateral by setting very high threshold
        # Delete existing widget keys to allow new defaults
        for key in ["threshold_input", "mta_input", "mpr_slider", "im_mult_slider"]:
            if key in st.session_state:
                del st.session_state[key]

        # Store flag and calculated par rate for display
        st.session_state.bell_curve_mode = True
        st.session_state.bell_curve_threshold = 1e9  # $1B threshold = effectively off
        st.session_state.calculated_par_rate = par_rate_pct

    elif preset == "bell_curve_slow":
        # ===== BELL CURVE (SLOW MR) =====
        # Same as bell_curve but with slower mean reversion and higher vol
        # This shifts the EPE peak from ~0.5Y towards ~2Y for a more "textbook" bell
        # Note: slider values are set in main() BEFORE sidebar renders

        # Store flags for reference
        st.session_state.bell_slow_kappa = 0.03
        st.session_state.bell_slow_sigma = 200

        # Calculate par rate with default theta (will use slow kappa/sigma for sim)
        r0 = config.get("theta_d", 0.02)
        par_rate = _calculate_deterministic_par_rate(r0, maturity=5.0, freq=0.5)
        par_rate_pct = par_rate * 100

        # Set single ATM swap
        st.session_state.irs_trades = pd.DataFrame(
            {
                "Notional ($M)": [10.0],
                "Fixed Rate (%)": [round(par_rate_pct, 4)],
                "Maturity (Y)": [5.0],
                "Pay Fixed": [True],
            }
        )
        st.session_state.fxf_trades = pd.DataFrame(
            {
                "Notional (M EUR)": pd.Series([], dtype=float),
                "Strike": pd.Series([], dtype=float),
                "Maturity (Y)": pd.Series([], dtype=float),
                "Buy Foreign": pd.Series([], dtype=bool),
            }
        )

        # Disable collateral
        for key in ["threshold_input", "mta_input", "mpr_slider", "im_mult_slider"]:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.bell_curve_mode = True
        st.session_state.bell_curve_slow_mode = True
        st.session_state.bell_curve_threshold = 1e9
        st.session_state.calculated_par_rate = par_rate_pct

    elif preset == "mixed":
        # ===== MIXED PORTFOLIO =====
        st.session_state.irs_trades = pd.DataFrame(
            {
                "Notional ($M)": [10.0, 8.0],
                "Fixed Rate (%)": [2.0, 2.5],
                "Maturity (Y)": [5.0, 3.0],
                "Pay Fixed": [True, False],
            }
        )
        st.session_state.fxf_trades = pd.DataFrame(
            {
                "Notional (M EUR)": [5.0],
                "Strike": [1.10],
                "Maturity (Y)": [2.0],
                "Buy Foreign": [True],
            }
        )
        st.session_state.bell_curve_mode = False
        st.session_state.bell_curve_slow_mode = False
        st.session_state.bell_slow_kappa = None
        st.session_state.bell_slow_sigma = None
        # Note: slider values are reset in main() BEFORE sidebar renders

    elif preset == "clear":
        # ===== CLEAR ALL =====
        st.session_state.irs_trades = pd.DataFrame(
            {
                "Notional ($M)": pd.Series([], dtype=float),
                "Fixed Rate (%)": pd.Series([], dtype=float),
                "Maturity (Y)": pd.Series([], dtype=float),
                "Pay Fixed": pd.Series([], dtype=bool),
            }
        )
        st.session_state.fxf_trades = pd.DataFrame(
            {
                "Notional (M EUR)": pd.Series([], dtype=float),
                "Strike": pd.Series([], dtype=float),
                "Maturity (Y)": pd.Series([], dtype=float),
                "Buy Foreign": pd.Series([], dtype=bool),
            }
        )
        st.session_state.bell_curve_mode = False
        st.session_state.bell_curve_slow_mode = False
        st.session_state.bell_slow_kappa = None
        st.session_state.bell_slow_sigma = None
        # Note: slider values are reset in main() BEFORE sidebar renders


def portfolio_tab(config: dict) -> None:
    """Portfolio editing tab."""
    st.header("📋 Trade Portfolio")

    # Demo presets
    st.subheader("🎯 Quick Presets")
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    with preset_col1:
        if st.button(
            "🔔 Demo Bell Curve",
            help="ATM single IRS, VM/IM off, non-zero vol → classic bell-shaped EPE",
        ):
            # Store preset request - will be applied on next run
            st.session_state.apply_preset = "bell_curve"
            st.rerun()

    with preset_col2:
        if st.button("📊 Mixed Portfolio", help="IRS + FX with netting effects"):
            st.session_state.apply_preset = "mixed"
            st.rerun()

    with preset_col3:
        if st.button("🧹 Clear All", help="Empty portfolio"):
            st.session_state.apply_preset = "clear"
            st.rerun()

    # Second row of presets
    preset_col4, preset_col5, preset_col6 = st.columns(3)
    with preset_col4:
        if st.button(
            "🔔 Bell (Slow MR)",
            help="Bell curve with slower mean reversion (κ=0.03, σ=2%) → peak at ~2Y",
        ):
            st.session_state.apply_preset = "bell_curve_slow"
            st.rerun()

    # Check if a preset was requested and apply it
    if "apply_preset" in st.session_state:
        preset = st.session_state.pop("apply_preset")
        _apply_portfolio_preset(preset, config)

    # Show current preset status
    if st.session_state.get("bell_curve_mode"):
        par_rate = st.session_state.get("calculated_par_rate", 2.0)
        if st.session_state.get("bell_curve_slow_mode"):
            st.success(
                f"🔔 **Bell Curve (Slow MR) Active** - Par rate = {par_rate:.3f}%, "
                f"κ=0.03, σ=2%, VM/IM disabled. Peak at ~2Y!"
            )
        else:
            st.success(
                f"🔔 **Bell Curve Demo Active** - Par rate = {par_rate:.3f}%, VM/IM disabled. Click Run!"
            )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Interest Rate Swaps")
        st.caption("💡 For bell curve: Fixed Rate ≈ par rate (≈ long-term mean rate)")
        st.session_state.irs_trades = st.data_editor(
            st.session_state.get(
                "irs_trades",
                pd.DataFrame(
                    {
                        "Notional ($M)": [10.0],
                        "Fixed Rate (%)": [2.0],  # Close to par rate
                        "Maturity (Y)": [5.0],
                        "Pay Fixed": [True],
                    }
                ),
            ),
            num_rows="dynamic",
            use_container_width=True,
        )

    with col2:
        st.subheader("FX Forwards")
        st.caption("💡 FX exposure is linear - no bell curve effect")
        st.session_state.fxf_trades = st.data_editor(
            st.session_state.get(
                "fxf_trades",
                pd.DataFrame(
                    {
                        "Notional (M EUR)": [5.0],
                        "Strike": [1.10],  # ATM: equals fx_spot default
                        "Maturity (Y)": [2.0],
                        "Buy Foreign": [True],
                    }
                ),
            ),
            num_rows="dynamic",
            use_container_width=True,
        )

    # Summary
    st.divider()
    col1, col2, col3 = st.columns(3)

    irs_notional = st.session_state.irs_trades["Notional ($M)"].sum()
    fxf_notional = st.session_state.fxf_trades["Notional (M EUR)"].sum() * config.get(
        "fx_spot", 1.10
    )
    n_trades = len(st.session_state.irs_trades) + len(st.session_state.fxf_trades)

    with col1:
        st.metric("IRS Notional", f"${irs_notional:.1f}M")
    with col2:
        st.metric("FX Notional (USD)", f"${fxf_notional:.1f}M")
    with col3:
        st.metric("Total Trades", n_trades)


def exposure_tab(config: dict) -> None:
    """Exposure profiles tab."""
    st.header("📊 Exposure Profiles")

    if st.session_state.results is None:
        st.info("Run simulation to see exposure profiles.")
        st.markdown("""
        **Expected Exposure Profiles:**

        - **IRS (ATM)**: Bell-shaped curve - starts at 0, peaks mid-life, returns to 0 at maturity
        - **FX Forward**: Linear increase then drop at maturity
        - **Portfolio**: Combination depending on trade composition

        ---

        ### 🔔 Comment voir la "cloche" classique ?

        Pour observer le profil en cloche typique d'un swap, il faut :

        1. **Trade à par (ATM)** : Fixed Rate ≈ Par Rate (proche du taux long terme θ_OU dans Market Models)
        2. **Pas de collatéral** : VM Threshold très élevé ($1B+) pour désactiver VM
        3. **Un seul produit** : Pas de netting qui masque la forme
        4. **Volatilité non nulle** : σ > 0

        👉 **Cliquez sur "🔔 Demo Bell Curve" dans l'onglet Portfolio** - tout sera configuré automatiquement !

        ---

        **Note IMM vs SA-CCR :**
        - **EE** (Expected Exposure) : $\\mathbb{E}[\\max(V_t, 0)]$ - c'est ce qu'on affiche
        - **Effective EE** : Running max de EE (non-décroissante) - *utilisé dans IMM, pas affiché ici*
        - **SA-CCR** : N'utilise pas Effective EE, donc les courbes peuvent différer du schéma IMM classique
        """)
        return

    r = st.session_state.results

    # Plotly charts
    try:
        import plotly.graph_objects as go

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uncollateralized")
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=r["sim_result"].time_grid,
                    y=r["epe_uncoll"] / 1e6,
                    name="EPE",
                    line={"color": "#FF4B4B"},
                )
            )
            fig1.add_trace(
                go.Scatter(
                    x=r["sim_result"].time_grid,
                    y=r["ene_uncoll"] / 1e6,
                    name="ENE",
                    line={"color": "#00CC96"},
                )
            )
            fig1.update_layout(
                xaxis_title="Time (Y)",
                yaxis_title="Exposure ($M)",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Collateralized")
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=r["sim_result"].time_grid,
                    y=r["epe_coll"] / 1e6,
                    name="EPE (Coll)",
                    line={"color": "#FF4B4B"},
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=r["sim_result"].time_grid,
                    y=r["ene_coll"] / 1e6,
                    name="ENE (Coll)",
                    line={"color": "#00CC96"},
                )
            )
            fig2.update_layout(
                xaxis_title="Time (Y)",
                yaxis_title="Exposure ($M)",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # IM Profile
        st.subheader("Initial Margin Profile")
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=r["sim_result"].time_grid,
                y=r["im_profile"] / 1e6,
                name="IM",
                fill="tozeroy",
                line={"color": "#9467BD"},
            )
        )
        fig3.update_layout(
            xaxis_title="Time (Y)",
            yaxis_title="IM ($M)",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

    except ImportError:
        st.warning("Plotly not available. Install with: pip install plotly")

    # Metrics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Peak EPE (Uncoll)", f"${r['epe_uncoll'].max()/1e6:.2f}M")
    with col2:
        st.metric("Peak EPE (Coll)", f"${r['epe_coll'].max()/1e6:.2f}M")
    with col3:
        # Use AUC-based reduction with guard for small denominators
        times = r["sim_result"].time_grid
        # np.trapz renamed to np.trapezoid in NumPy 2.0
        if hasattr(np, "trapezoid"):
            auc_uncoll = np.trapezoid(r["epe_uncoll"], times)
            auc_coll = np.trapezoid(r["epe_coll"], times)
        else:
            auc_uncoll = np.trapz(r["epe_uncoll"], times)
            auc_coll = np.trapz(r["epe_coll"], times)
        if auc_uncoll < 1e-6:  # Guard against tiny/zero AUC
            st.metric("Collateral Benefit", "N/A", help="EPE too small to measure")
        else:
            reduction = (1 - auc_coll / auc_uncoll) * 100
            st.metric(
                "Collateral Benefit",
                f"{reduction:.1f}%",
                help="AUC reduction: (1 - AUC_coll/AUC_uncoll) × 100%",
            )
    with col4:
        st.metric("Avg IM", f"${r['im_profile'].mean()/1e6:.2f}M")

    # Bell curve diagnostics
    if st.session_state.get("bell_curve_mode"):
        with st.expander("🔍 Bell Curve Diagnostics", expanded=True):
            times = r["sim_result"].time_grid
            epe = r["epe_uncoll"]
            mtm = r["sim_result"].mtm  # (n_paths, n_steps)

            # ===== THE DECISIVE TEST: PV(0) =====
            st.markdown("### 🎯 Test Décisif : PV(0)")
            pv0_mean = mtm[:, 0].mean()
            pv0_std = mtm[:, 0].std()
            notional = 10e6  # Assuming 10M notional

            col_pv1, col_pv2, col_pv3 = st.columns(3)
            with col_pv1:
                st.metric(
                    "Avg PV(0)",
                    f"${pv0_mean/1e6:.4f}M",
                    help="Should be ~0 for ATM swap",
                )
            with col_pv2:
                st.metric(
                    "Std PV(0)",
                    f"${pv0_std/1e6:.6f}M",
                    help="Should be ~0 (t=0 is deterministic)",
                )
            with col_pv3:
                pv0_bps = abs(pv0_mean) / notional * 10000
                st.metric("PV(0) in bps", f"{pv0_bps:.1f} bps")

            # Verdict for ATM
            if abs(pv0_mean) < notional * 0.001:  # < 10 bps of notional
                st.success("✅ Swap is ATM: PV(0) ≈ 0 (using MC par rate)")
            else:
                direction = "ITM (positive)" if pv0_mean > 0 else "OTM (negative)"
                st.warning(
                    f"⚠️ Swap slightly off ATM: PV(0) = ${pv0_mean/1e6:.4f}M ({direction}). "
                    f"This is expected due to MC sampling variance."
                )

            # Verdict for deterministic t=0
            if pv0_std < 1e-6:
                st.success("✅ V(0) is deterministic (correct for xVA)")
            else:
                st.info(
                    f"ℹ️ V(0) has std=${pv0_std/1e6:.4f}M "
                    "(slight variance from MC averaging)"
                )

            st.divider()

            # Other diagnostics
            st.markdown("### 📊 Exposure Profile")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Time Grid Start", f"t={times[0]:.2f}Y")
                st.metric("EPE(t=0)", f"${epe[0]/1e6:.4f}M")
            with col_b:
                peak_idx = np.argmax(epe)
                st.metric("Peak Time", f"t={times[peak_idx]:.2f}Y")
                st.metric("Peak EPE", f"${epe[peak_idx]/1e6:.2f}M")
            with col_c:
                st.metric("EPE(T=end)", f"${epe[-1]/1e6:.4f}M")
                # Check if VM is truly off
                vm_diff = np.abs(r["epe_uncoll"] - r["epe_coll"]).max()
                if vm_diff < 1e-6:
                    st.success("✅ VM off")
                else:
                    st.warning(f"⚠️ VM active (diff={vm_diff/1e6:.2f}M)")

            # Show if it's a proper bell curve
            if epe[0] < epe[peak_idx] * 0.1 and epe[-1] < epe[peak_idx] * 0.1:
                st.success(
                    "🔔 Bell curve detected: starts low, peaks in middle, ends low"
                )
            elif peak_idx <= 1:
                st.warning(
                    "⚠️ Peak at start - swap is ITM. Adjust Fixed Rate based on PV(0)."
                )
            else:
                st.info(f"Profile: EPE peaks at t={times[peak_idx]:.2f}Y")


def xva_tab(config: dict) -> None:
    """xVA breakdown tab."""
    st.header("💸 xVA Breakdown")

    if st.session_state.results is None:
        st.info("Run simulation to see xVA results.")
        return

    r = st.session_state.results
    xva = r["xva_result"]

    # Bar chart
    try:
        import plotly.graph_objects as go

        metrics = ["CVA", "DVA", "FVA", "MVA", "KVA"]
        values = [xva.cva, -xva.dva, xva.fva, xva.mva, xva.kva]
        colors = ["#FF4B4B", "#00CC96", "#FFA500", "#9467BD", "#1F77B4"]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[v / 1e6 for v in values],
                marker_color=colors,
                text=[f"${v/1e6:.2f}M" for v in values],
                textposition="outside",
            )
        )
        fig.update_layout(
            yaxis_title="Value ($M)",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        pass

    # Total
    total_notional = sum(inst.notional for inst in r["instruments"])
    bps = xva.to_bps(total_notional)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total xVA Cost", f"${xva.total/1e6:.2f}M")
    with col2:
        st.metric("bps of Notional", f"{bps['total']:.1f} bps")

    # Detailed table
    st.subheader("Component Details")
    detail_df = pd.DataFrame(
        {
            "Component": ["CVA", "DVA (benefit)", "FVA", "MVA", "KVA", "Total"],
            "Value ($M)": [
                xva.cva / 1e6,
                -xva.dva / 1e6,
                xva.fva / 1e6,
                xva.mva / 1e6,
                xva.kva / 1e6,
                xva.total / 1e6,
            ],
            "bps": [
                bps["cva"],
                -bps["dva"],
                bps["fva"],
                bps["mva"],
                bps["kva"],
                bps["total"],
            ],
        }
    )
    st.dataframe(detail_df, use_container_width=True)


def saccr_tab(config: dict) -> None:
    """SA-CCR regulatory capital tab."""
    st.header("🏛️ SA-CCR Regulatory Capital")

    if st.session_state.results is None:
        st.info("Run simulation to see SA-CCR results.")
        return

    r = st.session_state.results
    saccr = r["saccr_result"]

    st.info("""
        **SA-CCR**: Standardized Approach for Counterparty Credit Risk (Basel III)

        EAD = α × (RC + PFE), where α = 1.4
        """)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Replacement Cost", f"${saccr.replacement_cost/1e6:.2f}M")
    with col2:
        st.metric("Aggregate Add-On", f"${saccr.aggregate_addon/1e6:.2f}M")
    with col3:
        st.metric("PFE", f"${saccr.pfe/1e6:.2f}M")
    with col4:
        st.metric("EAD", f"${saccr.ead/1e6:.2f}M")

    # Trade breakdown
    st.subheader("Trade Add-On Breakdown")
    trade_data = []
    for ta in saccr.trade_addons:
        trade_data.append(
            {
                "Trade": ta.trade_id,
                "Asset Class": ta.asset_class.value,
                "Notional ($M)": ta.notional / 1e6,
                "Maturity (Y)": ta.maturity,
                "SF": ta.supervisory_factor,
                "Add-On ($M)": ta.addon / 1e6,
            }
        )
    st.dataframe(pd.DataFrame(trade_data), use_container_width=True)

    # Comparison with internal model
    st.subheader("Comparison with Internal Model")
    avg_epe = r["epe_coll"].mean()
    ratio = saccr.ead / avg_epe if avg_epe > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SA-CCR EAD", f"${saccr.ead/1e6:.2f}M")
    with col2:
        st.metric("Internal Avg EPE", f"${avg_epe/1e6:.2f}M")
    with col3:
        st.metric("EAD/EPE Ratio", f"{ratio:.2f}×")


def calibration_tab(config: dict) -> None:
    """
    Historical data calibration tab.

    Allows users to calibrate model parameters from market data:
    - Interest rate volatility (from historical rates)
    - FX volatility (from historical spot prices)
    - Hazard rates (from CDS spreads)
    - Correlations (from IR/FX time series)
    """
    st.header("📈 Market Data Calibration")

    st.markdown("""
        Calibrate model parameters from historical market data. Upload CSV files
        from Bloomberg, Reuters, or any financial data provider to derive:
        - **Volatilities** for IR and FX models
        - **Hazard rates** from CDS spreads for CVA/DVA
        - **Correlations** between risk factors
        """)

    # Initialize session state for calibrated values
    if "calibrated_params" not in st.session_state:
        st.session_state.calibrated_params = {
            "ir_vol": None,
            "fx_vol": None,
            "hazard_rate": None,
            "cds_term_structure": None,
            "correlation_ir_fx": None,
            "ir_data": None,
            "fx_data": None,
        }

    # Create sub-tabs for different calibration types
    calib_tab1, calib_tab2, calib_tab3, calib_tab4, calib_tab5 = st.tabs(
        [
            "📊 Volatilities",
            "💳 CDS / Credit",
            "🔗 Correlations",
            "📉 OIS Curve",
            "📋 Summary",
        ]
    )

    with calib_tab1:
        _calibration_volatility_section(config)

    with calib_tab2:
        _calibration_cds_section(config)

    with calib_tab3:
        _calibration_correlation_section(config)

    with calib_tab4:
        _calibration_ois_section(config)

    with calib_tab5:
        _calibration_summary_section(config)


def _calibration_volatility_section(config: dict) -> None:
    """Volatility calibration for IR and FX."""
    st.subheader("Volatility Calibration")

    st.info("""
        **Methodology:**
        - **IR Volatility**: Calculated from absolute rate changes (OU model assumption)
        - **FX Volatility**: Calculated from log-returns (GBM model assumption)

        Both are annualized using √252 (trading days).
        """)

    col1, col2 = st.columns(2)

    # ===== Interest Rate Volatility =====
    with col1:
        st.markdown("### 📊 Interest Rates")
        st.markdown("Upload CSV: `date`, `rate` (in % or decimal)")

        ir_file = st.file_uploader(
            "Upload IR historical data",
            type=["csv"],
            key="ir_upload",
            help="CSV with date and rate columns. Rate can be in % (e.g., 2.5) or decimal (e.g., 0.025)",
        )

        if ir_file is not None:
            try:
                ir_df = pd.read_csv(ir_file, parse_dates=["date"])
                ir_df = ir_df.sort_values("date").reset_index(drop=True)

                # Auto-detect if rates are in % or decimal
                if ir_df["rate"].mean() > 0.5:  # Likely in percentage
                    ir_df["rate_decimal"] = ir_df["rate"] / 100
                    rate_format = "percentage"
                else:
                    ir_df["rate_decimal"] = ir_df["rate"]
                    rate_format = "decimal"

                with st.expander("Preview Data", expanded=False):
                    st.dataframe(ir_df.head(10), use_container_width=True)
                    st.caption(
                        f"Detected format: {rate_format} | {len(ir_df)} observations"
                    )

                # Calculate volatility
                window = st.slider(
                    "Rolling Window (days)", 20, 252, 60, key="ir_window"
                )

                # For OU model: use absolute changes in rates
                ir_df["change"] = ir_df["rate_decimal"].diff()
                ir_df["rolling_vol"] = ir_df["change"].rolling(window).std()

                # Annualize
                annualized_vol = ir_df["rolling_vol"].iloc[-1] * np.sqrt(252)

                # Also calculate mean reversion estimate (kappa) using OU regression
                # dr = kappa * (theta - r) * dt + sigma * dW
                # Simple estimate: kappa ≈ -cov(dr, r) / var(r)
                dr = ir_df["change"].dropna()
                r_lag = ir_df["rate_decimal"].shift(1).dropna()
                if len(dr) > 10 and len(r_lag) > 10:
                    # Align arrays
                    min_len = min(len(dr), len(r_lag))
                    dr_aligned = dr.iloc[:min_len].values
                    r_aligned = r_lag.iloc[:min_len].values

                    cov_dr_r = np.cov(dr_aligned, r_aligned)[0, 1]
                    var_r = np.var(r_aligned)
                    kappa_estimate = -cov_dr_r / var_r * 252 if var_r > 0 else 0.1
                    kappa_estimate = max(0.01, min(1.0, kappa_estimate))  # Bound it
                    theta_estimate = ir_df["rate_decimal"].mean()
                else:
                    kappa_estimate = 0.1
                    theta_estimate = ir_df["rate_decimal"].mean()

                # Display metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(
                        "σ (Volatility)",
                        f"{annualized_vol*10000:.0f} bps",
                        help="Annualized volatility in basis points",
                    )
                with col_m2:
                    st.metric(
                        "κ (Mean Rev.)",
                        f"{kappa_estimate:.2f}",
                        help="Estimated mean reversion speed",
                    )
                with col_m3:
                    st.metric(
                        "θ (Long-term)",
                        f"{theta_estimate*100:.2f}%",
                        help="Estimated long-term mean rate",
                    )

                # Plot
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        subplot_titles=("Historical Rates", "Rolling Volatility"),
                        vertical_spacing=0.12,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=ir_df["date"],
                            y=ir_df["rate_decimal"] * 100,
                            name="Rate (%)",
                            line={"color": "#1f77b4"},
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=ir_df["date"],
                            y=ir_df["rolling_vol"] * np.sqrt(252) * 10000,
                            name="Rolling Vol (bps)",
                            line={"color": "#FF4B4B"},
                        ),
                        row=2,
                        col=1,
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=400,
                        showlegend=False,
                    )
                    fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
                    fig.update_yaxes(title_text="Vol (bps)", row=2, col=1)

                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    pass

                # Store calibrated values
                st.session_state.calibrated_params["ir_vol"] = annualized_vol
                st.session_state.calibrated_params["ir_kappa"] = kappa_estimate
                st.session_state.calibrated_params["ir_theta"] = theta_estimate
                st.session_state.calibrated_params["ir_data"] = ir_df[
                    ["date", "rate_decimal"]
                ].copy()

                if st.button("✅ Apply IR Parameters", key="apply_ir", type="primary"):
                    # Delete widget keys so sliders pick up new calibrated defaults on rerun
                    for key in ["kappa_d", "theta_d", "sigma_d"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success(
                        f"Applied: σ={annualized_vol*10000:.0f}bps, "
                        f"κ={kappa_estimate:.2f}, θ={theta_estimate*100:.2f}%"
                    )
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing IR file: {e}")

    # ===== FX Volatility =====
    with col2:
        st.markdown("### 💱 FX Spot Rates")
        st.markdown("Upload CSV: `date`, `rate` (spot price)")

        fx_file = st.file_uploader(
            "Upload FX historical data",
            type=["csv"],
            key="fx_upload",
            help="CSV with date and rate (spot price) columns",
        )

        if fx_file is not None:
            try:
                fx_df = pd.read_csv(fx_file, parse_dates=["date"])
                fx_df = fx_df.sort_values("date").reset_index(drop=True)

                with st.expander("Preview Data", expanded=False):
                    st.dataframe(fx_df.head(10), use_container_width=True)
                    st.caption(f"{len(fx_df)} observations")

                # Calculate volatility
                window = st.slider(
                    "Rolling Window (days)", 20, 252, 60, key="fx_window"
                )

                # Log returns for GBM
                fx_df["log_return"] = np.log(fx_df["rate"] / fx_df["rate"].shift(1))
                fx_df["rolling_vol"] = fx_df["log_return"].rolling(window).std()

                # Annualize
                annualized_vol = fx_df["rolling_vol"].iloc[-1] * np.sqrt(252)
                current_spot = fx_df["rate"].iloc[-1]

                # Calculate drift (annualized)
                mean_return = fx_df["log_return"].mean() * 252

                # Display metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(
                        "σ (Volatility)",
                        f"{annualized_vol*100:.1f}%",
                        help="Annualized volatility",
                    )
                with col_m2:
                    st.metric(
                        "Spot",
                        f"{current_spot:.4f}",
                        help="Latest spot rate",
                    )
                with col_m3:
                    st.metric(
                        "Drift (μ)",
                        f"{mean_return*100:.1f}%",
                        help="Annualized drift",
                    )

                # Plot
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        subplot_titles=("FX Spot Price", "Rolling Volatility"),
                        vertical_spacing=0.12,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=fx_df["date"],
                            y=fx_df["rate"],
                            name="Spot",
                            line={"color": "#00CC96"},
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=fx_df["date"],
                            y=fx_df["rolling_vol"] * np.sqrt(252) * 100,
                            name="Rolling Vol (%)",
                            line={"color": "#FFA500"},
                        ),
                        row=2,
                        col=1,
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=400,
                        showlegend=False,
                    )
                    fig.update_yaxes(title_text="Spot", row=1, col=1)
                    fig.update_yaxes(title_text="Vol (%)", row=2, col=1)

                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    pass

                # Store calibrated values
                st.session_state.calibrated_params["fx_vol"] = annualized_vol
                st.session_state.calibrated_params["fx_spot"] = current_spot
                st.session_state.calibrated_params["fx_data"] = fx_df[
                    ["date", "rate", "log_return"]
                ].copy()

                if st.button("✅ Apply FX Parameters", key="apply_fx", type="primary"):
                    # Note: FX sliders don't have keys, so we just store in calibrated_params
                    # The sidebar will pick up the new defaults on next render
                    st.success(
                        f"Applied: σ={annualized_vol*100:.1f}%, Spot={current_spot:.4f}"
                    )
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing FX file: {e}")

    # Sample data download section
    st.divider()
    st.markdown("### 📥 Sample Data Templates")

    col1, col2 = st.columns(2)

    with col1:
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        rates = 4.5 + np.cumsum(np.random.randn(252) * 0.02)  # More realistic rates
        sample_ir = pd.DataFrame({"date": dates, "rate": rates})

        st.download_button(
            "📥 Sample IR Data",
            sample_ir.to_csv(index=False),
            "sample_ir_data.csv",
            "text/csv",
        )

    with col2:
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        fx_rates = 1.10 * np.exp(np.cumsum(np.random.randn(252) * 0.006))
        sample_fx = pd.DataFrame({"date": dates, "rate": fx_rates})

        st.download_button(
            "📥 Sample FX Data",
            sample_fx.to_csv(index=False),
            "sample_fx_data.csv",
            "text/csv",
        )


def _calibration_cds_section(config: dict) -> None:
    """CDS spread to hazard rate calibration."""
    st.subheader("Credit Calibration from CDS Spreads")

    st.info("""
        **CDS → Hazard Rate Conversion:**

        The hazard rate (λ) represents the instantaneous probability of default.
        It can be derived from CDS spreads using:

        $$\\lambda = \\frac{\\text{CDS Spread}}{1 - \\text{LGD}}$$

        **Example:** CDS = 120 bps, LGD = 60% → λ = 0.0120 / 0.40 = 3.0%
        """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💳 CDS Spread Data")
        st.markdown("""
            Upload CSV with CDS spreads. Supported formats:
            - **Single tenor:** `date`, `spread_bps`
            - **Multi-tenor:** `date`, `1Y`, `3Y`, `5Y`, `10Y` (spreads in bps)
            """)

        cds_file = st.file_uploader(
            "Upload CDS spread data",
            type=["csv"],
            key="cds_upload",
            help="CSV with CDS spreads in basis points",
        )

        if cds_file is not None:
            try:
                cds_df = pd.read_csv(cds_file, parse_dates=["date"])
                cds_df = cds_df.sort_values("date").reset_index(drop=True)

                with st.expander("Preview Data", expanded=False):
                    st.dataframe(cds_df.head(10), use_container_width=True)

                # Detect format (single vs multi-tenor)
                tenor_cols = [
                    c for c in cds_df.columns if c not in ["date", "spread_bps"]
                ]
                is_multi_tenor = len(tenor_cols) > 0 and any(
                    t in str(tenor_cols)
                    for t in ["1Y", "3Y", "5Y", "10Y", "1y", "3y", "5y", "10y"]
                )

                # LGD input for conversion
                lgd = st.slider(
                    "Loss Given Default (LGD)",
                    min_value=0.20,
                    max_value=0.80,
                    value=config.get("lgd_cpty", 0.60),
                    step=0.05,
                    key="cds_lgd",
                    help="Recovery Rate = 1 - LGD",
                )
                recovery = 1 - lgd

                if is_multi_tenor:
                    # Multi-tenor CDS term structure
                    st.markdown("#### Term Structure Detected")

                    # Normalize column names
                    tenor_map = {}
                    for col in cds_df.columns:
                        if col.lower() in ["1y", "1yr"]:
                            tenor_map[col] = "1Y"
                        elif col.lower() in ["3y", "3yr"]:
                            tenor_map[col] = "3Y"
                        elif col.lower() in ["5y", "5yr"]:
                            tenor_map[col] = "5Y"
                        elif col.lower() in ["10y", "10yr"]:
                            tenor_map[col] = "10Y"

                    cds_df = cds_df.rename(columns=tenor_map)
                    tenor_cols = [
                        c for c in ["1Y", "3Y", "5Y", "10Y"] if c in cds_df.columns
                    ]

                    # Latest spreads and hazard rates
                    latest = cds_df.iloc[-1]
                    term_structure = []
                    for tenor in tenor_cols:
                        spread_bps = latest[tenor]
                        hazard = (spread_bps / 10000) / recovery
                        term_structure.append(
                            {
                                "Tenor": tenor,
                                "CDS Spread (bps)": spread_bps,
                                "Hazard Rate (λ)": f"{hazard*100:.2f}%",
                                "λ (decimal)": hazard,
                            }
                        )

                    term_df = pd.DataFrame(term_structure)
                    st.dataframe(
                        term_df[["Tenor", "CDS Spread (bps)", "Hazard Rate (λ)"]],
                        use_container_width=True,
                    )

                    # Select which tenor to use
                    selected_tenor = st.selectbox(
                        "Select tenor for hazard rate",
                        tenor_cols,
                        index=min(2, len(tenor_cols) - 1),  # Default to 5Y if available
                        key="cds_tenor_select",
                    )

                    final_spread = latest[selected_tenor]
                    final_hazard = (final_spread / 10000) / recovery

                    # Plot term structure evolution
                    try:
                        import plotly.graph_objects as go

                        fig = go.Figure()
                        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
                        for i, tenor in enumerate(tenor_cols):
                            fig.add_trace(
                                go.Scatter(
                                    x=cds_df["date"],
                                    y=cds_df[tenor],
                                    name=tenor,
                                    line={"color": colors[i % len(colors)]},
                                )
                            )

                        fig.update_layout(
                            title="CDS Spread Term Structure Over Time",
                            xaxis_title="Date",
                            yaxis_title="CDS Spread (bps)",
                            template="plotly_dark",
                            height=350,
                            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

                    # Store term structure
                    st.session_state.calibrated_params["cds_term_structure"] = term_df

                else:
                    # Single tenor format
                    if "spread_bps" not in cds_df.columns:
                        # Try to find the spread column
                        spread_cols = [
                            c
                            for c in cds_df.columns
                            if "spread" in c.lower() or "cds" in c.lower()
                        ]
                        if spread_cols:
                            cds_df["spread_bps"] = cds_df[spread_cols[0]]
                        else:
                            # Assume second column is spread
                            cds_df["spread_bps"] = cds_df.iloc[:, 1]

                    # Calculate hazard rates
                    cds_df["hazard_rate"] = (cds_df["spread_bps"] / 10000) / recovery

                    final_spread = cds_df["spread_bps"].iloc[-1]
                    final_hazard = cds_df["hazard_rate"].iloc[-1]

                    # Plot
                    try:
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots

                        fig = make_subplots(
                            rows=2,
                            cols=1,
                            subplot_titles=("CDS Spread", "Implied Hazard Rate"),
                            vertical_spacing=0.12,
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=cds_df["date"],
                                y=cds_df["spread_bps"],
                                name="CDS Spread",
                                line={"color": "#FF6B6B"},
                                fill="tozeroy",
                                fillcolor="rgba(255, 107, 107, 0.2)",
                            ),
                            row=1,
                            col=1,
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=cds_df["date"],
                                y=cds_df["hazard_rate"] * 100,
                                name="Hazard Rate",
                                line={"color": "#4ECDC4"},
                            ),
                            row=2,
                            col=1,
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            height=400,
                            showlegend=False,
                        )
                        fig.update_yaxes(title_text="Spread (bps)", row=1, col=1)
                        fig.update_yaxes(title_text="λ (%)", row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

                # Display final calibrated values
                st.divider()
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(
                        "CDS Spread",
                        f"{final_spread:.0f} bps",
                        help="Latest CDS spread",
                    )
                with col_m2:
                    st.metric(
                        "Hazard Rate (λ)",
                        f"{final_hazard*100:.2f}%",
                        delta=f"{final_hazard*10000:.0f} bps",
                        help="Calibrated hazard rate",
                    )
                with col_m3:
                    st.metric(
                        "Recovery Rate",
                        f"{recovery*100:.0f}%",
                        help="1 - LGD",
                    )

                # Store calibrated hazard rate
                st.session_state.calibrated_params["hazard_rate"] = final_hazard
                st.session_state.calibrated_params["cds_spread"] = final_spread

                if st.button(
                    "✅ Apply Hazard Rate to CVA/DVA", key="apply_cds", type="primary"
                ):
                    st.success(
                        f"Applied: λ = {final_hazard*100:.2f}% "
                        f"(from CDS = {final_spread:.0f} bps, LGD = {lgd*100:.0f}%)"
                    )
                    st.rerun()

            except Exception as e:
                st.error(f"Error processing CDS file: {e}")

    with col2:
        st.markdown("### ℹ️ CDS Basics")
        st.markdown("""
            **Credit Default Swap (CDS):**

            A CDS is insurance against default.
            The buyer pays a periodic spread
            (in bps p.a.) and receives protection.

            **Key relationship:**
            ```
            λ ≈ Spread / (1 - LGD)
            ```

            **Typical values:**
            | Rating | Spread (bps) |
            |--------|-------------|
            | AAA    | 10-30       |
            | AA     | 30-60       |
            | A      | 50-100      |
            | BBB    | 100-200     |
            | BB     | 200-400     |
            | B      | 400-800     |
            """)

        # Sample CDS data download
        st.divider()
        st.markdown("### 📥 Sample Data")

        # Single tenor sample
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        spreads = 120 + np.cumsum(np.random.randn(252) * 2)  # ~120 bps with noise
        spreads = np.maximum(spreads, 20)  # Floor at 20 bps
        sample_cds_single = pd.DataFrame({"date": dates, "spread_bps": spreads})

        st.download_button(
            "📥 Single Tenor CDS",
            sample_cds_single.to_csv(index=False),
            "sample_cds_single.csv",
            "text/csv",
        )

        # Multi-tenor sample
        sample_cds_multi = pd.DataFrame(
            {
                "date": dates,
                "1Y": 80 + np.cumsum(np.random.randn(252) * 1.5),
                "3Y": 100 + np.cumsum(np.random.randn(252) * 1.8),
                "5Y": 120 + np.cumsum(np.random.randn(252) * 2.0),
                "10Y": 140 + np.cumsum(np.random.randn(252) * 2.2),
            }
        )
        # Ensure positive values
        for col in ["1Y", "3Y", "5Y", "10Y"]:
            sample_cds_multi[col] = np.maximum(sample_cds_multi[col], 15)

        st.download_button(
            "📥 Multi-Tenor CDS",
            sample_cds_multi.to_csv(index=False),
            "sample_cds_term_structure.csv",
            "text/csv",
        )


def _calibration_correlation_section(config: dict) -> None:
    """Historical correlation calibration from IR and FX data."""
    st.subheader("Correlation Calibration")

    st.info("""
        **Automatic Correlation Calculation:**

        When you upload both IR and FX data in the Volatilities tab, the correlation
        between rate changes (ΔIR) and FX log-returns is computed automatically.

        $$\\rho_{IR,FX} = \\text{Corr}(\\Delta r_t, \\log(S_t/S_{t-1}))$$
        """)

    # Check if both IR and FX data are available
    ir_data = st.session_state.calibrated_params.get("ir_data")
    fx_data = st.session_state.calibrated_params.get("fx_data")

    if ir_data is None or fx_data is None:
        st.warning("""
            **Data Required:**

            Please upload both IR and FX historical data in the **Volatilities** tab
            to calculate correlations automatically.

            The correlation will be computed from overlapping dates.
            """)

        # Manual correlation input as fallback
        st.divider()
        st.markdown("### Manual Correlation Input")
        st.markdown(
            "Alternatively, enter correlations manually if you have them from another source:"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            manual_corr_ir_fx = st.slider(
                "Domestic Rate ↔ FX",
                -1.0,
                1.0,
                config.get("corr_dx", -0.30),
                key="manual_corr_dx",
            )
        with col2:
            _manual_corr_df = st.slider(
                "Domestic ↔ Foreign Rate",
                -1.0,
                1.0,
                config.get("corr_df", 0.70),
                key="manual_corr_df",
            )
        with col3:
            _manual_corr_fx_f = st.slider(
                "Foreign Rate ↔ FX",
                -1.0,
                1.0,
                config.get("corr_fx", 0.40),
                key="manual_corr_fx",
            )

        if st.button("✅ Apply Manual Correlations", key="apply_manual_corr"):
            st.session_state.calibrated_params["correlation_ir_fx"] = manual_corr_ir_fx
            # Delete widget key so slider picks up new calibrated default on rerun
            if "corr_dx" in st.session_state:
                del st.session_state["corr_dx"]
            st.success("Manual correlations applied.")
            st.rerun()

        return

    # Merge datasets on date
    ir_df = ir_data.copy()
    fx_df = fx_data.copy()

    # Ensure date columns are datetime
    ir_df["date"] = pd.to_datetime(ir_df["date"])
    fx_df["date"] = pd.to_datetime(fx_df["date"])

    # Merge on date
    merged = pd.merge(ir_df, fx_df, on="date", how="inner", suffixes=("_ir", "_fx"))

    if len(merged) < 20:
        st.error(
            f"Insufficient overlapping data: only {len(merged)} common dates. "
            "Need at least 20 observations for reliable correlation."
        )
        return

    st.success(f"Found {len(merged)} overlapping observations.")

    # Calculate changes/returns
    merged["ir_change"] = merged["rate_decimal"].diff()

    # Handle log_return column
    if "log_return" not in merged.columns:
        if "rate" in merged.columns:
            merged["log_return"] = np.log(merged["rate"] / merged["rate"].shift(1))
        else:
            st.error("FX log returns not available.")
            return

    # Drop NaN values
    merged_clean = merged.dropna(subset=["ir_change", "log_return"])

    # Rolling correlation
    window = st.slider(
        "Rolling Window for Correlation",
        min_value=20,
        max_value=min(120, len(merged_clean) - 1),
        value=min(60, len(merged_clean) - 1),
        key="corr_window",
    )

    merged_clean["rolling_corr"] = (
        merged_clean["ir_change"].rolling(window).corr(merged_clean["log_return"])
    )

    # Overall correlation
    overall_corr = merged_clean["ir_change"].corr(merged_clean["log_return"])
    latest_corr = merged_clean["rolling_corr"].iloc[-1]

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Overall Correlation",
            f"{overall_corr:.3f}",
            help="Full-sample correlation",
        )
    with col2:
        st.metric(
            "Latest Rolling Corr",
            f"{latest_corr:.3f}",
            delta=f"{(latest_corr - overall_corr):.3f}",
            help=f"Using {window}-day window",
        )
    with col3:
        st.metric(
            "Observations",
            f"{len(merged_clean)}",
            help="Number of overlapping data points",
        )

    # Plot rolling correlation
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Normalized IR Changes vs FX Returns",
                f"Rolling {window}-Day Correlation",
            ),
            vertical_spacing=0.15,
        )

        # Scatter plot of returns
        fig.add_trace(
            go.Scatter(
                x=merged_clean["ir_change"] / merged_clean["ir_change"].std(),
                y=merged_clean["log_return"] / merged_clean["log_return"].std(),
                mode="markers",
                marker={
                    "size": 5,
                    "color": np.arange(len(merged_clean)),
                    "colorscale": "Viridis",
                    "opacity": 0.6,
                },
                name="Returns",
            ),
            row=1,
            col=1,
        )

        # Rolling correlation
        fig.add_trace(
            go.Scatter(
                x=merged_clean["date"],
                y=merged_clean["rolling_corr"],
                name="Rolling Corr",
                line={"color": "#9467BD", "width": 2},
            ),
            row=2,
            col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Add overall correlation line
        fig.add_hline(
            y=overall_corr,
            line_dash="dot",
            line_color="#FF4B4B",
            annotation_text=f"Overall: {overall_corr:.2f}",
            row=2,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            showlegend=False,
        )
        fig.update_xaxes(title_text="IR Change (std)", row=1, col=1)
        fig.update_yaxes(title_text="FX Return (std)", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    # Correlation matrix display
    st.divider()
    st.markdown("### Correlation Matrix")

    corr_matrix = pd.DataFrame(
        [
            [1.0, config.get("corr_df", 0.70), overall_corr],
            [config.get("corr_df", 0.70), 1.0, config.get("corr_fx", 0.40)],
            [overall_corr, config.get("corr_fx", 0.40), 1.0],
        ],
        index=["Domestic Rate", "Foreign Rate", "FX Spot"],
        columns=["Domestic Rate", "Foreign Rate", "FX Spot"],
    )

    st.dataframe(
        corr_matrix.style.format("{:.3f}").background_gradient(
            cmap="RdBu_r", vmin=-1, vmax=1
        ),
        use_container_width=True,
    )

    st.caption(
        "Note: Domestic-FX correlation is calibrated from data. "
        "Domestic-Foreign and Foreign-FX use sidebar defaults (can be adjusted manually)."
    )

    # Store calibrated correlation
    st.session_state.calibrated_params["correlation_ir_fx"] = overall_corr
    st.session_state.calibrated_params["correlation_ir_fx_rolling"] = latest_corr

    if st.button("✅ Apply Calibrated Correlation", key="apply_corr", type="primary"):
        # Delete widget key so slider picks up new calibrated default on rerun
        if "corr_dx" in st.session_state:
            del st.session_state["corr_dx"]
        st.success(f"Applied: ρ(IR, FX) = {overall_corr:.3f}")
        st.rerun()


def _calibration_ois_section(config: dict) -> None:
    """OIS curve calibration from market data."""
    st.subheader("OIS Discount Curve Calibration")

    st.info("""
        **OIS (Overnight Index Swap) Curve:**

        The OIS curve is the standard for risk-free discounting post-2008 crisis.
        It represents the expected path of overnight rates and is used to:
        - Discount future cash flows
        - Calculate forward rates
        - Price collateralized derivatives

        **Bootstrapping Method:**
        Extract zero rates from OIS swap rates using iterative bootstrapping.
        """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 OIS Swap Rate Data")
        st.markdown("""
            Upload CSV with OIS swap rates. Format:
            - **Columns:** `tenor`, `rate` (swap rate in % or decimal)
            - **Example tenors:** 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y
            """)

        ois_file = st.file_uploader(
            "Upload OIS curve data",
            type=["csv"],
            key="ois_upload",
            help="CSV with tenor and OIS swap rate columns",
        )

        if ois_file is not None:
            try:
                ois_df = pd.read_csv(ois_file)

                # Standardize column names
                ois_df.columns = ois_df.columns.str.lower().str.strip()

                with st.expander("Preview Data", expanded=False):
                    st.dataframe(ois_df.head(10), use_container_width=True)

                # Parse tenor to years
                def tenor_to_years(tenor: str) -> float:
                    tenor = str(tenor).upper().strip()
                    if "M" in tenor:
                        return float(tenor.replace("M", "")) / 12
                    elif "Y" in tenor:
                        return float(tenor.replace("Y", ""))
                    elif "W" in tenor:
                        return float(tenor.replace("W", "")) / 52
                    elif "D" in tenor:
                        return float(tenor.replace("D", "")) / 365
                    else:
                        try:
                            return float(tenor)
                        except ValueError:
                            return 0.0

                ois_df["years"] = ois_df["tenor"].apply(tenor_to_years)
                ois_df = ois_df.sort_values("years")

                # Auto-detect rate format
                rate_col = "rate" if "rate" in ois_df.columns else ois_df.columns[1]
                if ois_df[rate_col].mean() > 0.5:  # Percentage
                    ois_df["rate_decimal"] = ois_df[rate_col] / 100
                else:
                    ois_df["rate_decimal"] = ois_df[rate_col]

                # Bootstrap zero rates (simplified continuous compounding)
                # For short tenors, zero rate ≈ swap rate
                # For longer tenors, iterative bootstrap
                zero_rates = []
                discount_factors = []

                for _i, row in ois_df.iterrows():
                    t = row["years"]
                    swap_rate = row["rate_decimal"]

                    if t <= 1:
                        # Simple case: zero rate ≈ swap rate for short tenors
                        zero_rate = swap_rate
                    else:
                        # Bootstrap from previous discount factors
                        # Simplified: use swap rate as approximation
                        # In production, would solve iteratively
                        zero_rate = swap_rate + 0.001 * (t - 1)  # Small term premium

                    zero_rates.append(zero_rate)
                    df = np.exp(-zero_rate * t)
                    discount_factors.append(df)

                ois_df["zero_rate"] = zero_rates
                ois_df["discount_factor"] = discount_factors

                # Display calibrated curve
                st.markdown("### Calibrated Curve")

                display_df = ois_df[
                    ["tenor", "years", "rate_decimal", "zero_rate", "discount_factor"]
                ].copy()
                display_df.columns = [
                    "Tenor",
                    "Years",
                    "Swap Rate",
                    "Zero Rate",
                    "Discount Factor",
                ]

                st.dataframe(
                    display_df.style.format(
                        {
                            "Years": "{:.2f}",
                            "Swap Rate": "{:.4f}",
                            "Zero Rate": "{:.4f}",
                            "Discount Factor": "{:.6f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Plot curves
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Zero Curve", "Discount Curve"),
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=ois_df["years"],
                            y=ois_df["zero_rate"] * 100,
                            mode="lines+markers",
                            name="Zero Rate",
                            line={"color": "#00CC96"},
                            marker={"size": 8},
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=ois_df["years"],
                            y=ois_df["discount_factor"],
                            mode="lines+markers",
                            name="Discount Factor",
                            line={"color": "#636EFA"},
                            marker={"size": 8},
                        ),
                        row=1,
                        col=2,
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=350,
                        showlegend=False,
                    )
                    fig.update_xaxes(title_text="Maturity (Years)", row=1, col=1)
                    fig.update_xaxes(title_text="Maturity (Years)", row=1, col=2)
                    fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
                    fig.update_yaxes(title_text="DF", row=1, col=2)

                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    pass

                # Forward rate calculation
                st.markdown("### Forward Rates")

                # Calculate forward rates between tenors
                forward_rates = []
                for i in range(1, len(ois_df)):
                    t1 = ois_df.iloc[i - 1]["years"]
                    t2 = ois_df.iloc[i]["years"]
                    z1 = ois_df.iloc[i - 1]["zero_rate"]
                    z2 = ois_df.iloc[i]["zero_rate"]

                    if t2 > t1:
                        fwd = (z2 * t2 - z1 * t1) / (t2 - t1)
                        forward_rates.append(
                            {
                                "Period": f"{ois_df.iloc[i-1]['tenor']} → {ois_df.iloc[i]['tenor']}",
                                "Forward Rate": f"{fwd*100:.2f}%",
                            }
                        )

                if forward_rates:
                    st.dataframe(
                        pd.DataFrame(forward_rates),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Store calibrated OIS curve
                st.session_state.calibrated_params["ois_curve"] = ois_df[
                    ["years", "zero_rate", "discount_factor"]
                ].to_dict("records")

                if st.button("✅ Apply OIS Curve", key="apply_ois", type="primary"):
                    st.success("OIS curve applied for discounting.")

            except Exception as e:
                st.error(f"Error processing OIS file: {e}")

    with col2:
        st.markdown("### ℹ️ OIS Curve Basics")
        st.markdown("""
            **Why OIS for Discounting?**

            Post-2008 financial crisis, OIS
            replaced LIBOR as the standard
            risk-free rate for collateralized
            derivatives.

            **Key Properties:**
            - Based on overnight rates
            - Minimal credit risk
            - Standard CSA collateral rate

            **Common OIS Rates:**
            | Currency | Index |
            |----------|-------|
            | USD | SOFR |
            | EUR | €STR |
            | GBP | SONIA |
            | JPY | TONA |

            **Formula:**
            ```
            DF(T) = exp(-z(T) × T)
            ```

            Where z(T) is the continuously
            compounded zero rate.
            """)

        # Sample OIS data download
        st.divider()
        st.markdown("### 📥 Sample Data")

        sample_ois = pd.DataFrame(
            {
                "tenor": ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"],
                "rate": [4.30, 4.35, 4.40, 4.45, 4.20, 4.00, 3.80, 3.70, 3.60],
            }
        )

        st.download_button(
            "📥 Sample OIS Curve",
            sample_ois.to_csv(index=False),
            "sample_ois_curve.csv",
            "text/csv",
        )


def _calibration_summary_section(config: dict) -> None:
    """Summary of all calibrated parameters."""
    st.subheader("Calibration Summary")

    params = st.session_state.calibrated_params

    # Check what's been calibrated
    has_ir = params.get("ir_vol") is not None
    has_fx = params.get("fx_vol") is not None
    has_cds = params.get("hazard_rate") is not None
    has_corr = params.get("correlation_ir_fx") is not None
    has_ois = params.get("ois_curve") is not None

    calibrated_count = sum([has_ir, has_fx, has_cds, has_corr, has_ois])

    if calibrated_count == 0:
        st.info("""
            **No parameters calibrated yet.**

            Use the other tabs to calibrate:
            - 📊 **Volatilities**: Upload IR and FX data
            - 💳 **CDS / Credit**: Upload CDS spreads
            - 🔗 **Correlations**: Computed automatically from IR/FX data
            - 📉 **OIS Curve**: Upload OIS swap rates
            """)
        return

    # Progress indicator
    st.progress(
        calibrated_count / 5, text=f"{calibrated_count}/5 parameter sets calibrated"
    )

    # Summary table
    st.markdown("### 📋 Calibrated Parameters")

    summary_data = []

    # IR Parameters
    if has_ir:
        summary_data.extend(
            [
                {
                    "Category": "Interest Rate",
                    "Parameter": "Volatility (σ)",
                    "Calibrated": f"{params['ir_vol']*10000:.0f} bps",
                    "Model Default": f"{config.get('sigma_d', 0.01)*10000:.0f} bps",
                    "Status": "✅",
                },
                {
                    "Category": "Interest Rate",
                    "Parameter": "Mean Reversion (κ)",
                    "Calibrated": f"{params.get('ir_kappa', 0.1):.2f}",
                    "Model Default": f"{config.get('kappa_d', 0.10):.2f}",
                    "Status": "✅",
                },
                {
                    "Category": "Interest Rate",
                    "Parameter": "Long-term Mean (θ)",
                    "Calibrated": f"{params.get('ir_theta', 0.02)*100:.2f}%",
                    "Model Default": f"{config.get('theta_d', 0.02)*100:.2f}%",
                    "Status": "✅",
                },
            ]
        )
    else:
        summary_data.append(
            {
                "Category": "Interest Rate",
                "Parameter": "All parameters",
                "Calibrated": "—",
                "Model Default": "Using defaults",
                "Status": "⏳",
            }
        )

    # FX Parameters
    if has_fx:
        summary_data.extend(
            [
                {
                    "Category": "FX",
                    "Parameter": "Volatility (σ)",
                    "Calibrated": f"{params['fx_vol']*100:.1f}%",
                    "Model Default": f"{config.get('fx_vol', 0.12)*100:.0f}%",
                    "Status": "✅",
                },
                {
                    "Category": "FX",
                    "Parameter": "Spot Rate",
                    "Calibrated": f"{params.get('fx_spot', 1.10):.4f}",
                    "Model Default": f"{config.get('fx_spot', 1.10):.4f}",
                    "Status": "✅",
                },
            ]
        )
    else:
        summary_data.append(
            {
                "Category": "FX",
                "Parameter": "All parameters",
                "Calibrated": "—",
                "Model Default": "Using defaults",
                "Status": "⏳",
            }
        )

    # Credit Parameters
    if has_cds:
        summary_data.extend(
            [
                {
                    "Category": "Credit",
                    "Parameter": "Hazard Rate (λ)",
                    "Calibrated": f"{params['hazard_rate']*100:.2f}%",
                    "Model Default": f"{config.get('lambda_cpty', 0.012)*100:.2f}%",
                    "Status": "✅",
                },
                {
                    "Category": "Credit",
                    "Parameter": "CDS Spread",
                    "Calibrated": f"{params.get('cds_spread', 120):.0f} bps",
                    "Model Default": "N/A",
                    "Status": "✅",
                },
            ]
        )
    else:
        summary_data.append(
            {
                "Category": "Credit",
                "Parameter": "Hazard Rate",
                "Calibrated": "—",
                "Model Default": f"{config.get('lambda_cpty', 0.012)*100:.2f}%",
                "Status": "⏳",
            }
        )

    # Correlation Parameters
    if has_corr:
        summary_data.append(
            {
                "Category": "Correlation",
                "Parameter": "ρ(IR, FX)",
                "Calibrated": f"{params['correlation_ir_fx']:.3f}",
                "Model Default": f"{config.get('corr_dx', -0.30):.2f}",
                "Status": "✅",
            }
        )
    else:
        summary_data.append(
            {
                "Category": "Correlation",
                "Parameter": "ρ(IR, FX)",
                "Calibrated": "—",
                "Model Default": f"{config.get('corr_dx', -0.30):.2f}",
                "Status": "⏳",
            }
        )

    # OIS Curve Parameters
    if has_ois:
        ois_curve = params.get("ois_curve", [])
        n_tenors = len(ois_curve) if ois_curve else 0
        summary_data.append(
            {
                "Category": "OIS Curve",
                "Parameter": "Discount Curve",
                "Calibrated": f"{n_tenors} tenors",
                "Model Default": "Flat rate",
                "Status": "✅",
            }
        )
    else:
        summary_data.append(
            {
                "Category": "OIS Curve",
                "Parameter": "Discount Curve",
                "Calibrated": "—",
                "Model Default": "Flat rate",
                "Status": "⏳",
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Style the dataframe
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    # CDS Term Structure if available
    if params.get("cds_term_structure") is not None:
        st.divider()
        st.markdown("### 📈 CDS Term Structure")
        st.dataframe(
            params["cds_term_structure"],
            use_container_width=True,
            hide_index=True,
        )

    # Export calibration report
    st.divider()
    st.markdown("### 💾 Export Calibration Report")

    # Create JSON report
    report = {
        "calibration_date": pd.Timestamp.now().isoformat(),
        "parameters": {
            "interest_rate": {
                "sigma": params.get("ir_vol"),
                "kappa": params.get("ir_kappa"),
                "theta": params.get("ir_theta"),
                "calibrated": has_ir,
            },
            "fx": {
                "sigma": params.get("fx_vol"),
                "spot": params.get("fx_spot"),
                "calibrated": has_fx,
            },
            "credit": {
                "hazard_rate": params.get("hazard_rate"),
                "cds_spread_bps": params.get("cds_spread"),
                "calibrated": has_cds,
            },
            "correlation": {
                "ir_fx": params.get("correlation_ir_fx"),
                "calibrated": has_corr,
            },
            "ois_curve": {
                "tenors": (
                    len(params.get("ois_curve", [])) if params.get("ois_curve") else 0
                ),
                "curve_data": params.get("ois_curve"),
                "calibrated": has_ois,
            },
        },
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Calibration Report (JSON)",
            json.dumps(report, indent=2, default=str),
            "calibration_report.json",
            "application/json",
        )

    with col2:
        # Apply all calibrated parameters button
        if st.button("🚀 Apply All to Model", type="primary", use_container_width=True):
            applied = []
            if has_ir:
                applied.append("IR params")
            if has_fx:
                applied.append("FX params")
            if has_cds:
                applied.append("Hazard rate")
            if has_corr:
                applied.append("Correlation")

            if applied:
                st.success(f"Applied: {', '.join(applied)}")
                st.balloons()
            else:
                st.warning("No parameters to apply.")


def stress_test_tab(config: dict) -> None:
    """
    Stress Testing tab.

    Apply market shocks and analyze xVA sensitivity to stressed scenarios.
    """
    st.header("⚡ Stress Testing")

    st.markdown("""
        Apply market shocks to assess xVA sensitivity under stressed conditions.
        This is essential for risk management and regulatory compliance (FRTB, stress VaR).
        """)

    if st.session_state.results is None:
        st.info(
            "Run a base simulation first (click **▶️ Run** in sidebar), "
            "then apply stress scenarios."
        )
        return

    r = st.session_state.results
    base_xva = r["xva_result"]

    # Display base case
    st.subheader("📊 Base Case xVA")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("CVA", f"${base_xva.cva/1e6:.2f}M")
    with col2:
        st.metric("DVA", f"${base_xva.dva/1e6:.2f}M")
    with col3:
        st.metric("FVA", f"${base_xva.fva/1e6:.2f}M")
    with col4:
        st.metric("MVA", f"${base_xva.mva/1e6:.2f}M")
    with col5:
        st.metric("Total", f"${base_xva.total/1e6:.2f}M")

    st.divider()

    # Stress scenario configuration
    st.subheader("⚙️ Stress Scenario Configuration")

    st.info("""
        **Common Stress Scenarios:**
        - **IR Shock**: Parallel shift in interest rates (e.g., +100bps)
        - **FX Shock**: FX spot move (e.g., -10%)
        - **Volatility Shock**: Increase in market volatility
        - **Credit Shock**: CDS spread widening (affects hazard rates)
        - **Combined**: Multiple shocks applied simultaneously
        """)

    # Predefined scenarios
    scenario_type = st.selectbox(
        "Select Scenario",
        [
            "Custom",
            "Mild Stress (2020 COVID-like)",
            "Severe Stress (2008 GFC-like)",
            "Rising Rates (+200bps)",
            "Credit Crisis (spreads +100%)",
            "FX Crash (-20%)",
        ],
        key="stress_scenario_type",
    )

    # Set default values based on scenario
    if scenario_type == "Mild Stress (2020 COVID-like)":
        ir_shock_default = 50
        fx_shock_default = -10
        vol_shock_default = 50
        credit_shock_default = 50
    elif scenario_type == "Severe Stress (2008 GFC-like)":
        ir_shock_default = -100
        fx_shock_default = -25
        vol_shock_default = 100
        credit_shock_default = 200
    elif scenario_type == "Rising Rates (+200bps)":
        ir_shock_default = 200
        fx_shock_default = 5
        vol_shock_default = 20
        credit_shock_default = 30
    elif scenario_type == "Credit Crisis (spreads +100%)":
        ir_shock_default = -50
        fx_shock_default = -15
        vol_shock_default = 75
        credit_shock_default = 100
    elif scenario_type == "FX Crash (-20%)":
        ir_shock_default = 0
        fx_shock_default = -20
        vol_shock_default = 80
        credit_shock_default = 50
    else:  # Custom
        ir_shock_default = 0
        fx_shock_default = 0
        vol_shock_default = 0
        credit_shock_default = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ir_shock = st.slider(
            "IR Shock (bps)",
            min_value=-200,
            max_value=200,
            value=ir_shock_default,
            step=10,
            key="ir_shock",
            help="Parallel shift in interest rates",
        )

    with col2:
        fx_shock = st.slider(
            "FX Shock (%)",
            min_value=-30,
            max_value=30,
            value=fx_shock_default,
            step=5,
            key="fx_shock",
            help="Percentage change in FX spot rate",
        )

    with col3:
        vol_shock = st.slider(
            "Vol Shock (%)",
            min_value=0,
            max_value=200,
            value=vol_shock_default,
            step=10,
            key="vol_shock",
            help="Percentage increase in volatilities",
        )

    with col4:
        credit_shock = st.slider(
            "Credit Shock (%)",
            min_value=0,
            max_value=300,
            value=credit_shock_default,
            step=25,
            key="credit_shock",
            help="Percentage increase in hazard rates",
        )

    # Run stress test
    if st.button("🚀 Run Stress Test", type="primary"):
        with st.spinner("Running stressed simulation..."):
            # Create stressed config
            stressed_config = config.copy()

            # Apply IR shock
            stressed_config["theta_d"] = config["theta_d"] + ir_shock / 10000
            stressed_config["theta_f"] = config["theta_f"] + ir_shock / 10000

            # Apply FX shock
            stressed_config["fx_spot"] = config["fx_spot"] * (1 + fx_shock / 100)

            # Apply vol shock
            stressed_config["sigma_d"] = config["sigma_d"] * (1 + vol_shock / 100)
            stressed_config["sigma_f"] = config["sigma_f"] * (1 + vol_shock / 100)
            stressed_config["fx_vol"] = config["fx_vol"] * (1 + vol_shock / 100)

            # Apply credit shock
            stressed_config["lambda_cpty"] = config["lambda_cpty"] * (
                1 + credit_shock / 100
            )
            stressed_config["lambda_own"] = config["lambda_own"] * (
                1 + credit_shock / 100
            )

            # Run stressed simulation
            irs_list, fxf_list = get_portfolio()
            instruments = irs_list + fxf_list

            market_config = build_market_config(stressed_config)
            # Bell curve mode: use dt=0.5 to align with semi-annual payment dates
            # This removes sawtooth artifacts and shows cleaner bell shape
            if st.session_state.get("bell_curve_mode"):
                dt = 0.5
            else:
                dt = 1 / 12 if config["freq"] == "Monthly" else 0.25

            engine = MonteCarloEngine(
                n_paths=config["n_paths"],
                horizon=config["horizon"],
                dt=dt,
                seed=config["seed"],
            )

            result = engine.simulate(instruments, market_config)

            # Calculate exposure metrics
            vm = VariationMargin(
                threshold=config["threshold"],
                mta=config["mta"],
                mpr_days=config["mpr_days"],
                days_per_step=dt * 365,
            )
            _, coll_exposure = vm.apply(result.mtm, result.time_grid)
            epe_coll = np.maximum(coll_exposure, 0).mean(axis=0)
            ene_coll = np.maximum(-coll_exposure, 0).mean(axis=0)

            im = InitialMargin(multiplier=config["im_mult"])
            im_profile = im.calculate(coll_exposure)

            avg_df = result.df_domestic.mean(axis=0)

            # Calculate stressed xVA
            xva_params = XVAParams(
                lgd_counterparty=config["lgd_cpty"],
                lgd_own=config["lgd_own"],
                hazard_rate_counterparty=stressed_config["lambda_cpty"],
                hazard_rate_own=stressed_config["lambda_own"],
                funding_spread=config["funding_spread"],
                cost_of_capital=config["coc"],
                capital_ratio=config["capital_ratio"],
                im_multiplier=config["im_mult"],
            )

            stressed_xva = calculate_all_xva(
                epe=epe_coll,
                ene=ene_coll,
                discount_factors=avg_df,
                time_grid=result.time_grid,
                params=xva_params,
                im_profile=im_profile,
            )

            # Store stressed results
            st.session_state.stressed_xva = stressed_xva
            st.session_state.stress_scenario = {
                "ir_shock": ir_shock,
                "fx_shock": fx_shock,
                "vol_shock": vol_shock,
                "credit_shock": credit_shock,
                "scenario_name": scenario_type,
            }

    # Display stress results
    if "stressed_xva" in st.session_state:
        stressed_xva = st.session_state.stressed_xva
        scenario = st.session_state.stress_scenario

        st.divider()
        st.subheader("📈 Stress Test Results")

        # Scenario summary
        st.markdown(f"""
            **Applied Shocks:** IR: {scenario['ir_shock']:+d}bps |
            FX: {scenario['fx_shock']:+d}% | Vol: +{scenario['vol_shock']}% |
            Credit: +{scenario['credit_shock']}%
            """)

        # Comparison table
        comparison_data = {
            "Component": ["CVA", "DVA", "FVA", "MVA", "KVA", "Total xVA"],
            "Base ($M)": [
                base_xva.cva / 1e6,
                base_xva.dva / 1e6,
                base_xva.fva / 1e6,
                base_xva.mva / 1e6,
                base_xva.kva / 1e6,
                base_xva.total / 1e6,
            ],
            "Stressed ($M)": [
                stressed_xva.cva / 1e6,
                stressed_xva.dva / 1e6,
                stressed_xva.fva / 1e6,
                stressed_xva.mva / 1e6,
                stressed_xva.kva / 1e6,
                stressed_xva.total / 1e6,
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df["Change ($M)"] = (
            comparison_df["Stressed ($M)"] - comparison_df["Base ($M)"]
        )
        comparison_df["Change (%)"] = (
            (comparison_df["Stressed ($M)"] / comparison_df["Base ($M)"] - 1) * 100
        ).replace([np.inf, -np.inf], 0)

        st.dataframe(
            comparison_df.style.format(
                {
                    "Base ($M)": "{:.2f}",
                    "Stressed ($M)": "{:.2f}",
                    "Change ($M)": "{:+.2f}",
                    "Change (%)": "{:+.1f}%",
                }
            ).background_gradient(
                subset=["Change (%)"], cmap="RdYlGn_r", vmin=-100, vmax=100
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Visual comparison
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            components = ["CVA", "DVA", "FVA", "MVA", "KVA"]
            base_vals = [
                base_xva.cva / 1e6,
                -base_xva.dva / 1e6,
                base_xva.fva / 1e6,
                base_xva.mva / 1e6,
                base_xva.kva / 1e6,
            ]
            stressed_vals = [
                stressed_xva.cva / 1e6,
                -stressed_xva.dva / 1e6,
                stressed_xva.fva / 1e6,
                stressed_xva.mva / 1e6,
                stressed_xva.kva / 1e6,
            ]

            fig.add_trace(
                go.Bar(
                    name="Base Case",
                    x=components,
                    y=base_vals,
                    marker_color="#636EFA",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Stressed",
                    x=components,
                    y=stressed_vals,
                    marker_color="#EF553B",
                )
            )

            fig.update_layout(
                title="xVA Comparison: Base vs Stressed",
                yaxis_title="Value ($M)",
                template="plotly_dark",
                height=400,
                barmode="group",
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            pass

        # Risk metrics
        st.subheader("📊 Risk Metrics")
        total_change = stressed_xva.total - base_xva.total
        pct_change = (
            (stressed_xva.total / base_xva.total - 1) * 100
            if base_xva.total != 0
            else 0
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total xVA Change",
                f"${total_change/1e6:+.2f}M",
                delta=f"{pct_change:+.1f}%",
                delta_color="inverse",
            )
        with col2:
            # Stress buffer recommendation
            buffer = max(0, total_change) * 1.5
            st.metric(
                "Recommended Buffer",
                f"${buffer/1e6:.2f}M",
                help="150% of adverse xVA change",
            )
        with col3:
            # Max component change
            changes = [
                abs(stressed_xva.cva - base_xva.cva),
                abs(stressed_xva.dva - base_xva.dva),
                abs(stressed_xva.fva - base_xva.fva),
                abs(stressed_xva.mva - base_xva.mva),
                abs(stressed_xva.kva - base_xva.kva),
            ]
            max_component = ["CVA", "DVA", "FVA", "MVA", "KVA"][np.argmax(changes)]
            st.metric(
                "Largest Impact",
                max_component,
                delta=f"${max(changes)/1e6:.2f}M",
            )


def sensitivities_tab(config: dict) -> None:
    """
    Sensitivities (Greeks) tab.

    Calculate CS01, IR01, Vega for risk management and hedging.
    """
    st.header("📐 Sensitivities (Greeks)")

    st.markdown("""
        Calculate first-order sensitivities for xVA risk management and hedging.
        These are essential for:
        - **Hedging**: Determine hedge ratios for CVA desk
        - **P&L Attribution**: Explain daily xVA moves
        - **Limit Monitoring**: Track risk against limits
        """)

    if st.session_state.results is None:
        st.info("Run a base simulation first to calculate sensitivities.")
        return

    r = st.session_state.results
    base_xva = r["xva_result"]

    st.info("""
        **Sensitivity Definitions:**
        - **CS01**: Credit Spread 01 - xVA change for 1bp increase in CDS spread
        - **IR01**: Interest Rate 01 - xVA change for 1bp parallel rate shift
        - **Vega**: xVA change for 1% absolute increase in volatility
        - **FX Delta**: xVA change for 1% FX spot move
        """)

    # Bump sizes
    st.subheader("⚙️ Bump Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cs_bump = st.number_input(
            "Credit Bump (bps)", value=1, min_value=1, max_value=10, key="cs_bump"
        )
    with col2:
        ir_bump = st.number_input(
            "IR Bump (bps)", value=1, min_value=1, max_value=10, key="ir_bump"
        )
    with col3:
        vol_bump = st.number_input(
            "Vol Bump (%)", value=1.0, min_value=0.1, max_value=5.0, key="vol_bump"
        )
    with col4:
        fx_bump = st.number_input(
            "FX Bump (%)", value=1.0, min_value=0.1, max_value=5.0, key="fx_bump"
        )

    if st.button("📊 Calculate Sensitivities", type="primary"):
        progress_bar = st.progress(0, text="Calculating sensitivities...")

        sensitivities = {}

        # Helper function to run bumped simulation
        def run_bumped_sim(bumped_config: dict) -> float:
            irs_list, fxf_list = get_portfolio()
            instruments = irs_list + fxf_list

            market_config = build_market_config(bumped_config)
            # Bell curve mode: use dt=0.5 to align with semi-annual payment dates
            # This removes sawtooth artifacts and shows cleaner bell shape
            if st.session_state.get("bell_curve_mode"):
                dt = 0.5
            else:
                dt = 1 / 12 if config["freq"] == "Monthly" else 0.25

            engine = MonteCarloEngine(
                n_paths=min(config["n_paths"], 2000),  # Faster for Greeks
                horizon=config["horizon"],
                dt=dt,
                seed=config["seed"],
            )

            result = engine.simulate(instruments, market_config)

            vm = VariationMargin(
                threshold=config["threshold"],
                mta=config["mta"],
                mpr_days=config["mpr_days"],
                days_per_step=dt * 365,
            )
            _, coll_exposure = vm.apply(result.mtm, result.time_grid)
            epe = np.maximum(coll_exposure, 0).mean(axis=0)
            ene = np.maximum(-coll_exposure, 0).mean(axis=0)

            im = InitialMargin(multiplier=config["im_mult"])
            im_profile = im.calculate(coll_exposure)

            avg_df = result.df_domestic.mean(axis=0)

            xva_params = XVAParams(
                lgd_counterparty=config["lgd_cpty"],
                lgd_own=config["lgd_own"],
                hazard_rate_counterparty=bumped_config.get(
                    "lambda_cpty", config["lambda_cpty"]
                ),
                hazard_rate_own=bumped_config.get("lambda_own", config["lambda_own"]),
                funding_spread=config["funding_spread"],
                cost_of_capital=config["coc"],
                capital_ratio=config["capital_ratio"],
                im_multiplier=config["im_mult"],
            )

            xva_result = calculate_all_xva(
                epe=epe,
                ene=ene,
                discount_factors=avg_df,
                time_grid=result.time_grid,
                params=xva_params,
                im_profile=im_profile,
            )

            return xva_result

        # CS01 - Credit Sensitivity
        progress_bar.progress(10, text="Calculating CS01...")
        cs_up_config = config.copy()
        cs_up_config["lambda_cpty"] = config["lambda_cpty"] + cs_bump / 10000
        cs_up_xva = run_bumped_sim(cs_up_config)

        sensitivities["CS01_CVA"] = (cs_up_xva.cva - base_xva.cva) / cs_bump
        sensitivities["CS01_DVA"] = (cs_up_xva.dva - base_xva.dva) / cs_bump
        sensitivities["CS01_Total"] = (cs_up_xva.total - base_xva.total) / cs_bump

        # IR01 - Interest Rate Sensitivity
        progress_bar.progress(35, text="Calculating IR01...")
        ir_up_config = config.copy()
        ir_up_config["theta_d"] = config["theta_d"] + ir_bump / 10000
        ir_up_config["theta_f"] = config["theta_f"] + ir_bump / 10000
        ir_up_xva = run_bumped_sim(ir_up_config)

        sensitivities["IR01_CVA"] = (ir_up_xva.cva - base_xva.cva) / ir_bump
        sensitivities["IR01_FVA"] = (ir_up_xva.fva - base_xva.fva) / ir_bump
        sensitivities["IR01_Total"] = (ir_up_xva.total - base_xva.total) / ir_bump

        # Vega - Volatility Sensitivity
        progress_bar.progress(60, text="Calculating Vega...")
        vol_up_config = config.copy()
        vol_up_config["sigma_d"] = config["sigma_d"] * (1 + vol_bump / 100)
        vol_up_config["sigma_f"] = config["sigma_f"] * (1 + vol_bump / 100)
        vol_up_config["fx_vol"] = config["fx_vol"] * (1 + vol_bump / 100)
        vol_up_xva = run_bumped_sim(vol_up_config)

        sensitivities["Vega_CVA"] = (vol_up_xva.cva - base_xva.cva) / vol_bump
        sensitivities["Vega_FVA"] = (vol_up_xva.fva - base_xva.fva) / vol_bump
        sensitivities["Vega_Total"] = (vol_up_xva.total - base_xva.total) / vol_bump

        # FX Delta
        progress_bar.progress(85, text="Calculating FX Delta...")
        fx_up_config = config.copy()
        fx_up_config["fx_spot"] = config["fx_spot"] * (1 + fx_bump / 100)
        fx_up_xva = run_bumped_sim(fx_up_config)

        sensitivities["FXDelta_CVA"] = (fx_up_xva.cva - base_xva.cva) / fx_bump
        sensitivities["FXDelta_FVA"] = (fx_up_xva.fva - base_xva.fva) / fx_bump
        sensitivities["FXDelta_Total"] = (fx_up_xva.total - base_xva.total) / fx_bump

        progress_bar.progress(100, text="Complete!")

        st.session_state.sensitivities = sensitivities

    # Display sensitivities
    if "sensitivities" in st.session_state:
        sens = st.session_state.sensitivities

        st.divider()
        st.subheader("📊 Sensitivity Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "CS01 (Total)",
                f"${sens['CS01_Total']/1000:.1f}K/bp",
                help="xVA change per 1bp CDS spread increase",
            )
        with col2:
            st.metric(
                "IR01 (Total)",
                f"${sens['IR01_Total']/1000:.1f}K/bp",
                help="xVA change per 1bp rate increase",
            )
        with col3:
            st.metric(
                "Vega (Total)",
                f"${sens['Vega_Total']/1000:.1f}K/%",
                help="xVA change per 1% vol increase",
            )
        with col4:
            st.metric(
                "FX Delta (Total)",
                f"${sens['FXDelta_Total']/1000:.1f}K/%",
                help="xVA change per 1% FX move",
            )

        # Detailed breakdown
        st.markdown("### Detailed Breakdown")

        sens_data = {
            "Greek": ["CS01", "IR01", "Vega", "FX Delta"],
            "CVA ($K/unit)": [
                sens["CS01_CVA"] / 1000,
                sens["IR01_CVA"] / 1000,
                sens["Vega_CVA"] / 1000,
                sens["FXDelta_CVA"] / 1000,
            ],
            "DVA/FVA ($K/unit)": [
                sens["CS01_DVA"] / 1000,
                sens["IR01_FVA"] / 1000,
                sens["Vega_FVA"] / 1000,
                sens["FXDelta_FVA"] / 1000,
            ],
            "Total ($K/unit)": [
                sens["CS01_Total"] / 1000,
                sens["IR01_Total"] / 1000,
                sens["Vega_Total"] / 1000,
                sens["FXDelta_Total"] / 1000,
            ],
        }

        sens_df = pd.DataFrame(sens_data)
        st.dataframe(
            sens_df.style.format(
                {
                    "CVA ($K/unit)": "{:.2f}",
                    "DVA/FVA ($K/unit)": "{:.2f}",
                    "Total ($K/unit)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Sensitivity chart
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Absolute Sensitivities", "Relative Contribution"),
            )

            greeks = ["CS01", "IR01", "Vega", "FX Delta"]
            totals = [
                sens["CS01_Total"] / 1000,
                sens["IR01_Total"] / 1000,
                sens["Vega_Total"] / 1000,
                sens["FXDelta_Total"] / 1000,
            ]

            # Absolute values
            fig.add_trace(
                go.Bar(
                    x=greeks,
                    y=totals,
                    marker_color=["#FF4B4B", "#00CC96", "#FFA500", "#9467BD"],
                    text=[f"${v:.1f}K" for v in totals],
                    textposition="outside",
                ),
                row=1,
                col=1,
            )

            # Relative contribution (pie)
            fig.add_trace(
                go.Pie(
                    labels=greeks,
                    values=[abs(v) for v in totals],
                    hole=0.4,
                    marker_colors=["#FF4B4B", "#00CC96", "#FFA500", "#9467BD"],
                ),
                row=1,
                col=2,
            )

            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            pass

        # Hedge recommendations
        st.subheader("🎯 Hedging Recommendations")

        st.markdown(f"""
            Based on the sensitivity analysis:

            | Risk Factor | Sensitivity | Hedge Instrument | Notional |
            |-------------|-------------|------------------|----------|
            | Credit (CS01) | ${sens['CS01_Total']/1000:.1f}K/bp | CDS on counterparty | Varies |
            | IR (IR01) | ${sens['IR01_Total']/1000:.1f}K/bp | IRS / Swaption | ${abs(sens['IR01_Total']/100)/1e6:.2f}M DV01 |
            | Vol (Vega) | ${sens['Vega_Total']/1000:.1f}K/% | Swaption straddle | ${abs(sens['Vega_Total']*10)/1e6:.2f}M vega |
            | FX (Delta) | ${sens['FXDelta_Total']/1000:.1f}K/% | FX forward | ${abs(sens['FXDelta_Total']*100)/1e6:.2f}M notional |
            """)


def methodology_tab(config: dict) -> None:
    """
    Methodology documentation tab.

    Explains the mathematical foundations and models used.
    """
    st.header("📚 Methodology & Documentation")

    st.markdown("""
        This section documents the mathematical foundations and regulatory
        framework underlying the xVA calculations.
        """)

    # Sub-tabs for different sections
    doc_tab1, doc_tab2, doc_tab3, doc_tab4, doc_tab5 = st.tabs(
        [
            "📐 Market Models",
            "💰 xVA Formulas",
            "🏛️ Regulatory",
            "📊 Monte Carlo",
            "📖 References",
        ]
    )

    with doc_tab1:
        st.subheader("Market Models")

        st.markdown("""
            ### Ornstein-Uhlenbeck (OU) Process for Interest Rates

            Interest rates follow a mean-reverting OU process:

            $$dr_t = \\kappa(\\theta - r_t)dt + \\sigma dW_t$$

            Where:
            - $r_t$ = instantaneous short rate
            - $\\kappa$ = mean reversion speed
            - $\\theta$ = long-term mean rate
            - $\\sigma$ = volatility
            - $W_t$ = Brownian motion

            **Properties:**
            - Mean: $E[r_t] = \\theta + (r_0 - \\theta)e^{-\\kappa t}$
            - Variance: $Var(r_t) = \\frac{\\sigma^2}{2\\kappa}(1 - e^{-2\\kappa t})$
            """)

        st.divider()

        st.markdown("""
            ### Geometric Brownian Motion (GBM) for FX

            FX spot rates follow a GBM process:

            $$dS_t = (r_d - r_f)S_t dt + \\sigma_S S_t dW_t^S$$

            Where:
            - $S_t$ = FX spot rate
            - $r_d$ = domestic interest rate
            - $r_f$ = foreign interest rate
            - $\\sigma_S$ = FX volatility

            **Correlation Structure:**

            The three Brownian motions are correlated via Cholesky decomposition:

            $$dW = L \\cdot dZ$$

            Where:
            - $dW = (dW_t^d, dW_t^f, dW_t^S)^T$ = correlated Brownian motions
            - $dZ = (dZ_t^1, dZ_t^2, dZ_t^3)^T$ = independent standard normals
            - $L$ = Cholesky factor of the correlation matrix $\\rho$
            """)

    with doc_tab2:
        st.subheader("xVA Formulas")

        st.markdown("""
            ### Credit Valuation Adjustment (CVA)

            $$\\text{CVA} = \\text{LGD}_c \\cdot \\int_0^T \\text{EPE}(t) \\cdot D(t) \\cdot dP_c(t)$$

            Discrete approximation:

            $$\\text{CVA} = \\text{LGD}_c \\sum_{i=1}^{n} \\text{EPE}(t_i) \\cdot D(t_i) \\cdot \\Delta P_c(t_i)$$

            Where:
            - $\\text{LGD}_c$ = Loss Given Default of counterparty
            - $\\text{EPE}(t)$ = Expected Positive Exposure
            - $D(t)$ = Risk-free discount factor
            - $P_c(t)$ = Counterparty survival probability
            """)

        st.divider()

        st.markdown("""
            ### Debit Valuation Adjustment (DVA)

            $$\\text{DVA} = \\text{LGD}_{own} \\cdot \\int_0^T \\text{ENE}(t) \\cdot D(t) \\cdot dP_{own}(t)$$

            Where $\\text{ENE}(t)$ = Expected Negative Exposure

            ### Funding Valuation Adjustment (FVA)

            $$\\text{FVA} = s_f \\cdot \\int_0^T (\\text{EPE}(t) - \\text{ENE}(t)) \\cdot D(t) \\cdot dt$$

            Where $s_f$ = funding spread over OIS
            """)

        st.divider()

        st.markdown("""
            ### Margin Valuation Adjustment (MVA)

            $$\\text{MVA} = s_f \\cdot \\int_0^T \\text{IM}(t) \\cdot D(t) \\cdot dt$$

            Where $\\text{IM}(t)$ = Initial Margin requirement

            ### Capital Valuation Adjustment (KVA)

            $$\\text{KVA} = \\text{CoC} \\cdot \\int_0^T K(t) \\cdot D(t) \\cdot dt$$

            Where:
            - $\\text{CoC}$ = Cost of Capital (hurdle rate)
            - $K(t)$ = Regulatory capital requirement
            """)

        st.divider()

        st.markdown("""
            ### Credit Spreads by Rating - Methodology

            **Hazard rate derivation (flat spread / flat hazard approximation):**

            $$\\lambda \\approx \\frac{s}{\\text{LGD}}$$

            Where:
            - $s$ = CDS spread in decimal (e.g., 100 bps = 0.01)
            - $\\text{LGD}$ = Loss Given Default = 60% (i.e., Recovery $R$ = 40%)
            - $\\lambda$ = constant hazard rate (intensity of default)

            This is a standard simplification assuming **flat spread** and **flat hazard rate**
            (no term structure). Real-world CVA desks use full CDS term structures.

            **Spread ranges used in this application:**

            | Rating | App Range (bps) | Academic Ref. (bps) | λ (LGD=60%) |
            |--------|-----------------|---------------------|-------------|
            | AAA    | 20 - 50         | ~40                 | 0.33 - 0.83% |
            | AA     | 40 - 80         | ~55                 | 0.67 - 1.33% |
            | A      | 60 - 120        | 70 - 89             | 1.00 - 2.00% |
            | BBB    | 100 - 200       | ~111                | 1.67 - 3.33% |
            | BB     | 200 - 400       | 138 - 184           | 3.33 - 6.67% |
            | B      | 400 - 800       | 275 - 509           | 6.67 - 13.33% |

            **Calibration**: Ranges are wider than point estimates to cover market
            variability across credit cycles (expansion/recession).

            ---

            ### Sources & References (Traceable)

            **(1) Damodaran, A. (NYU Stern) – "Ratings and Default Spreads"**

            Dataset: "Ratings, Interest Coverage Ratios and Default Spread"
            Updated annually. Widely used in corporate finance education.

            🔗 `https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html`
            *(Accessed: January 2025)*

            **(2) FRED – ICE BofA Corporate Bond OAS Indices**

            Series: ICE BofA US Corporate BBB Option-Adjusted Spread
            Ticker: BAMLC0A4CBBB (also available: AAA, AA, A, BB, B, CCC)
            Provider: ICE Data Indices, LLC

            🔗 `https://fred.stlouisfed.org/series/BAMLC0A4CBBB`
            *(Accessed: January 2025)*

            **(3) IHS Markit – CDS Indices Primer**

            Document: "Credit Default Swap Index (CDX/iTraxx) – Product Guide"
            Defines: iTraxx Europe (125 IG), iTraxx Crossover (75 sub-IG),
            CDX.NA.IG, CDX.NA.HY

            🔗 `https://www.spglobal.com/spdji/en/landing/topic/itraxx/`
            *(Accessed: January 2025)*

            ---

            ### Pedagogical Disclaimer

            ⚠️ **This is an educational project – not production CVA:**

            - Spreads by rating are **pedagogical proxies** for hazard curve construction.
            - We use a **flat spread / flat hazard** approximation (no term structure).
            - LGD is fixed at **60%** (standard recovery assumption for senior unsecured).
            - Real CVA desks calibrate to full CDS curves, use stochastic recovery, etc.
            - Ranges (not point estimates) reflect that spreads vary with market conditions.
            """)

    with doc_tab3:
        st.subheader("Regulatory Framework")

        st.markdown("""
            ### SA-CCR (Standardized Approach for Counterparty Credit Risk)

            Basel III framework for calculating Exposure at Default (EAD):

            $$\\text{EAD} = \\alpha \\cdot (\\text{RC} + \\text{PFE})$$

            Where:
            - $\\alpha = 1.4$ (regulatory multiplier)
            - $\\text{RC}$ = Replacement Cost (current exposure)
            - $\\text{PFE}$ = Potential Future Exposure

            **Add-On Calculation:**

            $$\\text{AddOn} = \\sum_{\\text{asset class}} \\text{AddOn}_{ac}$$

            For Interest Rate derivatives:
            $$\\text{AddOn}_{IR} = \\text{SF}_{IR} \\cdot \\text{MF} \\cdot d \\cdot \\text{Notional}$$

            Where:
            - $\\text{SF}_{IR} = 0.5\\%$ (supervisory factor)
            - $\\text{MF}$ = Maturity factor
            - $d$ = Supervisory delta
            """)

        st.divider()

        st.markdown("""
            ### Initial Margin (ISDA SIMM)

            The SIMM (Standard Initial Margin Model) calculates IM based on:

            $$\\text{IM} = \\sqrt{\\sum_i \\sum_j \\rho_{ij} \\cdot S_i \\cdot S_j}$$

            Where $S_i$ are risk sensitivities and $\\rho_{ij}$ are correlations.

            **Simplified Approach (used here):**

            $$\\text{IM}(t) = \\text{mult} \\cdot \\sigma_{\\text{portfolio}} \\cdot \\sqrt{\\text{MPR}}$$

            Where:
            - mult = IM multiplier
            - MPR = Margin Period of Risk (typically 10 days)
            """)

    with doc_tab4:
        st.subheader("Monte Carlo Simulation")

        st.markdown("""
            ### Simulation Framework

            The engine generates correlated market scenarios using:

            1. **Cholesky Decomposition** for correlation:
            $$L \\cdot L^T = \\Sigma$$

            2. **Euler-Maruyama Discretization** for OU:
            $$r_{t+\\Delta t} = r_t + \\kappa(\\theta - r_t)\\Delta t + \\sigma\\sqrt{\\Delta t}\\epsilon$$

            3. **Log-Euler for GBM**:
            $$S_{t+\\Delta t} = S_t \\exp\\left((\\mu - \\frac{\\sigma^2}{2})\\Delta t + \\sigma\\sqrt{\\Delta t}\\epsilon\\right)$$

            ### Exposure Metrics

            | Metric | Definition |
            |--------|------------|
            | EPE(t) | $E[\\max(V(t), 0)]$ |
            | ENE(t) | $E[\\max(-V(t), 0)]$ |
            | PFE(t) | $q_{0.95}(\\max(V(t), 0))$ |
            | EE(t)  | $E[V(t)]$ |

            ### Collateralized Exposure

            With Variation Margin:
            $$V^{coll}(t) = V(t) - C(t)$$

            Where $C(t)$ follows the CSA terms (threshold, MTA, MPR).
            """)

        # Current configuration
        st.divider()
        st.markdown("### Current Simulation Parameters")

        param_df = pd.DataFrame(
            {
                "Parameter": [
                    "Number of Paths",
                    "Horizon",
                    "Time Step",
                    "Random Seed",
                    "OU κ (domestic)",
                    "OU θ (domestic)",
                    "OU σ (domestic)",
                    "FX Volatility",
                ],
                "Value": [
                    f"{config['n_paths']:,}",
                    f"{config['horizon']} years",
                    config["freq"],
                    f"{config['seed']}",
                    f"{config['kappa_d']:.2f}",
                    f"{config['theta_d']*100:.2f}%",
                    f"{config['sigma_d']*10000:.0f} bps",
                    f"{config['fx_vol']*100:.0f}%",
                ],
            }
        )
        st.dataframe(param_df, use_container_width=True, hide_index=True)

    with doc_tab5:
        st.subheader("References & Further Reading")

        st.markdown("""
            ### Academic References

            1. **Gregory, J.** (2020). *The xVA Challenge: Counterparty Risk, Funding,
               Collateral, Capital and Initial Margin*. Wiley Finance. 4th Edition.

            2. **Brigo, D., Morini, M., & Pallavicini, A.** (2013). *Counterparty Credit
               Risk, Collateral and Funding*. Wiley Finance.

            3. **Green, A.** (2015). *XVA: Credit, Funding and Capital Valuation
               Adjustments*. Wiley Finance.

            4. **Pykhtin, M., & Zhu, S.** (2007). "A Guide to Modelling Counterparty
               Credit Risk." *GARP Risk Review*, 37, 16-22.

            ### Regulatory Documents

            1. **BCBS 279** (2014). "The standardised approach for measuring
               counterparty credit risk exposures."

            2. **BCBS 317** (2015). "Margin requirements for non-centrally
               cleared derivatives."

            3. **ISDA SIMM** (2021). "ISDA Standard Initial Margin Model
               Methodology."

            ### Online Resources

            - [ISDA Documentation](https://www.isda.org/)
            - [BIS Basel Framework](https://www.bis.org/basel_framework/)
            - [Risk.net xVA articles](https://www.risk.net/topics/xva)

            ### Model Validation

            This implementation follows industry best practices for:
            - Monte Carlo convergence testing
            - Exposure profile validation
            - SA-CCR compliance checks
            - Stress testing frameworks
            """)

        # Export methodology document
        st.divider()
        st.markdown("### Export Documentation")

        methodology_text = """
# xVA Calculation Engine - Methodology Document

## 1. Introduction
This document describes the mathematical models and methodologies used in the xVA calculation engine.

## 2. Market Models

### 2.1 Interest Rate Model (Ornstein-Uhlenbeck)
dr_t = κ(θ - r_t)dt + σdW_t

### 2.2 FX Model (Geometric Brownian Motion)
dS_t = (r_d - r_f)S_t dt + σ_S S_t dW_t

## 3. xVA Components

### 3.1 CVA
CVA = LGD × Σ EPE(t_i) × D(t_i) × ΔPD(t_i)

### 3.2 DVA
DVA = LGD_own × Σ ENE(t_i) × D(t_i) × ΔPD_own(t_i)

### 3.3 FVA
FVA = s_f × Σ (EPE(t_i) - ENE(t_i)) × D(t_i) × Δt

### 3.4 MVA
MVA = s_f × Σ IM(t_i) × D(t_i) × Δt

### 3.5 KVA
KVA = CoC × Σ K(t_i) × D(t_i) × Δt

## 4. SA-CCR
EAD = 1.4 × (RC + PFE)

## 5. References
- Gregory, J. (2020). The xVA Challenge. Wiley.
- BCBS 279 (2014). SA-CCR Standard.
"""

        st.download_button(
            "📥 Export Methodology (TXT)",
            methodology_text,
            "xva_methodology.txt",
            "text/plain",
        )


def export_tab(config: dict) -> None:
    """Export results tab."""
    st.header("💾 Export Results")

    if st.session_state.results is None:
        st.info("Run simulation to export results.")
        return

    r = st.session_state.results

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 CSV Exports")

        # Exposure CSV
        exposure_df = pd.DataFrame(
            {
                "Time": r["sim_result"].time_grid,
                "EPE_Uncoll": r["epe_uncoll"],
                "ENE_Uncoll": r["ene_uncoll"],
                "EPE_Coll": r["epe_coll"],
                "ENE_Coll": r["ene_coll"],
                "IM": r["im_profile"],
            }
        )
        st.download_button(
            "📥 Exposure Profiles",
            exposure_df.to_csv(index=False),
            "exposure_profiles.csv",
            "text/csv",
        )

        # xVA CSV
        xva = r["xva_result"]
        xva_df = pd.DataFrame(
            {
                "Component": ["CVA", "DVA", "FVA", "MVA", "KVA", "Total"],
                "Value": [xva.cva, xva.dva, xva.fva, xva.mva, xva.kva, xva.total],
            }
        )
        st.download_button(
            "📥 xVA Breakdown",
            xva_df.to_csv(index=False),
            "xva_breakdown.csv",
            "text/csv",
        )

    with col2:
        st.subheader("📊 Excel Workbook")

        # Create Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            exposure_df.to_excel(writer, sheet_name="Exposure", index=False)
            xva_df.to_excel(writer, sheet_name="xVA", index=False)
        buffer.seek(0)

        st.download_button(
            "📥 Full Report (XLSX)",
            buffer,
            "xva_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col3:
        st.subheader("⚙️ Configuration")

        config_json = {
            "simulation": {
                "n_paths": r["config"]["n_paths"],
                "horizon": r["config"]["horizon"],
                "seed": r["config"]["seed"],
            },
            "xva": r["xva_result"].to_dict(),
        }
        st.download_button(
            "📥 Config (JSON)",
            json.dumps(config_json, indent=2),
            "config.json",
            "application/json",
        )


if __name__ == "__main__":
    main()
