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
    st.markdown(
        """
        *Production-grade counterparty credit risk and valuation adjustments*

        This application calculates **CVA, DVA, FVA, MVA, KVA** for a portfolio
        of interest rate swaps and FX forwards using Monte Carlo simulation.
        """
    )

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0

    # Sidebar configuration
    config = build_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📋 Portfolio",
            "📊 Exposure",
            "💸 xVA",
            "🏛️ SA-CCR",
            "📈 Calibration",
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
        export_tab(config)


def build_sidebar() -> dict:
    """Build sidebar with all configuration options."""
    config = {}

    with st.sidebar:
        st.title("⚙️ Configuration")

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
            config["kappa_d"] = st.slider(
                "Mean Reversion κ", 0.01, 0.50, 0.10, key="kappa_d"
            )
            config["theta_d"] = (
                st.slider("Long-term θ (%)", 0.0, 5.0, 2.0, key="theta_d") / 100
            )
            config["sigma_d"] = (
                st.slider("Volatility σ (bps)", 10, 200, 100, key="sigma_d") / 10000
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
            config["fx_spot"] = st.number_input(
                "Initial Spot", value=1.10, format="%.4f"
            )
            config["fx_vol"] = st.slider("Volatility (%)", 5, 30, 12) / 100

        # Correlations
        with st.expander("🔗 Correlations"):
            config["corr_df"] = st.slider(
                "Domestic-Foreign", -1.0, 1.0, 0.7, key="corr_df"
            )
            config["corr_dx"] = st.slider("Domestic-FX", -1.0, 1.0, -0.3, key="corr_dx")
            config["corr_fx"] = st.slider("Foreign-FX", -1.0, 1.0, 0.4, key="corr_fx")

        # Collateral
        with st.expander("🏦 Collateral"):
            config["threshold"] = (
                st.number_input("Threshold ($M)", value=1.0, step=0.1) * 1e6
            )
            config["mta"] = st.number_input("MTA ($K)", value=100.0, step=10.0) * 1e3
            config["mpr_days"] = st.slider("MPR (days)", 0, 20, 10)
            config["im_mult"] = st.slider("IM Multiplier", 1.0, 3.0, 1.5)

        # Credit & Funding
        with st.expander("💰 Credit & Funding"):
            config["lgd_cpty"] = st.slider("LGD Counterparty (%)", 0, 100, 60) / 100
            config["lgd_own"] = st.slider("LGD Own (%)", 0, 100, 60) / 100
            config["lambda_cpty"] = (
                st.slider("λ Counterparty (bps)", 0, 500, 120) / 10000
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


def portfolio_tab(config: dict) -> None:
    """Portfolio editing tab."""
    st.header("📋 Trade Portfolio")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Interest Rate Swaps")
        st.session_state.irs_trades = st.data_editor(
            st.session_state.get(
                "irs_trades",
                pd.DataFrame(
                    {
                        "Notional ($M)": [10.0, 15.0, 8.0],
                        "Fixed Rate (%)": [2.5, 2.0, 3.0],
                        "Maturity (Y)": [5.0, 3.0, 7.0],
                        "Pay Fixed": [True, False, True],
                    }
                ),
            ),
            num_rows="dynamic",
            use_container_width=True,
        )

    with col2:
        st.subheader("FX Forwards")
        st.session_state.fxf_trades = st.data_editor(
            st.session_state.get(
                "fxf_trades",
                pd.DataFrame(
                    {
                        "Notional (M EUR)": [5.0, 3.0],
                        "Strike": [1.12, 1.08],
                        "Maturity (Y)": [1.0, 2.0],
                        "Buy Foreign": [True, False],
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
        reduction = (1 - r["epe_coll"].max() / r["epe_uncoll"].max()) * 100
        st.metric("Collateral Benefit", f"{reduction:.1f}%")
    with col4:
        st.metric("Avg IM", f"${r['im_profile'].mean()/1e6:.2f}M")


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

    st.info(
        """
        **SA-CCR**: Standardized Approach for Counterparty Credit Risk (Basel III)

        EAD = α × (RC + PFE), where α = 1.4
        """
    )

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

    st.markdown(
        """
        Calibrate model parameters from historical market data. Upload CSV files
        from Bloomberg, Reuters, or any financial data provider to derive:
        - **Volatilities** for IR and FX models
        - **Hazard rates** from CDS spreads for CVA/DVA
        - **Correlations** between risk factors
        """
    )

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
    calib_tab1, calib_tab2, calib_tab3, calib_tab4 = st.tabs(
        ["📊 Volatilities", "💳 CDS / Credit", "🔗 Correlations", "📋 Summary"]
    )

    with calib_tab1:
        _calibration_volatility_section(config)

    with calib_tab2:
        _calibration_cds_section(config)

    with calib_tab3:
        _calibration_correlation_section(config)

    with calib_tab4:
        _calibration_summary_section(config)


def _calibration_volatility_section(config: dict) -> None:
    """Volatility calibration for IR and FX."""
    st.subheader("Volatility Calibration")

    st.info(
        """
        **Methodology:**
        - **IR Volatility**: Calculated from absolute rate changes (OU model assumption)
        - **FX Volatility**: Calculated from log-returns (GBM model assumption)

        Both are annualized using √252 (trading days).
        """
    )

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
                    st.caption(f"Detected format: {rate_format} | {len(ir_df)} observations")

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
                        rows=2, cols=1,
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
                        row=1, col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=ir_df["date"],
                            y=ir_df["rolling_vol"] * np.sqrt(252) * 10000,
                            name="Rolling Vol (bps)",
                            line={"color": "#FF4B4B"},
                        ),
                        row=2, col=1,
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
                st.session_state.calibrated_params["ir_data"] = ir_df[["date", "rate_decimal"]].copy()

                if st.button("✅ Apply IR Parameters", key="apply_ir", type="primary"):
                    st.success(
                        f"Applied: σ={annualized_vol*10000:.0f}bps, "
                        f"κ={kappa_estimate:.2f}, θ={theta_estimate*100:.2f}%"
                    )

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
                        rows=2, cols=1,
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
                        row=1, col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=fx_df["date"],
                            y=fx_df["rolling_vol"] * np.sqrt(252) * 100,
                            name="Rolling Vol (%)",
                            line={"color": "#FFA500"},
                        ),
                        row=2, col=1,
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
                st.session_state.calibrated_params["fx_data"] = fx_df[["date", "rate", "log_return"]].copy()

                if st.button("✅ Apply FX Parameters", key="apply_fx", type="primary"):
                    st.success(
                        f"Applied: σ={annualized_vol*100:.1f}%, Spot={current_spot:.4f}"
                    )

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

    st.info(
        """
        **CDS → Hazard Rate Conversion:**

        The hazard rate (λ) represents the instantaneous probability of default.
        It can be derived from CDS spreads using:

        $$\\lambda = \\frac{\\text{CDS Spread}}{1 - \\text{LGD}}$$

        **Example:** CDS = 120 bps, LGD = 60% → λ = 0.0120 / 0.40 = 3.0%
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💳 CDS Spread Data")
        st.markdown(
            """
            Upload CSV with CDS spreads. Supported formats:
            - **Single tenor:** `date`, `spread_bps`
            - **Multi-tenor:** `date`, `1Y`, `3Y`, `5Y`, `10Y` (spreads in bps)
            """
        )

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
                tenor_cols = [c for c in cds_df.columns if c not in ["date", "spread_bps"]]
                is_multi_tenor = len(tenor_cols) > 0 and any(
                    t in str(tenor_cols) for t in ["1Y", "3Y", "5Y", "10Y", "1y", "3y", "5y", "10y"]
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
                    tenor_cols = [c for c in ["1Y", "3Y", "5Y", "10Y"] if c in cds_df.columns]

                    # Latest spreads and hazard rates
                    latest = cds_df.iloc[-1]
                    term_structure = []
                    for tenor in tenor_cols:
                        spread_bps = latest[tenor]
                        hazard = (spread_bps / 10000) / recovery
                        term_structure.append({
                            "Tenor": tenor,
                            "CDS Spread (bps)": spread_bps,
                            "Hazard Rate (λ)": f"{hazard*100:.2f}%",
                            "λ (decimal)": hazard,
                        })

                    term_df = pd.DataFrame(term_structure)
                    st.dataframe(term_df[["Tenor", "CDS Spread (bps)", "Hazard Rate (λ)"]],
                                use_container_width=True)

                    # Select which tenor to use
                    selected_tenor = st.selectbox(
                        "Select tenor for hazard rate",
                        tenor_cols,
                        index=min(2, len(tenor_cols)-1),  # Default to 5Y if available
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
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
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
                        spread_cols = [c for c in cds_df.columns if "spread" in c.lower() or "cds" in c.lower()]
                        if spread_cols:
                            cds_df["spread_bps"] = cds_df[spread_cols[0]]
                        else:
                            # Assume second column is spread
                            cds_df["spread_bps"] = cds_df.iloc[:, 1]

                    # Calculate hazard rates
                    cds_df["hazard_rate"] = (cds_df["spread_bps"] / 10000) / recovery

                    final_spread = cds_df["spread_bps"].iloc[-1]
                    final_hazard = cds_df["hazard_rate"].iloc[-1]
                    avg_spread = cds_df["spread_bps"].mean()

                    # Plot
                    try:
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots

                        fig = make_subplots(
                            rows=2, cols=1,
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
                            row=1, col=1,
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=cds_df["date"],
                                y=cds_df["hazard_rate"] * 100,
                                name="Hazard Rate",
                                line={"color": "#4ECDC4"},
                            ),
                            row=2, col=1,
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

                if st.button("✅ Apply Hazard Rate to CVA/DVA", key="apply_cds", type="primary"):
                    st.success(
                        f"Applied: λ = {final_hazard*100:.2f}% "
                        f"(from CDS = {final_spread:.0f} bps, LGD = {lgd*100:.0f}%)"
                    )

            except Exception as e:
                st.error(f"Error processing CDS file: {e}")

    with col2:
        st.markdown("### ℹ️ CDS Basics")
        st.markdown(
            """
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
            """
        )

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
        sample_cds_multi = pd.DataFrame({
            "date": dates,
            "1Y": 80 + np.cumsum(np.random.randn(252) * 1.5),
            "3Y": 100 + np.cumsum(np.random.randn(252) * 1.8),
            "5Y": 120 + np.cumsum(np.random.randn(252) * 2.0),
            "10Y": 140 + np.cumsum(np.random.randn(252) * 2.2),
        })
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

    st.info(
        """
        **Automatic Correlation Calculation:**

        When you upload both IR and FX data in the Volatilities tab, the correlation
        between rate changes (ΔIR) and FX log-returns is computed automatically.

        $$\\rho_{IR,FX} = \\text{Corr}(\\Delta r_t, \\log(S_t/S_{t-1}))$$
        """
    )

    # Check if both IR and FX data are available
    ir_data = st.session_state.calibrated_params.get("ir_data")
    fx_data = st.session_state.calibrated_params.get("fx_data")

    if ir_data is None or fx_data is None:
        st.warning(
            """
            **Data Required:**

            Please upload both IR and FX historical data in the **Volatilities** tab
            to calculate correlations automatically.

            The correlation will be computed from overlapping dates.
            """
        )

        # Manual correlation input as fallback
        st.divider()
        st.markdown("### Manual Correlation Input")
        st.markdown("Alternatively, enter correlations manually if you have them from another source:")

        col1, col2, col3 = st.columns(3)
        with col1:
            manual_corr_ir_fx = st.slider(
                "Domestic Rate ↔ FX",
                -1.0, 1.0,
                config.get("corr_dx", -0.30),
                key="manual_corr_dx",
            )
        with col2:
            manual_corr_df = st.slider(
                "Domestic ↔ Foreign Rate",
                -1.0, 1.0,
                config.get("corr_df", 0.70),
                key="manual_corr_df",
            )
        with col3:
            manual_corr_fx_f = st.slider(
                "Foreign Rate ↔ FX",
                -1.0, 1.0,
                config.get("corr_fx", 0.40),
                key="manual_corr_fx",
            )

        if st.button("✅ Apply Manual Correlations", key="apply_manual_corr"):
            st.session_state.calibrated_params["correlation_ir_fx"] = manual_corr_ir_fx
            st.success("Manual correlations applied.")

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
        merged_clean["ir_change"]
        .rolling(window)
        .corr(merged_clean["log_return"])
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
            rows=2, cols=1,
            subplot_titles=(
                "Normalized IR Changes vs FX Returns",
                f"Rolling {window}-Day Correlation"
            ),
            vertical_spacing=0.15,
        )

        # Scatter plot of returns
        fig.add_trace(
            go.Scatter(
                x=merged_clean["ir_change"] / merged_clean["ir_change"].std(),
                y=merged_clean["log_return"] / merged_clean["log_return"].std(),
                mode="markers",
                marker=dict(
                    size=5,
                    color=np.arange(len(merged_clean)),
                    colorscale="Viridis",
                    opacity=0.6,
                ),
                name="Returns",
            ),
            row=1, col=1,
        )

        # Rolling correlation
        fig.add_trace(
            go.Scatter(
                x=merged_clean["date"],
                y=merged_clean["rolling_corr"],
                name="Rolling Corr",
                line={"color": "#9467BD", "width": 2},
            ),
            row=2, col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Add overall correlation line
        fig.add_hline(
            y=overall_corr,
            line_dash="dot",
            line_color="#FF4B4B",
            annotation_text=f"Overall: {overall_corr:.2f}",
            row=2, col=1,
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
        corr_matrix.style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
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
        st.success(f"Applied: ρ(IR, FX) = {overall_corr:.3f}")


def _calibration_summary_section(config: dict) -> None:
    """Summary of all calibrated parameters."""
    st.subheader("Calibration Summary")

    params = st.session_state.calibrated_params

    # Check what's been calibrated
    has_ir = params.get("ir_vol") is not None
    has_fx = params.get("fx_vol") is not None
    has_cds = params.get("hazard_rate") is not None
    has_corr = params.get("correlation_ir_fx") is not None

    calibrated_count = sum([has_ir, has_fx, has_cds, has_corr])

    if calibrated_count == 0:
        st.info(
            """
            **No parameters calibrated yet.**

            Use the other tabs to calibrate:
            - 📊 **Volatilities**: Upload IR and FX data
            - 💳 **CDS / Credit**: Upload CDS spreads
            - 🔗 **Correlations**: Computed automatically from IR/FX data
            """
        )
        return

    # Progress indicator
    st.progress(calibrated_count / 4, text=f"{calibrated_count}/4 parameter sets calibrated")

    # Summary table
    st.markdown("### 📋 Calibrated Parameters")

    summary_data = []

    # IR Parameters
    if has_ir:
        summary_data.extend([
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
        ])
    else:
        summary_data.append({
            "Category": "Interest Rate",
            "Parameter": "All parameters",
            "Calibrated": "—",
            "Model Default": "Using defaults",
            "Status": "⏳",
        })

    # FX Parameters
    if has_fx:
        summary_data.extend([
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
        ])
    else:
        summary_data.append({
            "Category": "FX",
            "Parameter": "All parameters",
            "Calibrated": "—",
            "Model Default": "Using defaults",
            "Status": "⏳",
        })

    # Credit Parameters
    if has_cds:
        summary_data.extend([
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
        ])
    else:
        summary_data.append({
            "Category": "Credit",
            "Parameter": "Hazard Rate",
            "Calibrated": "—",
            "Model Default": f"{config.get('lambda_cpty', 0.012)*100:.2f}%",
            "Status": "⏳",
        })

    # Correlation Parameters
    if has_corr:
        summary_data.append({
            "Category": "Correlation",
            "Parameter": "ρ(IR, FX)",
            "Calibrated": f"{params['correlation_ir_fx']:.3f}",
            "Model Default": f"{config.get('corr_dx', -0.30):.2f}",
            "Status": "✅",
        })
    else:
        summary_data.append({
            "Category": "Correlation",
            "Parameter": "ρ(IR, FX)",
            "Calibrated": "—",
            "Model Default": f"{config.get('corr_dx', -0.30):.2f}",
            "Status": "⏳",
        })

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
