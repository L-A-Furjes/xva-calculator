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
    CVACalculator,
    DVACalculator,
    FVACalculator,
    FXForward,
    IRSwap,
    KVACalculator,
    MonteCarloEngine,
    MVACalculator,
    SACCRCalculator,
)
from xva_core.collateral import InitialMargin, VariationMargin
from xva_core.config.loader import create_default_market_config
from xva_core.config.models import (
    CollateralConfig,
    CorrelationConfig,
    CreditConfig,
    FXModelConfig,
    MarketConfig,
    OUModelConfig,
    SimulationConfig,
)
from xva_core.exposure.metrics import ExposureMetrics
from xva_core.xva.result import XVAParams, XVAResult, calculate_all_xva

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
    st.markdown('<p class="main-header">📊 xVA Calculation Engine</p>', unsafe_allow_html=True)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Portfolio", "📊 Exposure", "💸 xVA", "🏛️ SA-CCR", "💾 Export"]
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
            config["freq"] = st.selectbox("Time Step", ["Quarterly", "Monthly"], index=0)
            config["seed"] = st.number_input("Random Seed", value=42, step=1)

        # Market models
        with st.expander("📈 Market Models"):
            st.subheader("Domestic Rates (OU)")
            config["kappa_d"] = st.slider("Mean Reversion κ", 0.01, 0.50, 0.10, key="kappa_d")
            config["theta_d"] = st.slider("Long-term θ (%)", 0.0, 5.0, 2.0, key="theta_d") / 100
            config["sigma_d"] = st.slider("Volatility σ (bps)", 10, 200, 100, key="sigma_d") / 10000

            st.subheader("Foreign Rates (OU)")
            config["kappa_f"] = st.slider("Mean Reversion κ", 0.01, 0.50, 0.08, key="kappa_f")
            config["theta_f"] = st.slider("Long-term θ (%)", 0.0, 5.0, 1.5, key="theta_f") / 100
            config["sigma_f"] = st.slider("Volatility σ (bps)", 10, 200, 120, key="sigma_f") / 10000

            st.subheader("FX Model (GBM)")
            config["fx_spot"] = st.number_input("Initial Spot", value=1.10, format="%.4f")
            config["fx_vol"] = st.slider("Volatility (%)", 5, 30, 12) / 100

        # Correlations
        with st.expander("🔗 Correlations"):
            config["corr_df"] = st.slider("Domestic-Foreign", -1.0, 1.0, 0.7, key="corr_df")
            config["corr_dx"] = st.slider("Domestic-FX", -1.0, 1.0, -0.3, key="corr_dx")
            config["corr_fx"] = st.slider("Foreign-FX", -1.0, 1.0, 0.4, key="corr_fx")

        # Collateral
        with st.expander("🏦 Collateral"):
            config["threshold"] = st.number_input("Threshold ($M)", value=1.0, step=0.1) * 1e6
            config["mta"] = st.number_input("MTA ($K)", value=100.0, step=10.0) * 1e3
            config["mpr_days"] = st.slider("MPR (days)", 0, 20, 10)
            config["im_mult"] = st.slider("IM Multiplier", 1.0, 3.0, 1.5)

        # Credit & Funding
        with st.expander("💰 Credit & Funding"):
            config["lgd_cpty"] = st.slider("LGD Counterparty (%)", 0, 100, 60) / 100
            config["lgd_own"] = st.slider("LGD Own (%)", 0, 100, 60) / 100
            config["lambda_cpty"] = st.slider("λ Counterparty (bps)", 0, 500, 120) / 10000
            config["lambda_own"] = st.slider("λ Own (bps)", 0, 500, 100) / 10000
            config["funding_spread"] = st.slider("Funding Spread (bps)", 0, 300, 100) / 10000
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
