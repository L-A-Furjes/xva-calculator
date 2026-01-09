"""
Plotting utilities for xVA visualization.

Provides both Matplotlib and Plotly chart generators.
"""

from typing import Any

import matplotlib.pyplot as plt

from xva_core._types import FloatArray

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots  # noqa: F401

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore


def create_exposure_plot(
    time_grid: FloatArray,
    epe: FloatArray,
    ene: FloatArray | None = None,
    pfe_95: FloatArray | None = None,
    title: str = "Exposure Profile",
    use_plotly: bool = True,
) -> Any:
    """
    Create exposure profile plot.

    Parameters
    ----------
    time_grid : FloatArray
        Time points in years
    epe : FloatArray
        Expected positive exposure
    ene : FloatArray | None
        Expected negative exposure
    pfe_95 : FloatArray | None
        95% PFE
    title : str
        Chart title
    use_plotly : bool
        Use Plotly if available (default True)

    Returns
    -------
    Any
        Plotly Figure or Matplotlib Figure
    """
    if use_plotly and PLOTLY_AVAILABLE:
        return _create_exposure_plot_plotly(time_grid, epe, ene, pfe_95, title)
    else:
        return _create_exposure_plot_mpl(time_grid, epe, ene, pfe_95, title)


def _create_exposure_plot_plotly(
    time_grid: FloatArray,
    epe: FloatArray,
    ene: FloatArray | None,
    pfe_95: FloatArray | None,
    title: str,
) -> Any:
    """Create exposure plot using Plotly."""
    fig = go.Figure()

    # Convert to millions for display
    epe_m = epe / 1e6

    fig.add_trace(
        go.Scatter(
            x=time_grid,
            y=epe_m,
            mode="lines",
            name="EPE",
            line={"color": "#FF4B4B", "width": 2},
            hovertemplate="<b>EPE</b><br>Time: %{x:.2f}Y<br>Value: $%{y:.2f}M<extra></extra>",
        )
    )

    if ene is not None:
        ene_m = ene / 1e6
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=ene_m,
                mode="lines",
                name="ENE",
                line={"color": "#00CC96", "width": 2},
                hovertemplate="<b>ENE</b><br>Time: %{x:.2f}Y<br>Value: $%{y:.2f}M<extra></extra>",
            )
        )

    if pfe_95 is not None:
        pfe_m = pfe_95 / 1e6
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pfe_m,
                mode="lines",
                name="PFE 95%",
                line={"color": "#AB63FA", "width": 2, "dash": "dash"},
                hovertemplate="<b>PFE 95%</b><br>Time: %{x:.2f}Y<br>Value: $%{y:.2f}M<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (years)",
        yaxis_title="Exposure ($M)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    return fig


def _create_exposure_plot_mpl(
    time_grid: FloatArray,
    epe: FloatArray,
    ene: FloatArray | None,
    pfe_95: FloatArray | None,
    title: str,
) -> plt.Figure:
    """Create exposure plot using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epe_m = epe / 1e6
    ax.plot(time_grid, epe_m, "r-", linewidth=2, label="EPE")

    if ene is not None:
        ene_m = ene / 1e6
        ax.plot(time_grid, ene_m, "g-", linewidth=2, label="ENE")

    if pfe_95 is not None:
        pfe_m = pfe_95 / 1e6
        ax.plot(time_grid, pfe_m, "m--", linewidth=2, label="PFE 95%")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure ($M)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_xva_bar_chart(
    xva_values: dict[str, float],
    title: str = "xVA Components",
    use_plotly: bool = True,
) -> Any:
    """
    Create xVA bar chart.

    Parameters
    ----------
    xva_values : dict[str, float]
        Dictionary with CVA, DVA, FVA, MVA, KVA values
    title : str
        Chart title
    use_plotly : bool
        Use Plotly if available

    Returns
    -------
    Any
        Figure object
    """
    if use_plotly and PLOTLY_AVAILABLE:
        return _create_xva_bar_chart_plotly(xva_values, title)
    else:
        return _create_xva_bar_chart_mpl(xva_values, title)


def _create_xva_bar_chart_plotly(
    xva_values: dict[str, float],
    title: str,
) -> Any:
    """Create xVA bar chart using Plotly."""
    colors = {
        "CVA": "#FF4B4B",
        "DVA": "#00CC96",
        "FVA": "#FFA500",
        "MVA": "#9467BD",
        "KVA": "#1F77B4",
    }

    metrics = ["CVA", "DVA", "FVA", "MVA", "KVA"]
    values = [xva_values.get(m, 0) for m in metrics]
    # DVA is shown as negative (benefit)
    display_values = [
        v if m != "DVA" else -v for m, v in zip(metrics, values, strict=False)
    ]
    values_m = [v / 1e6 for v in display_values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values_m,
            marker_color=[colors.get(m, "#1F77B4") for m in metrics],
            text=[f"${v:.2f}M" for v in values_m],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>$%{y:.2f}M<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title="Value ($M)",
        showlegend=False,
        template="plotly_dark",
        height=400,
    )

    return fig


def _create_xva_bar_chart_mpl(
    xva_values: dict[str, float],
    title: str,
) -> plt.Figure:
    """Create xVA bar chart using Matplotlib."""
    colors = ["#FF4B4B", "#00CC96", "#FFA500", "#9467BD", "#1F77B4"]

    metrics = ["CVA", "DVA", "FVA", "MVA", "KVA"]
    values = [xva_values.get(m, 0) for m in metrics]
    display_values = [
        v if m != "DVA" else -v for m, v in zip(metrics, values, strict=False)
    ]
    values_m = [v / 1e6 for v in display_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(metrics, values_m, color=colors)

    # Add value labels
    for bar, val in zip(bars, values_m, strict=False):
        height = bar.get_height()
        ax.annotate(
            f"${val:.2f}M",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Value ($M)")
    ax.set_title(title)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return fig


def create_xva_waterfall(
    xva_values: dict[str, float],
    title: str = "xVA Waterfall",
    use_plotly: bool = True,
) -> Any:
    """
    Create xVA waterfall chart.

    Parameters
    ----------
    xva_values : dict[str, float]
        Dictionary with xVA values
    title : str
        Chart title
    use_plotly : bool
        Use Plotly if available

    Returns
    -------
    Any
        Figure object
    """
    if use_plotly and PLOTLY_AVAILABLE:
        return _create_xva_waterfall_plotly(xva_values, title)
    else:
        # Fallback to bar chart
        return _create_xva_bar_chart_mpl(xva_values, title)


def _create_xva_waterfall_plotly(
    xva_values: dict[str, float],
    title: str,
) -> Any:
    """Create xVA waterfall chart using Plotly."""
    metrics = ["CVA", "DVA", "FVA", "MVA", "KVA", "Total"]
    values = [
        xva_values.get("CVA", 0),
        -xva_values.get("DVA", 0),  # DVA is a benefit
        xva_values.get("FVA", 0),
        xva_values.get("MVA", 0),
        xva_values.get("KVA", 0),
        xva_values.get("total", sum(xva_values.values())),
    ]
    values_m = [v / 1e6 for v in values]

    measures = ["relative", "relative", "relative", "relative", "relative", "total"]

    fig = go.Figure(
        go.Waterfall(
            name="xVA",
            orientation="v",
            measure=measures,
            x=metrics,
            y=values_m,
            textposition="outside",
            text=[f"${v:.2f}M" for v in values_m],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#FF4B4B"}},
            decreasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#1F77B4"}},
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title="Value ($M)",
        template="plotly_dark",
        height=400,
    )

    return fig


def create_comparison_plot(
    time_grid: FloatArray,
    data_sets: dict[str, FloatArray],
    title: str = "Comparison",
    ylabel: str = "Value ($M)",
) -> Any:
    """
    Create comparison plot with multiple data series.

    Parameters
    ----------
    time_grid : FloatArray
        Time points
    data_sets : dict[str, FloatArray]
        Named data series to plot
    title : str
        Chart title
    ylabel : str
        Y-axis label

    Returns
    -------
    Any
        Figure object
    """
    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        for name, data in data_sets.items():
            fig.add_trace(
                go.Scatter(
                    x=time_grid,
                    y=data / 1e6,
                    mode="lines",
                    name=name,
                    hovertemplate=f"<b>{name}</b><br>Time: %{{x:.2f}}Y<br>Value: $%{{y:.2f}}M<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time (years)",
            yaxis_title=ylabel,
            template="plotly_dark",
            height=400,
        )

        return fig

    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, data in data_sets.items():
            ax.plot(time_grid, data / 1e6, linewidth=2, label=name)

        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
