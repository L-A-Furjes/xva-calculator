"""
Export utilities for xVA results.

Provides CSV, JSON, and Excel export functionality.
"""

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from xva_core._types import FloatArray
from xva_core.xva.result import XVAResult


def export_to_csv(
    df: pd.DataFrame,
    path: str | Path,
    float_format: str = "%.4f",
) -> None:
    """
    Export DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    path : str | Path
        Output file path
    float_format : str
        Format string for floats
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)


def export_to_json(
    data: dict[str, Any],
    path: str | Path,
    indent: int = 2,
) -> None:
    """
    Export dictionary to JSON.

    Parameters
    ----------
    data : dict
        Data to export
    path : str | Path
        Output file path
    indent : int
        JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists
    def convert(obj: Any) -> Any:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    converted = convert(data)

    with open(path, "w") as f:
        json.dump(converted, f, indent=indent)


def create_excel_report(
    exposure_df: pd.DataFrame,
    xva_df: pd.DataFrame,
    saccr_df: pd.DataFrame | None = None,
    portfolio_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> BytesIO:
    """
    Create multi-sheet Excel report.

    Parameters
    ----------
    exposure_df : pd.DataFrame
        Exposure time series
    xva_df : pd.DataFrame
        xVA breakdown
    saccr_df : pd.DataFrame | None
        SA-CCR table
    portfolio_df : pd.DataFrame | None
        Portfolio table
    config : dict | None
        Configuration snapshot

    Returns
    -------
    BytesIO
        Excel file as bytes buffer
    """
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Exposure sheet
        exposure_df.to_excel(writer, sheet_name="Exposure", index=False)

        # xVA sheet
        xva_df.to_excel(writer, sheet_name="xVA Breakdown", index=False)

        # SA-CCR sheet
        if saccr_df is not None:
            saccr_df.to_excel(writer, sheet_name="SA-CCR", index=False)

        # Portfolio sheet
        if portfolio_df is not None:
            portfolio_df.to_excel(writer, sheet_name="Portfolio", index=False)

        # Config sheet
        if config is not None:
            config_df = _config_to_df(config)
            config_df.to_excel(writer, sheet_name="Configuration", index=False)

    buffer.seek(0)
    return buffer


def _config_to_df(config: dict[str, Any]) -> pd.DataFrame:
    """Convert nested config dict to flat DataFrame."""
    rows = []

    def flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                flatten(v, new_prefix)
        else:
            rows.append({"Parameter": prefix, "Value": str(obj)})

    flatten(config)
    return pd.DataFrame(rows)


def create_summary_report(
    time_grid: FloatArray,
    epe: FloatArray,
    ene: FloatArray,
    xva_result: XVAResult,
    notional: float,
    output_dir: str | Path,
    prefix: str = "xva_report",
) -> dict[str, Path]:
    """
    Create complete summary report with multiple files.

    Parameters
    ----------
    time_grid : FloatArray
        Time grid
    epe : FloatArray
        EPE profile
    ene : FloatArray
        ENE profile
    xva_result : XVAResult
        xVA calculation result
    notional : float
        Total notional
    output_dir : str | Path
        Output directory
    prefix : str
        File name prefix

    Returns
    -------
    dict[str, Path]
        Dictionary of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files = {}

    # Exposure CSV
    exposure_df = pd.DataFrame(
        {
            "Time": time_grid,
            "EPE": epe,
            "ENE": ene,
        }
    )
    exposure_path = output_dir / f"{prefix}_exposure.csv"
    export_to_csv(exposure_df, exposure_path)
    created_files["exposure"] = exposure_path

    # xVA CSV
    from xva_core.reporting.tables import create_xva_breakdown_table

    xva_df = create_xva_breakdown_table(xva_result, notional)
    xva_path = output_dir / f"{prefix}_xva.csv"
    export_to_csv(xva_df, xva_path)
    created_files["xva"] = xva_path

    # Summary JSON
    summary = {
        "xva": xva_result.to_dict(),
        "exposure": {
            "peak_epe": float(epe.max()),
            "peak_ene": float(ene.max()),
            "avg_epe": float(epe.mean()),
            "avg_ene": float(ene.mean()),
        },
        "notional": notional,
        "horizon": float(time_grid[-1]),
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    export_to_json(summary, summary_path)
    created_files["summary"] = summary_path

    return created_files


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency string.

    Parameters
    ----------
    value : float
        Value to format
    decimals : int
        Decimal places

    Returns
    -------
    str
        Formatted string like "$1,234.56M"
    """
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"${value/1e9:,.{decimals}f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:,.{decimals}f}M"
    elif abs_value >= 1e3:
        return f"${value/1e3:,.{decimals}f}K"
    else:
        return f"${value:,.{decimals}f}"
