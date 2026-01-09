"""
Table generation utilities for xVA reporting.

Creates formatted pandas DataFrames for display and export.
"""

from typing import Any, Sequence

import pandas as pd

from xva_core._types import FloatArray
from xva_core.instruments.base import Instrument
from xva_core.reg.saccr import SACCRResult
from xva_core.xva.result import XVAResult


def create_xva_breakdown_table(
    xva_result: XVAResult,
    notional: float,
    additional_info: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Create xVA breakdown table.

    Parameters
    ----------
    xva_result : XVAResult
        xVA calculation result
    notional : float
        Total notional for bps calculation
    additional_info : dict[str, str] | None
        Additional info for each component (e.g., key drivers)

    Returns
    -------
    pd.DataFrame
        Formatted breakdown table
    """
    bps = xva_result.to_bps(notional)

    data = {
        "Component": ["CVA", "DVA (benefit)", "FVA", "MVA", "KVA", "Total xVA"],
        "Value ($M)": [
            xva_result.cva / 1e6,
            -xva_result.dva / 1e6,  # Show as negative (benefit)
            xva_result.fva / 1e6,
            xva_result.mva / 1e6,
            xva_result.kva / 1e6,
            xva_result.total / 1e6,
        ],
        "bps of Notional": [
            bps["cva"],
            -bps["dva"],
            bps["fva"],
            bps["mva"],
            bps["kva"],
            bps["total"],
        ],
    }

    if additional_info:
        data["Key Driver"] = [
            additional_info.get("CVA", ""),
            additional_info.get("DVA", ""),
            additional_info.get("FVA", ""),
            additional_info.get("MVA", ""),
            additional_info.get("KVA", ""),
            "",
        ]

    df = pd.DataFrame(data)

    return df


def create_exposure_summary_table(
    time_grid: FloatArray,
    epe: FloatArray,
    ene: FloatArray,
    epe_coll: FloatArray | None = None,
    ene_coll: FloatArray | None = None,
    im_profile: FloatArray | None = None,
) -> pd.DataFrame:
    """
    Create exposure summary table with key metrics.

    Parameters
    ----------
    time_grid : FloatArray
        Time points
    epe : FloatArray
        Uncollateralized EPE
    ene : FloatArray
        Uncollateralized ENE
    epe_coll : FloatArray | None
        Collateralized EPE
    ene_coll : FloatArray | None
        Collateralized ENE
    im_profile : FloatArray | None
        Initial margin profile

    Returns
    -------
    pd.DataFrame
        Summary metrics table
    """
    import numpy as np

    metrics = []

    # Peak and average uncollateralized
    metrics.append(
        {
            "Metric": "Peak EPE (uncoll)",
            "Value ($M)": np.max(epe) / 1e6,
            "Time (Y)": time_grid[np.argmax(epe)],
        }
    )
    metrics.append(
        {
            "Metric": "Average EPE (uncoll)",
            "Value ($M)": np.mean(epe) / 1e6,
            "Time (Y)": None,
        }
    )
    metrics.append(
        {
            "Metric": "Peak ENE (uncoll)",
            "Value ($M)": np.max(ene) / 1e6,
            "Time (Y)": time_grid[np.argmax(ene)],
        }
    )

    # Collateralized if available
    if epe_coll is not None:
        metrics.append(
            {
                "Metric": "Peak EPE (coll)",
                "Value ($M)": np.max(epe_coll) / 1e6,
                "Time (Y)": time_grid[np.argmax(epe_coll)],
            }
        )
        reduction = (1 - np.max(epe_coll) / np.max(epe)) * 100 if np.max(epe) > 0 else 0
        metrics.append(
            {
                "Metric": "Collateral Reduction (%)",
                "Value ($M)": reduction,
                "Time (Y)": None,
            }
        )

    if ene_coll is not None:
        metrics.append(
            {
                "Metric": "Peak ENE (coll)",
                "Value ($M)": np.max(ene_coll) / 1e6,
                "Time (Y)": time_grid[np.argmax(ene_coll)],
            }
        )

    # IM if available
    if im_profile is not None:
        metrics.append(
            {
                "Metric": "Average IM",
                "Value ($M)": np.mean(im_profile) / 1e6,
                "Time (Y)": None,
            }
        )
        metrics.append(
            {
                "Metric": "Peak IM",
                "Value ($M)": np.max(im_profile) / 1e6,
                "Time (Y)": time_grid[np.argmax(im_profile)],
            }
        )

    return pd.DataFrame(metrics)


def create_exposure_timeseries_table(
    time_grid: FloatArray,
    epe: FloatArray,
    ene: FloatArray,
    epe_coll: FloatArray | None = None,
    ene_coll: FloatArray | None = None,
    im_profile: FloatArray | None = None,
) -> pd.DataFrame:
    """
    Create exposure time series table.

    Parameters
    ----------
    time_grid : FloatArray
        Time points
    epe : FloatArray
        EPE profile
    ene : FloatArray
        ENE profile
    epe_coll : FloatArray | None
        Collateralized EPE
    ene_coll : FloatArray | None
        Collateralized ENE
    im_profile : FloatArray | None
        IM profile

    Returns
    -------
    pd.DataFrame
        Time series table
    """
    data = {
        "Time (Y)": time_grid,
        "EPE ($M)": epe / 1e6,
        "ENE ($M)": ene / 1e6,
    }

    if epe_coll is not None:
        data["EPE Coll ($M)"] = epe_coll / 1e6

    if ene_coll is not None:
        data["ENE Coll ($M)"] = ene_coll / 1e6

    if im_profile is not None:
        data["IM ($M)"] = im_profile / 1e6

    return pd.DataFrame(data)


def create_saccr_table(
    saccr_result: SACCRResult,
) -> pd.DataFrame:
    """
    Create SA-CCR breakdown table.

    Parameters
    ----------
    saccr_result : SACCRResult
        SA-CCR calculation result

    Returns
    -------
    pd.DataFrame
        SA-CCR table
    """
    data = []

    # Per-trade breakdown
    for ta in saccr_result.trade_addons:
        data.append(
            {
                "Trade": ta.trade_id,
                "Asset Class": ta.asset_class.value,
                "Notional ($M)": ta.notional / 1e6,
                "Maturity (Y)": ta.maturity,
                "SF": ta.supervisory_factor,
                "Add-On ($M)": ta.addon / 1e6,
            }
        )

    # Summary row
    data.append(
        {
            "Trade": "TOTAL",
            "Asset Class": "-",
            "Notional ($M)": sum(ta.notional for ta in saccr_result.trade_addons) / 1e6,
            "Maturity (Y)": None,
            "SF": None,
            "Add-On ($M)": saccr_result.aggregate_addon / 1e6,
        }
    )

    return pd.DataFrame(data)


def create_saccr_summary_table(
    saccr_result: SACCRResult,
) -> pd.DataFrame:
    """
    Create SA-CCR summary table.

    Parameters
    ----------
    saccr_result : SACCRResult
        SA-CCR calculation result

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    data = [
        {"Component": "Replacement Cost (RC)", "Value ($M)": saccr_result.replacement_cost / 1e6},
        {"Component": "Aggregate Add-On", "Value ($M)": saccr_result.aggregate_addon / 1e6},
        {"Component": "Multiplier", "Value ($M)": saccr_result.multiplier},
        {"Component": "PFE", "Value ($M)": saccr_result.pfe / 1e6},
        {"Component": "EAD (alpha=1.4)", "Value ($M)": saccr_result.ead / 1e6},
    ]

    return pd.DataFrame(data)


def create_portfolio_table(
    instruments: Sequence[Instrument],
) -> pd.DataFrame:
    """
    Create portfolio summary table.

    Parameters
    ----------
    instruments : Sequence[Instrument]
        Portfolio instruments

    Returns
    -------
    pd.DataFrame
        Portfolio table
    """
    data = []

    for i, inst in enumerate(instruments):
        info = inst.to_dict()
        info["ID"] = f"Trade_{i+1}"
        info["Notional ($M)"] = inst.notional / 1e6
        info["Maturity (Y)"] = inst.maturity
        data.append(info)

    return pd.DataFrame(data)
