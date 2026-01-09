"""
Reporting module for xVA visualization and export.

Provides:
- Matplotlib plots for exposure profiles
- Plotly interactive charts
- DataFrame formatters
- CSV/Excel export utilities
"""

from xva_core.reporting.export import (
    create_excel_report,
    create_summary_report,
    export_to_csv,
    export_to_json,
)
from xva_core.reporting.plots import (
    create_exposure_plot,
    create_xva_bar_chart,
    create_xva_waterfall,
)
from xva_core.reporting.tables import (
    create_exposure_summary_table,
    create_saccr_table,
    create_xva_breakdown_table,
)

__all__ = [
    # Plots
    "create_exposure_plot",
    "create_xva_bar_chart",
    "create_xva_waterfall",
    # Tables
    "create_xva_breakdown_table",
    "create_exposure_summary_table",
    "create_saccr_table",
    # Export
    "export_to_csv",
    "export_to_json",
    "create_excel_report",
    "create_summary_report",
]
