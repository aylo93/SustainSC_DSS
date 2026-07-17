"""Central Plotly template and chart helpers."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from .theme import (
    BACKGROUND,
    BORDER,
    DIMENSION_COLORS,
    ECONOMIC,
    ENVIRONMENTAL,
    PRIMARY,
    SOCIAL,
    SURFACE,
    TECHNOLOGICAL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

SCENARIO_COLORS = [PRIMARY, "#376B8F", "#8A6A45", "#695C9E", "#4F7E65", "#A35757"]

SUSTAINSCM_PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={"family": "Inter, system-ui, -apple-system, Segoe UI, sans-serif", "size": 13, "color": TEXT_PRIMARY},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SCENARIO_COLORS,
        margin={"l": 55, "r": 25, "t": 65, "b": 50},
        title={"x": 0.01, "xanchor": "left", "font": {"size": 18}},
        legend={"orientation": "h", "y": 1.08, "x": 0, "title": None},
        hoverlabel={"bgcolor": SURFACE, "bordercolor": BORDER, "font": {"color": TEXT_PRIMARY}},
        hovermode="x unified",
        xaxis={"gridcolor": "#E7ECEA", "linecolor": BORDER, "zerolinecolor": BORDER, "title_font": {"color": TEXT_SECONDARY}},
        yaxis={"gridcolor": "#E7ECEA", "linecolor": BORDER, "zerolinecolor": BORDER, "title_font": {"color": TEXT_SECONDARY}},
    )
)

pio.templates["sustainscm"] = SUSTAINSCM_PLOTLY_TEMPLATE

DIMENSION_COLOR_MAP = {
    "environmental": ENVIRONMENTAL,
    "economic": ECONOMIC,
    "social": SOCIAL,
    "technological": TECHNOLOGICAL,
}
