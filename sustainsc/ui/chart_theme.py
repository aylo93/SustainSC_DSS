"""Central Plotly template and chart helpers."""

from __future__ import annotations

import logging

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd

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
logger = logging.getLogger(__name__)

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


def ranking_chart_height(category_count: int) -> int:
    """Provide enough vertical room for every horizontal ranking category."""
    return max(440, min(1100, 34 * max(int(category_count), 0) + 140))


def ranking_left_margin(labels: list[str]) -> int:
    """Reserve space for the longest scenario label without wasting the canvas."""
    longest = max((len(str(label)) for label in labels), default=0)
    return min(max(120, longest * 8 + 28), 300)


def build_horizontal_ranking_chart(
    frame: pd.DataFrame,
    *,
    scenario_col: str,
    score_col: str,
    title: str,
    x_title: str,
    color: str,
    decimals: int = 2,
) -> go.Figure:
    """Build a deterministic ranking chart with one visible label per bar."""
    missing = {scenario_col, score_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Ranking data missing columns: {sorted(missing)}")
    plot_frame = (
        frame.dropna(subset=[scenario_col, score_col])
        .sort_values([score_col, scenario_col], ascending=[True, True])
        .copy()
    )
    if plot_frame.empty:
        raise ValueError("Ranking data contain no plottable rows.")
    if not plot_frame[scenario_col].is_unique:
        raise ValueError("Ranking scenario labels must be unique.")
    if plot_frame[scenario_col].isna().any() or plot_frame[score_col].isna().any():
        raise ValueError("Ranking labels and scores must be non-null.")

    plot_frame[scenario_col] = plot_frame[scenario_col].astype(str)
    categories = plot_frame[scenario_col].tolist()
    if len(categories) != plot_frame[scenario_col].nunique():
        raise ValueError("Every ranking bar must have exactly one scenario category.")
    logger.debug("Ranking plot scenarios (%s): %s", title, categories)

    figure = px.bar(
        plot_frame,
        x=score_col,
        y=scenario_col,
        orientation="h",
        title=title,
        labels={score_col: x_title, scenario_col: "Scenario"},
        template="sustainscm",
        color_discrete_sequence=[color],
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=categories,
        showticklabels=True,
        automargin=True,
        ticklabelposition="outside",
        title_text="Scenario",
    )
    figure.update_traces(
        texttemplate=f"%{{x:.{int(decimals)}f}}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            f"<b>%{{y}}</b><br>{x_title}: %{{x:.{int(decimals)}f}}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=ranking_chart_height(len(categories)),
        margin={
            "l": ranking_left_margin(categories),
            "r": 80,
            "t": 85,
            "b": 60,
        },
    )
    return figure
