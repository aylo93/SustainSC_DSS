"""Plotly views of retrieved MCDA56 evidence; no scientific scores are calculated."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from .mcda_evidence import (
    DIMENSIONS,
    MCDAEvidence,
    get_strategy_dimension_profile,
)


def build_reference_profile_figure(
    *,
    archetype: str,
    selected_strategy: str,
    leader_strategy: str,
    evidence: MCDAEvidence,
):
    """Compare compatible 0-100 dimension scores at one reference anchor."""

    strategies = [selected_strategy]
    if leader_strategy != selected_strategy:
        strategies.append(leader_strategy)
    records = []
    for strategy in strategies:
        profile = get_strategy_dimension_profile(archetype, strategy, evidence)
        role = (
            "Selected strategy / reference leader"
            if len(strategies) == 1
            else "Selected strategy" if strategy == selected_strategy else "Reference MCDA leader"
        )
        for dimension in [*DIMENSIONS, "SUSTAIN_INDEX_GEOM"]:
            records.append(
                {
                    "archetype": archetype,
                    "strategy": strategy,
                    "role": role,
                    "dimension": dimension.replace("SUSTAIN_INDEX_GEOM", "Global geometric index"),
                    "score": profile[dimension],
                }
            )
    frame = pd.DataFrame(records)
    figure = px.bar(
        frame,
        x="score",
        y="dimension",
        color="role",
        barmode="group",
        orientation="h",
        custom_data=["archetype", "strategy", "dimension", "score"],
        title="Reference Sustainability Profile",
        template="sustainscm",
        color_discrete_sequence=["#087F78", "#E39A29"],
    )
    figure.update_traces(
        hovertemplate=(
            "Archetype: %{customdata[0]}<br>Strategy: %{customdata[1]}<br>"
            "Dimension: %{customdata[2]}<br>Score: %{customdata[3]:.3f}<extra></extra>"
        )
    )
    figure.update_xaxes(title="Normalized sustainability score (0–100)", range=[0, 100])
    figure.update_yaxes(title=None, categoryorder="array", categoryarray=list(reversed(frame["dimension"].unique())))
    figure.update_layout(legend_title_text=None, height=430, margin=dict(l=20, r=20, t=60, b=30))
    return figure


def build_mean_rank_figure(
    *, method: str, selected_strategy: str, evidence: MCDAEvidence
):
    """Show all six stored mean ranks with rank 1 at the top of the axis."""

    method = method.upper()
    if method not in {"WSM", "TOPSIS"}:
        raise ValueError("method must be WSM or TOPSIS")
    frame = evidence.strategy_robustness.loc[
        evidence.strategy_robustness["method"].eq(method)
    ].copy()
    frame["selection"] = frame["strategy"].map(
        lambda value: "Strategy being evaluated" if value == selected_strategy else "Other strategy"
    )
    frame = frame.sort_values(["mean_rank", "strategy"])
    figure = px.scatter(
        frame,
        x="strategy",
        y="mean_rank",
        color="selection",
        custom_data=["strategy", "method", "mean_rank", "rank1_count", "top2_count"],
        title="Cross-Archetype Mean Strategy Rank",
        template="sustainscm",
        color_discrete_map={
            "Strategy being evaluated": "#E39A29",
            "Other strategy": "#087F78",
        },
    )
    figure.update_traces(
        hovertemplate=(
            "Strategy: %{customdata[0]}<br>Method: %{customdata[1]}<br>"
            "Mean rank: %{customdata[2]:.3f}<br>Rank-1 count: %{customdata[3]}<br>"
            "Top-2 count: %{customdata[4]}<extra></extra>"
        ),
        marker={"size": 18, "line": {"width": 1, "color": "white"}},
    )
    figure.update_yaxes(
        title="Mean rank (1 = strongest)",
        autorange="reversed",
        range=[6.25, 0.75],
        dtick=1,
    )
    figure.update_xaxes(title=None, categoryorder="array", categoryarray=frame["strategy"].tolist())
    figure.update_layout(showlegend=True, legend_title_text=None, height=430, margin=dict(l=20, r=20, t=60, b=30))
    return figure
