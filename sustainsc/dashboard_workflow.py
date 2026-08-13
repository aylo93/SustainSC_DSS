"""Pure helpers for Streamlit dashboard filtering and analysis readiness."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_DIMENSIONS = {"environmental", "economic", "social", "technological"}


def format_reference_value(
    reference_value: float | None,
    normalization_type: str | None,
) -> str:
    """Format reference semantics without contaminating numeric persistence."""

    if (normalization_type or "").strip() == "absolute_continuous":
        return "N/A — absolute thresholds"
    if reference_value is None or pd.isna(reference_value):
        return "Not available"
    return f"{float(reference_value):.6g}"


@dataclass(frozen=True)
class AnalysisReadiness:
    ready: bool
    missing_dimensions: tuple[str, ...]
    missing_scenarios: tuple[str, ...]
    message: str


def has_restrictive_filters(
    selected_dimensions,
    all_dimensions,
    selected_levels,
    all_levels,
    selected_flows,
    all_flows,
) -> bool:
    """Detect table filters with order-independent set comparisons."""

    return (
        set(selected_dimensions) != set(all_dimensions)
        or set(selected_levels) != set(all_levels)
        or set(selected_flows) != set(all_flows)
    )


def assess_analysis_readiness(
    dashboard_df: pd.DataFrame,
    *,
    all_scenarios: list[str],
    reference_scenario: str,
) -> AnalysisReadiness:
    """Validate the minimum complete input required by integrated analyses."""

    present_dimensions = set(dashboard_df.get("dimension", pd.Series(dtype=str)).dropna())
    present_scenarios = set(dashboard_df.get("scenario_code", pd.Series(dtype=str)).dropna())
    missing_dimensions = tuple(sorted(REQUIRED_DIMENSIONS - present_dimensions))
    missing_scenarios = tuple(sorted(set(all_scenarios) - present_scenarios))
    problems = []
    if dashboard_df.empty:
        problems.append("The normalized KPI dataset is empty.")
    if missing_dimensions:
        problems.append("Missing dimensions: " + ", ".join(missing_dimensions) + ".")
    if reference_scenario not in present_scenarios:
        problems.append(f"Reference scenario {reference_scenario} is missing.")
    if missing_scenarios:
        problems.append("Missing scenarios: " + ", ".join(missing_scenarios) + ".")
    return AnalysisReadiness(
        ready=not problems,
        missing_dimensions=missing_dimensions,
        missing_scenarios=missing_scenarios,
        message=" ".join(problems),
    )
