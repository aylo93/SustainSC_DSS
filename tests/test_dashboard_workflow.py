from __future__ import annotations

from pathlib import Path

import pandas as pd

from sustainsc.dashboard_workflow import (
    assess_analysis_readiness,
    has_restrictive_filters,
)


def complete_dashboard_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario_code": scenario, "dimension": dimension, "kpi_code": f"{scenario}-{dimension}"}
            for scenario in ("BASE", "SC-1")
            for dimension in ("environmental", "economic", "social", "technological")
        ]
    )


def test_restrictive_filters_are_order_independent():
    all_dimensions = ["environmental", "economic"]
    all_levels = ["operational", "strategic"]
    all_flows = ["energy", "information"]
    assert not has_restrictive_filters(
        list(reversed(all_dimensions)),
        all_dimensions,
        list(reversed(all_levels)),
        all_levels,
        list(reversed(all_flows)),
        all_flows,
    )
    assert has_restrictive_filters(
        ["environmental"],
        all_dimensions,
        all_levels,
        all_levels,
        all_flows,
        all_flows,
    )


def test_filtered_table_does_not_mutate_full_dataset():
    full_dashboard_df = complete_dashboard_frame()
    original = full_dashboard_df.copy(deep=True)
    filtered_table_df = full_dashboard_df.copy()
    filtered_table_df = filtered_table_df[
        filtered_table_df["dimension"] == "environmental"
    ]
    pd.testing.assert_frame_equal(full_dashboard_df, original)
    assert set(filtered_table_df["dimension"]) == {"environmental"}


def test_complete_data_is_ready_and_filters_can_be_cleared():
    frame = complete_dashboard_frame()
    readiness = assess_analysis_readiness(
        frame,
        all_scenarios=["SC-1", "BASE"],
        reference_scenario="BASE",
    )
    assert readiness.ready
    assert not readiness.message


def test_incomplete_data_returns_actionable_readiness_message():
    frame = complete_dashboard_frame()
    frame = frame[
        (frame["dimension"] != "social") & (frame["scenario_code"] == "BASE")
    ]
    readiness = assess_analysis_readiness(
        frame,
        all_scenarios=["BASE", "SC-1"],
        reference_scenario="BASE",
    )
    assert not readiness.ready
    assert readiness.missing_dimensions == ("social",)
    assert readiness.missing_scenarios == ("SC-1",)
    assert "At least two scenarios" in readiness.message


def test_dashboard_has_single_import_page_and_dpp_after_analytics():
    source = Path("kpi_dashboard.py").read_text(encoding="utf-8")
    assert "Rebuild demo (full)" not in source
    assert "Import measurements (CSV)" not in source
    assert source.count("render_scenario_completion_page(") == 1
    assert source.rfind("render_dpp_section()") > source.find(
        "Composite Indices, Sensitivity & MCDA"
    )
    assert 'st.session_state["last_import_run_id"]' in source
    assert "load_normalized_results.clear()" in source
