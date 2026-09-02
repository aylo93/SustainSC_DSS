from __future__ import annotations

from pathlib import Path

from batch_completion_engine import BatchScenarioCompletionEngine
from sustainsc.case_studies import CASE_STUDIES
from sustainsc.dpp_import import read_dpp_workbook
from sustainsc.template_downloads import TEMPLATE_DIRECTORY, load_template_bytes


EXPECTED_CASES = {
    "cuba": {"scenarios": 24, "batches": 18, "events": 24, "reference": "BASE"},
    "romania": {
        "scenarios": 18,
        "batches": 13,
        "events": 26,
        "reference": "REEL_BASE",
    },
}


def test_case_study_assets_are_runnable_and_paired() -> None:
    for case in CASE_STUDIES:
        expected = EXPECTED_CASES[case.slug]
        result = BatchScenarioCompletionEngine("config").complete_batch_from_excel(
            TEMPLATE_DIRECTORY / case.mrv_workbook.filename
        )
        critical = result.qa_report[
            result.qa_report["severity"].eq("Critical")
            & result.qa_report["status"].eq("FAIL")
        ]
        batches, events = read_dpp_workbook(load_template_bytes(case.dpp_workbook))
        assert len(result.scenario_results) == expected["scenarios"]
        assert critical.empty
        assert len(batches) == expected["batches"]
        assert len(events) == expected["events"]
        assert set(batches["scenario_code"].dropna().astype(str)) == {expected["reference"]}
        assert expected["reference"] in result.scenario_results


def test_home_page_places_case_examples_in_the_right_column() -> None:
    source = Path("kpi_dashboard.py").read_text(encoding="utf-8")
    assert 'st.columns([1.55, 1], gap="large")' in source
    assert "with case_examples:" in source
    assert "render_case_study_examples()" in source
    assert 'key=f"download_{case.slug}_mrv_case_study"' in source
    assert 'key=f"download_{case.slug}_dpp_case_study"' in source
