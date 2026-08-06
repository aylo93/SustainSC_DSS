from pathlib import Path

import numpy as np

from batch_completion_engine import BatchScenarioCompletionEngine


FIXTURE = Path("tests/fixtures/mrv_final/SustainSCM_Cuba_MRV_Scenario_Completion_FINAL.xlsx")


def _complete_cuba():
    return BatchScenarioCompletionEngine("config").complete_batch_from_excel(FIXTURE)


def test_corrected_cuba_completion_is_structurally_complete():
    result = _complete_cuba()
    completed = result.software_upload

    assert len(result.scenario_results) == 24
    assert completed["variable_name"].nunique() == 107
    assert len(completed) == 2568
    assert not completed[["scenario_code", "variable_name"]].duplicated().any()
    assert completed["value"].notna().all()
    assert np.isfinite(completed["value"].astype(float)).all()
    assert result.completion_review["rule_level"].value_counts().to_dict() == {
        "L2": 1345, "L6": 802, "L3": 149, "BASE": 107,
        "L1": 71, "L4": 68, "L5": 26,
    }


def test_production_qa_and_historical_regression_are_independent():
    result = _complete_cuba()
    critical = result.production_qa_report[
        (result.production_qa_report["severity"] == "Critical")
        & (result.production_qa_report["status"] == "FAIL")
    ]
    warnings = result.production_qa_report[
        result.production_qa_report["status"] == "WARN"
    ]

    assert critical.empty
    assert not result.has_critical_failures
    assert set(warnings["check_id"]) <= {
        "QA_DPP_VALIDATION_PENDING", "QA_GHG_BOUNDARY",
    }
    assert "QA_CH7_COMPARISON" not in set(result.production_qa_report["check_id"])
    assert set(result.regression_comparison_report["comparison_status"]) <= {
        "MATCH", "ROUNDING_ONLY", "MISSING_EXPECTED_VALUE", "UNRESOLVED_DIFFERENCE"
    }
    assert result.regression_comparison_report["comparison_status"].eq("MATCH").all()


def test_corrected_milp_name_and_unit_aware_tolerances_resolve():
    result = _complete_cuba()
    comparison = result.regression_comparison_report

    assert "MILP_CO2CAP" in result.scenario_results
    assert "MILP_CO2CAP_940" not in result.scenario_results
    assert (comparison["tolerance"] > 0).all()
    assert comparison["relative_difference"].notna().all()
