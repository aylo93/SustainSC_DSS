from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batch_completion_engine import BatchScenarioCompletionEngine
from scenario_completion_engine import ScenarioCompletionEngine
from sustainsc.mrv_schema_v2 import parse_mrv_workbook


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
        "L2": 1355, "L6": 817, "L3": 124, "BASE": 107,
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


def test_des_ghg_is_factor_recalculated_after_energy_scaling_and_maintenance_is_retained():
    result = _complete_cuba()
    review = result.completion_review.set_index(["scenario_code", "variable_name"])
    base = review.loc[("BASE", "ghg_intensity_fu"), "completed_value"]
    des = review.loc[("DES_BASE_2035", "ghg_intensity_fu"), "completed_value"]
    ghg = review.loc[("DES_BASE_2035", "ghg_total_s1s2")]
    assert ghg["rule_level"] == "L2"
    assert ghg["rule_id"] == "MRV_R_GHG_S1S2_FACTORS"
    assert ghg["completed_value"] < review.loc[("BASE", "ghg_total_s1s2"), "completed_value"]
    assert des == pytest.approx(base, rel=1e-9)
    assert review.loc[("DES_BASE_2035", "maintenance_cost_eur"), "completed_value"] == review.loc[("BASE", "maintenance_cost_eur"), "completed_value"]
    assert review.loc[("DES_BASE_2035", "maintenance_cost_eur"), "rule_level"] == "L6"


def test_strong_ghg_evidence_is_not_overwritten_by_factor_rule():
    review = _complete_cuba().completion_review.set_index(["scenario_code", "variable_name"])
    assert review.loc[("MILP_MIN_COST", "ghg_total_s1s2"), "rule_level"] == "L1"
    assert review.loc[("SD_BAU_2035", "ghg_total_s1s2"), "rule_level"] == "L4"
    assert review.loc[("VSMC_KAIZEN", "ghg_total_s1s2"), "rule_level"] == "L1"


def test_factor_register_rejects_reference_only_and_missing_factors():
    parsed = parse_mrv_workbook(FIXTURE)
    engine = ScenarioCompletionEngine(
        "config",
        config_frames={
            "dictionary": parsed.variable_dictionary,
            "scope": parsed.strategy_scope,
            "overrides": parsed.variable_overrides,
            "rules": parsed.mrv_rules,
            "bridges": parsed.bridge_rules,
            "factor_register": parsed.factor_register,
            "default_factor_set_id": "CUBA_CASE_FINAL",
        },
    )
    timestamp = pd.Timestamp("2035-12-31")
    assert engine._resolve_analytical_factor("CUBA_CASE_FINAL", "EF_ELECTRICITY_CASE", timestamp) == pytest.approx(0.6161)
    with pytest.raises(ValueError, match="not authorized"):
        engine._resolve_analytical_factor("CUBA_CASE_FINAL", "EF_GRID_LOCATION_REFERENCE", timestamp)
    with pytest.raises(ValueError, match="found 0"):
        engine._resolve_analytical_factor("CUBA_CASE_FINAL", "MISSING", timestamp)
