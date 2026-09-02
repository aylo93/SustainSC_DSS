from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batch_completion_engine import BatchScenarioCompletionEngine
from scenario_completion_engine import ScenarioCompletionEngine
from sustainsc.mrv_schema_v2 import parse_mrv_workbook


FIXTURE = Path("tests/fixtures/mrv_final/SustainSCM_Cuba_MRV_Scenario_Completion_FINAL_BOUNDARY_RECONCILED.xlsx")


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
        "L2": 1355, "L6": 811, "L3": 140, "BASE": 107,
        "L1": 71, "L4": 58, "L5": 26,
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
        "QA_DPP_VALIDATION_PENDING",
    }
    assert "QA_CH7_COMPARISON" not in set(result.production_qa_report["check_id"])
    assert set(result.regression_comparison_report["comparison_status"]) <= {
        "MATCH", "ROUNDING_ONLY", "MISSING_EXPECTED_VALUE", "UNRESOLVED_DIFFERENCE"
    }
    assert not result.regression_comparison_report["comparison_status"].eq("MISSING_EXPECTED_VALUE").any()
    assert not result.production_qa_report["check_id"].eq("QA_STRICT_REGRESSION").any()


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


def test_l3_diagnostics_prove_exact_and_scope_resolution():
    result = _complete_cuba()
    diagnostics = result.l3_permission_diagnostics.set_index(["scenario_code", "variable_name"])
    des = diagnostics.loc[("DES_BASE_2035", "electricity_kwh")]
    maintenance = diagnostics.loc[("DES_BASE_2035", "maintenance_cost_eur")]
    sd_water = diagnostics.loc[("SD_BAU_2035", "water_withdrawn_m3")]
    assert des["exact_override_active"] and des["L3_selected"]
    assert maintenance["exact_override_active"] and not maintenance["L3_selected"]
    assert not sd_water["exact_override_found"]
    assert "SD_MODEL_OUTPUTS:L2,L3,L6" in sd_water["scope_rules"]
    assert sd_water["L3_selected"]


def test_permission_precedence_fallback_inactive_and_declared_strategies_only():
    parsed = parse_mrv_workbook(FIXTURE)
    frames = {
        "dictionary": parsed.variable_dictionary,
        "scope": parsed.strategy_scope,
        "overrides": parsed.variable_overrides.copy(),
        "rules": parsed.mrv_rules,
        "bridges": parsed.bridge_rules,
    }
    engine = ScenarioCompletionEngine("config", config_frames=frames)
    maintenance = engine.resolve_permission(["LOGISTICS_REDESIGN"], "maintenance_cost_eur", "L3")
    fallback = engine.resolve_permission(["LOGISTICS_REDESIGN"], "shipped_volume_total", "L3")
    assert maintenance.resolution_source == "exact override" and not maintenance.allows("L3")
    assert fallback.resolution_source == "strategy scope" and fallback.allows("L3")

    frames["overrides"].loc[
        (frames["overrides"].strategy_code == "LOGISTICS_REDESIGN")
        & (frames["overrides"].variable_name == "electricity_kwh"), "active"
    ] = "No"
    inactive_engine = ScenarioCompletionEngine("config", config_frames=frames)
    inactive = inactive_engine.resolve_permission(["LOGISTICS_REDESIGN"], "electricity_kwh", "L3")
    assert inactive.resolution_source == "strategy scope"
    assert not inactive_engine.resolve_permission(["SD_BAU"], "water_withdrawn_m3", "L3").allows("L3")
    assert inactive_engine.resolve_permission(
        ["SD_BAU", "SD_MODEL_OUTPUTS"], "water_withdrawn_m3", "L3"
    ).allows("L3")


def test_sd_unmodelled_physical_flows_preserve_intensity_and_t3_base():
    review = _complete_cuba().completion_review.set_index(["scenario_code", "variable_name"])
    for scenario in sorted({code for code, _ in review.index if code.startswith("SD_")}):
        assert review.loc[(scenario, "water_intensity_fu"), "completed_value"] == pytest.approx(
            review.loc[("BASE", "water_intensity_fu"), "completed_value"], rel=1e-9
        )
        assert review.loc[(scenario, "mrv_coverage"), "completed_value"] == pytest.approx(70.0)


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


def test_transport_multiply_factor_dispatch_and_unit_contract():
    parsed = parse_mrv_workbook(FIXTURE)
    engine = ScenarioCompletionEngine(
        "config", config_frames={
            "dictionary": parsed.variable_dictionary, "scope": parsed.strategy_scope,
            "overrides": parsed.variable_overrides, "rules": parsed.mrv_rules,
            "bridges": parsed.bridge_rules, "factor_register": parsed.factor_register,
            "default_factor_set_id": "CUBA_CASE_FINAL",
        },
    )
    rule = parsed.mrv_rules[parsed.mrv_rules.rule_id.eq("CUBA_R060")].iloc[0]
    state = {"transport_work_tkm": {"value": 44_928_000.0}}
    scenario = {"emission_factor_set_id": "CUBA_CASE_FINAL", "evaluation_timestamp": "2025-12-31"}
    assert engine._evaluate_mrv_rule(rule, state, {}, scenario=scenario) == pytest.approx(
        740.992699753271
    )
    active_analytical = parsed.factor_register.copy()
    active_analytical.loc[
        active_analytical.factor_code.eq("TRANSPORT_GHG_PER_TKM"), "analytical_role"
    ] = "Active analytical factor"
    engine.factor_register = active_analytical
    assert engine._evaluate_mrv_rule(rule, state, {}, scenario=scenario) == pytest.approx(
        740.992699753271
    )
    reference_only = active_analytical.copy()
    reference_only.loc[
        reference_only.factor_code.eq("TRANSPORT_GHG_PER_TKM"), "analytical_role"
    ] = "Reference only"
    engine.factor_register = reference_only
    with pytest.raises(ValueError, match="not authorized"):
        engine._evaluate_mrv_rule(rule, state, {}, scenario=scenario)
    engine.factor_register = active_analytical
    wrong_type = active_analytical.copy()
    wrong_type.loc[
        wrong_type.factor_code.eq("TRANSPORT_GHG_PER_TKM"), "factor_type"
    ] = "REFERENCE"
    engine.factor_register = wrong_type
    with pytest.raises(ValueError, match="must use factor type EMISSION"):
        engine._evaluate_mrv_rule(rule, state, {}, scenario=scenario)
    engine.factor_register = active_analytical
    malformed = rule.copy()
    malformed["parameter"] = "not-json"
    with pytest.raises(ValueError, match="valid JSON"):
        engine._evaluate_mrv_rule(malformed, state, {}, scenario=scenario)
    wrong_unit = parsed.factor_register.copy()
    wrong_unit.loc[wrong_unit.factor_code.eq("TRANSPORT_GHG_PER_TKM"), "unit"] = "kgCO2e/tkm"
    engine.factor_register = wrong_unit
    with pytest.raises(ValueError, match="must use tCO2e/tkm"):
        engine._evaluate_mrv_rule(rule, state, {}, scenario=scenario)


def test_e7_boundary_precedence_placeholders_and_intensity_recalculation():
    result = _complete_cuba()
    review = result.completion_review.set_index(["scenario_code", "variable_name"])
    parsed = result.parsed_workbook
    base = review.loc[("BASE", "transport_ghg_tco2e")]
    des = review.loc[("DES_BASE_2025", "transport_ghg_tco2e")]
    social = review.loc[("SOCIAL_PUSH", "transport_ghg_tco2e")]
    vsm = review.loc[("VSMC_KAIZEN", "transport_ghg_tco2e")]
    assert base.completed_value == pytest.approx(740.992699753271)
    assert "TRANSPORT_GHG_PER_TKM" in base.provenance and "EF_DIESEL" not in base.provenance
    assert des.completed_value == pytest.approx(607.64) and des.rule_id == "BR_DES_TRANSPORT_GHG"
    assert social.completed_value == pytest.approx(base.completed_value) and social.rule_id == "CUBA_R060"
    assert vsm.completed_value == pytest.approx(666.893429777944) and vsm.rule_id == "CUBA_R060"
    assert review.loc[("VSMC_KAIZEN", "transport_ghg_intensity_fu"), "completed_value"] == pytest.approx(2.2053354159323546)
    legacy = parsed.native_outputs[
        parsed.native_outputs.native_variable.astype(str).eq("transport_ghg_total_tco2e")
        & parsed.native_outputs.scenario_code.isin(["BASE", "SOCIAL_PUSH"])
    ]
    assert len(legacy) == 2 and legacy.use_in_completion.astype(str).str.lower().eq("no").all()
    assert not parsed.direct_inputs[
        parsed.direct_inputs.scenario_code.eq("VSMC_KAIZEN")
        & parsed.direct_inputs.variable_name.eq("transport_ghg_tco2e")
    ].any(axis=None)
    transport_qa = result.qa_report[result.qa_report.check_id.str.startswith("TRANSPORT_GHG_")]
    assert not transport_qa.status.eq("FAIL").any()
