from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sustainsc.dashboard_workflow import format_reference_value
from sustainsc.formula_registry import resolve_formula_id
from sustainsc.mrv_validation import CompletionValidationError, validate_completed_mrv


def test_current_dictionary_resolves_107_common_variables():
    dictionary = pd.read_csv("config/mrv_dictionary.csv")
    common = dictionary[
        dictionary["common_upload_variable"].astype(str).str.lower() == "yes"
    ]
    assert len(common) == 107
    assert common["variable_name"].is_unique


def test_completed_dataset_schema_and_failures_are_explicit():
    completed = pd.read_csv("data/measurements.csv", sep="\t")
    base = completed[completed["scenario_code"] == "BASE"].copy()
    legacy_result = validate_completed_mrv(
        base,
        dictionary_path="config/mrv_dictionary.csv",
        raise_on_error=False,
    )
    assert not legacy_result.is_valid
    assert (
        (legacy_result.report["finding"] == "unit_mismatch")
        & (legacy_result.report["variable_name"] == "dpp_valid_volume")
    ).any()

    dictionary = pd.read_csv("config/mrv_dictionary.csv").set_index("variable_name")
    base["unit"] = base["variable_name"].map(dictionary["canonical_unit"])
    result = validate_completed_mrv(
        base,
        dictionary_path="config/mrv_dictionary.csv",
        raise_on_error=False,
    )
    assert result.required_variable_count == 107
    assert result.is_valid

    broken = base[base["variable_name"] != "electricity_kwh"].copy()
    broken = pd.concat([broken, broken.iloc[[0]]], ignore_index=True)
    broken.loc[broken.index[0], "value"] = None
    result = validate_completed_mrv(
        broken,
        dictionary_path="config/mrv_dictionary.csv",
        raise_on_error=False,
    )
    assert not result.is_valid
    assert {"missing_variable", "duplicate_variable", "null_value"}.issubset(
        set(result.report["finding"])
    )
    with pytest.raises(CompletionValidationError):
        validate_completed_mrv(
            broken,
            dictionary_path="config/mrv_dictionary.csv",
        )


def test_formula_aliases_are_backward_compatible():
    assert resolve_formula_id("energy_fuel_cost_share") == "energy_cost_share"
    assert resolve_formula_id("oee_total") == "oee"
    assert resolve_formula_id("community_complaints_count") == "community_incidents_total"
    assert resolve_formula_id("energy_cost_share") == "energy_cost_share"
    assert resolve_formula_id("unknown_formula") == "unknown_formula"


def test_environmental_kpi_mapping_and_normalization_are_canonical():
    kpis = pd.read_csv("data/kpis.csv").set_index("code")
    expected = {
        "E1": ("Total GHG emissions (Scope 1+2)", "ghg_total_s1s2"),
        "E2": ("Specific GHG emissions per FU", "ghg_intensity_fu"),
        "E3": ("Energy intensity", "energy_intensity_fu"),
        "E4": ("Share of renewable energy", "renewable_energy_share"),
        "E5": ("Waste generation intensity", "waste_generation_intensity_fu"),
        "E6": ("Waste recovery rate", "waste_recovery_rate"),
        "E7": ("Transport GHG emissions intensity", "transport_ghg_intensity_fu"),
        "E8": ("Circularity ratio", "circularity_ratio"),
        "E9": ("Water consumption intensity", "water_intensity_fu"),
    }
    assert not kpis.index.duplicated().any()
    for code, (name, formula) in expected.items():
        assert kpis.loc[code, "name"] == name
        assert kpis.loc[code, "formula_id"] == formula

    rules = pd.read_csv("data/kpi_normalization_rules.csv").set_index("kpi_code")
    assert rules.loc["E5", "direction"] == "lower_better"
    assert rules.loc["E6", "direction"] == "higher_better"
    assert rules.loc["E6", "norm_method"] == "absolute_continuous"


def test_production_rules_do_not_fabricate_cost_or_dpp_validity():
    rules = pd.read_csv("config/mrv_rules.csv").set_index("target_variable")
    assert rules.loc["dpp_valid_volume", "rule_status"] == "INACTIVE"
    assert rules.loc["dpp_valid_volume", "operation"] == "DPP_GENERATED"
    assert rules.loc["electricity_cost_eur", "rule_status"] == "INACTIVE"
    assert rules.loc["fuel_cost_eur", "rule_status"] == "INACTIVE"
    assert not (
        (rules["rule_status"] == "ACTIVE")
        & (rules.index.isin(["electricity_cost_eur", "fuel_cost_eur"]))
        & (rules["operation"] == "MULTIPLY_CONSTANT")
    ).any()


def test_absolute_reference_display_preserves_numeric_null_semantics():
    assert format_reference_value(None, "absolute_continuous") == "N/A — absolute thresholds"
    assert format_reference_value(None, "relative_vs_base_pct") == "Not available"
    assert format_reference_value(12.5, "relative_vs_base_pct") == "12.5"


def test_des_bridges_target_canonical_variables_and_are_case_scoped():
    bridges = pd.read_csv("config/bridge_rules.csv").set_index("bridge_rule_id")
    assert bridges.loc["BR_DES_LEAD_TIME", "target_mrv_variable"] == "average_lead_time"
    assert bridges.loc["BR_DES_SERVICE_LEVEL", "target_mrv_variable"] == "service_level"
    assert "Cuba case" in bridges.loc["BR_DES_SERVICE_LEVEL", "case_validity"]
