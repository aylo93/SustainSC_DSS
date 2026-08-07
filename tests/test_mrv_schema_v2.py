from pathlib import Path
import hashlib

import pandas as pd
import pytest

from batch_completion_engine import BatchScenarioCompletionEngine
from scenario_completion_engine import UPLOAD_COLUMNS, _expand_rule_expression
from sustainsc.mrv_schema_v2 import detect_mrv_workbook_schema, parse_mrv_workbook


FIXTURES = Path("tests/fixtures/mrv_final")
V2 = FIXTURES / "SustainSCM_MRV_Causal_Completion_Template_FINAL_BOUNDARY_RECONCILED.xlsx"
FINAL_CUBA = FIXTURES / "SustainSCM_Cuba_MRV_Scenario_Completion_FINAL_BOUNDARY_RECONCILED.xlsx"


def test_v2_is_detected_from_metadata_and_empty_template_is_structurally_valid():
    parsed = parse_mrv_workbook(V2)
    assert parsed.schema.version == "2.0"
    assert parsed.schema.workbook_type == "MRV_V2"
    assert parsed.scenarios.empty
    assert len(parsed.variable_dictionary) == 107
    assert parsed.expected_case_mrv is not None and parsed.expected_case_mrv.empty
    assert parsed.bridge_rules["rule_status"].astype(str).str.upper().eq("ACTIVE").sum() == 0


def test_filename_does_not_select_v2(tmp_path):
    fake = tmp_path / "SustainSCM_MRV_Causal_Completion_Template_v2_0.xlsx"
    with pd.ExcelWriter(fake) as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, index=False, sheet_name="01_SCENARIOS")
    with pytest.raises(ValueError, match="no 18_CASE_METADATA"):
        detect_mrv_workbook_schema(fake)


def test_final_cuba_is_current_and_never_uses_legacy_adapter():
    parsed = parse_mrv_workbook(FINAL_CUBA)
    result = BatchScenarioCompletionEngine("config").complete_batch_from_excel(FINAL_CUBA)
    assert parsed.schema.workbook_type == "MRV_V2"
    assert parsed.schema.version == "2.0"
    assert not parsed.schema.migration_required
    assert parsed.migration_adapter is None
    assert parsed.metadata.get("case_id") == "CUBA_HOLGUIN_AGGREGATES"
    assert parsed.metadata.get("dataset_id") == "CUBA_HOLGUIN_SCENARIOS_FINAL"
    assert result.workbook_sha256 == hashlib.sha256(FINAL_CUBA.read_bytes()).hexdigest()
    assert (len(parsed.scenarios), len(parsed.variable_dictionary)) == (24, 107)
    assert (len(parsed.direct_inputs), len(parsed.native_outputs), len(parsed.assumptions)) == (112, 182, 26)
    assert result.can_commit
    assert result.structural_summary == {
        "scenario_count": 24, "required_variable_count": 107,
        "final_row_count": 2568, "expected_row_count": 2568,
        "duplicate_pairs": 0, "null_values": 0, "non_finite_values": 0,
        "unknown_variables": 0, "rule_level_total": 2568, "complete": True,
    }
    assert not result.qa_report["check_id"].eq("QA_STRICT_REGRESSION").any()
    assert not result.has_critical_failures


def test_factor_rule_and_maintenance_override_are_synchronized_between_templates():
    cuba = parse_mrv_workbook(FINAL_CUBA)
    generic = parse_mrv_workbook(V2)
    for parsed, expected_status in ((cuba, "ACTIVE"), (generic, "CONFIG_REQUIRED")):
        rule = parsed.mrv_rules[parsed.mrv_rules["rule_id"] == "MRV_R_GHG_S1S2_FACTORS"].iloc[0]
        assert rule["operation"] == "GHG_FROM_ENERGY_FACTORS"
        assert rule["rule_status"] == expected_status
        override = parsed.variable_overrides[
            (parsed.variable_overrides["strategy_code"] == "LOGISTICS_REDESIGN")
            & (parsed.variable_overrides["variable_name"] == "maintenance_cost_eur")
        ].iloc[0]
        assert override["permitted_rules"] == "L1,L5,L6"
        assert int(override["priority"]) >= 100
        electricity = parsed.variable_overrides[
            (parsed.variable_overrides["strategy_code"] == "LOGISTICS_REDESIGN")
            & (parsed.variable_overrides["variable_name"] == "electricity_kwh")
        ].iloc[0]
        assert "L3" in electricity["permitted_rules"]
        if parsed is cuba:
            assert str(electricity["active"]).lower() in {"yes", "true", "1"}

    cuba_transport = cuba.mrv_rules[cuba.mrv_rules.rule_id.eq("CUBA_R060")].iloc[0]
    generic_transport = generic.mrv_rules[
        generic.mrv_rules.rule_id.eq("MRV_R_TRANSPORT_GHG_TKM")
    ].iloc[0]
    assert cuba_transport.operation == generic_transport.operation == "MULTIPLY_FACTOR"
    assert cuba_transport.rule_status == "ACTIVE"
    assert generic_transport.rule_status == "CONFIG_REQUIRED"
    factor = cuba.factor_register[cuba.factor_register.factor_code.eq("TRANSPORT_GHG_PER_TKM")].iloc[0]
    assert factor.value == pytest.approx(1.6492893067870172e-05)
    assert factor.unit == "tCO2e/tkm"


def test_empty_v2_template_reports_configuration_failure_not_parser_failure():
    result = BatchScenarioCompletionEngine("config").complete_batch_from_excel(V2)
    assert result.scenario_results == {}
    assert result.has_critical_failures
    assert "QA_SCENARIO_CONFIGURATION" in set(result.qa_report["check_id"])


def test_strict_comparison_path_does_not_reference_undefined_state():
    result = BatchScenarioCompletionEngine("config").complete_batch_from_excel(V2)
    assert result.comparison_report.empty


@pytest.mark.parametrize(("expression", "expected"), [
    ("L1-L5", {"L1", "L2", "L3", "L4", "L5"}),
    ("L1, L4, L6", {"L1", "L4", "L6"}),
    ("L1/L2", {"L1", "L2"}),
    ("L6", {"L6"}),
])
def test_rule_level_parser_is_structured(expression, expected):
    assert _expand_rule_expression(expression) == expected


def test_malformed_rule_level_expression_is_rejected():
    with pytest.raises(ValueError, match="Unsupported completion-level expression"):
        _expand_rule_expression("L1 and L2")


def test_normal_user_interface_hides_internal_version_label():
    source = Path("scenario_completion_page.py").read_text(encoding="utf-8")
    assert '"MRV Scenario Workbook v2"' not in source
    assert '"MRV Scenario Workbook"' in source
    assert '"Import diagnostics"' in source
    assert "completion-5:rules-5:transport-boundary-1" in source
    assert "checksum" in source and "normalization-2" in source
