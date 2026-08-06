from pathlib import Path

import pandas as pd
import pytest

from batch_completion_engine import BatchScenarioCompletionEngine
from scenario_completion_engine import UPLOAD_COLUMNS, _expand_rule_expression
from sustainsc.mrv_schema_v2 import detect_mrv_workbook_schema, parse_mrv_workbook


FIXTURES = Path("tests/fixtures/mrv_v2")
V2 = FIXTURES / "SustainSCM_MRV_Causal_Completion_Template_v2_0.xlsx"
LEGACY = FIXTURES / "SustainSCM_Cuba_Batch_MRV_Input_FILLED_MILP_CORRECTED.xlsx"


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


def test_recognized_legacy_workbook_uses_explicit_adapter():
    parsed = parse_mrv_workbook(LEGACY)
    assert parsed.schema.workbook_type == "LEGACY_MRV_ADAPTER"
    assert parsed.schema.version == "legacy"
    assert parsed.warnings


def test_legacy_regression_completes_common_measurements():
    result = BatchScenarioCompletionEngine("config").complete_batch_from_excel(LEGACY)
    assert len(result.scenario_results) == 24
    assert result.software_upload.shape == (2568, 7)
    assert list(result.software_upload.columns) == UPLOAD_COLUMNS
    assert not result.software_upload[["scenario_code", "variable_name"]].duplicated().any()
    assert result.software_upload["value"].map(pd.notna).all()


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
    ("L6", {"L6"}),
])
def test_rule_level_parser_is_structured(expression, expected):
    assert _expand_rule_expression(expression) == expected
