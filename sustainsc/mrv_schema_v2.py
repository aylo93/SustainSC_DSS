"""Versioned schema and parser for SustainSCM MRV completion workbooks.

The workbook is a measurement-completion contract.  Its expected-value and
software-output sheets are never treated as production evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

SCHEMA_VERSION = "2.0"
PRODUCTION_SHEETS = {
    "scenarios": "01_SCENARIOS", "direct_inputs": "02_DIRECT_MRV_INPUT",
    "native_outputs": "03_NATIVE_OUTPUTS", "assumptions": "04_APPROVED_ASSUMPTIONS",
    "base_reference": "05_REFERENCE_BASE", "variable_dictionary": "06_MRV_DICTIONARY",
    "strategy_scope": "07_STRATEGY_SCOPE", "variable_overrides": "08_VARIABLE_OVERRIDES",
    "mrv_rules": "09_MRV_RULES", "bridge_rules": "10_BRIDGE_RULES",
    "factor_register": "16_FACTOR_REGISTER", "case_metadata": "18_CASE_METADATA",
}
OPTIONAL_SHEETS = ("11_EXPECTED_CASE_MRV",)
DOCUMENTATION_SHEETS = ("00_GUIDE", "17_SOURCE_RECONCILIATION", "19_REFERENCE_LISTS")
OUTPUT_SHEETS = ("12_COMPLETION_REVIEW", "13_SOFTWARE_UPLOAD", "14_QA_REPORT", "15_SCENARIO_SUMMARY")
LEGACY_REQUIRED_SHEETS = (
    "01_SCENARIOS", "02_DIRECT_MRV_INPUT", "03_NATIVE_OUTPUTS",
    "04_APPROVED_ASSUMPTIONS", "05_REFERENCE_BASE", "06_MRV_DICTIONARY",
    "07_STRATEGY_SCOPE", "08_VARIABLE_OVERRIDES", "09_MRV_RULES",
    "10_BRIDGE_RULES", "11_EXPECTED_CH7_MRV", "16_EMISSION_FACTORS",
)
UPLOAD_COLUMNS = ("variable_name", "value", "unit", "timestamp", "scenario_code", "source_system", "comment")


@dataclass(frozen=True)
class WorkbookSchemaInfo:
    version: str
    workbook_type: str
    required_sheets: tuple[str, ...]
    optional_sheets: tuple[str, ...]


@dataclass(frozen=True)
class CaseMetadata:
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass
class ParsedMRVWorkbook:
    schema: WorkbookSchemaInfo
    metadata: CaseMetadata
    scenarios: pd.DataFrame
    direct_inputs: pd.DataFrame
    native_outputs: pd.DataFrame
    assumptions: pd.DataFrame
    base_reference: pd.DataFrame
    variable_dictionary: pd.DataFrame
    strategy_scope: pd.DataFrame
    variable_overrides: pd.DataFrame
    mrv_rules: pd.DataFrame
    bridge_rules: pd.DataFrame
    factor_register: pd.DataFrame
    expected_case_mrv: pd.DataFrame | None
    warnings: tuple[str, ...] = ()


def _table(book: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(book, sheet_name=sheet, header=3)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.dropna(how="all").copy()


def _metadata_values(book: pd.ExcelFile) -> dict[str, Any]:
    frame = _table(book, "18_CASE_METADATA")
    if not {"field", "value"}.issubset(frame.columns):
        raise ValueError("18_CASE_METADATA must contain field and value columns.")
    populated = frame.dropna(subset=["field"])
    if populated["field"].astype(str).duplicated().any():
        raise ValueError("18_CASE_METADATA contains duplicate fields.")
    return {str(row.field).strip(): row.value for row in populated.itertuples()}


def detect_mrv_workbook_schema(source: str | Path | BinaryIO) -> WorkbookSchemaInfo:
    book = source if isinstance(source, pd.ExcelFile) else pd.ExcelFile(source)
    sheets = set(book.sheet_names)
    if "18_CASE_METADATA" in sheets:
        values = _metadata_values(book)
        raw_version = values.get("template_schema_version")
        version = str(raw_version).strip()
        if version.endswith(".0.0"):
            version = version[:-2]
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MRV workbook schema: {version or 'missing'}. "
                f"Supported versions: {SCHEMA_VERSION} and explicitly migrated legacy schemas."
            )
        missing = set(PRODUCTION_SHEETS.values()) - sheets
        if missing:
            raise ValueError(f"MRV v2 workbook is missing required sheets: {sorted(missing)}")
        return WorkbookSchemaInfo(SCHEMA_VERSION, "MRV_V2", tuple(PRODUCTION_SHEETS.values()), OPTIONAL_SHEETS)
    if set(LEGACY_REQUIRED_SHEETS).issubset(sheets):
        return WorkbookSchemaInfo("legacy", "LEGACY_MRV_ADAPTER", LEGACY_REQUIRED_SHEETS, ())
    raise ValueError(
        "MRV workbook has no 18_CASE_METADATA version marker and does not match a supported legacy schema."
    )


class MRVWorkbookV2Parser:
    def parse(self, source: str | Path | BinaryIO) -> ParsedMRVWorkbook:
        book = pd.ExcelFile(source)
        schema = detect_mrv_workbook_schema(book)
        if schema.workbook_type != "MRV_V2":
            raise ValueError("MRVWorkbookV2Parser only accepts native v2 workbooks.")
        metadata = _metadata_values(book)
        required_fields = {
            "template_schema_version", "case_id", "dataset_id", "dataset_name", "company",
            "site_or_network", "country", "reporting_period_start", "reporting_period_end",
            "functional_unit", "default_reference_scenario", "approval_status",
        }
        missing_fields = sorted(field for field in required_fields if pd.isna(metadata.get(field)) or str(metadata.get(field, "")).strip() == "")
        # The empty generic template is structurally valid but intentionally incomplete.
        warnings = tuple(["Incomplete case configuration: " + ", ".join(missing_fields)]) if missing_fields else ()
        frames = {name: _table(book, sheet) for name, sheet in PRODUCTION_SHEETS.items() if name != "case_metadata"}
        expected = _table(book, OPTIONAL_SHEETS[0]) if OPTIONAL_SHEETS[0] in book.sheet_names else None
        return ParsedMRVWorkbook(schema, CaseMetadata(metadata), expected_case_mrv=expected, warnings=warnings, **frames)


class LegacyMRVWorkbookAdapter:
    """Explicit compatibility adapter; it never claims legacy data are native v2."""
    def parse(self, source: str | Path | BinaryIO) -> ParsedMRVWorkbook:
        book = pd.ExcelFile(source)
        schema = detect_mrv_workbook_schema(book)
        if schema.workbook_type != "LEGACY_MRV_ADAPTER":
            raise ValueError("LegacyMRVWorkbookAdapter only accepts recognized legacy workbooks.")
        frames = {key: _table(book, sheet) for key, sheet in PRODUCTION_SHEETS.items()
                  if key not in {"factor_register", "case_metadata"}}
        scenarios = frames["scenarios"]
        reference = frames["base_reference"]
        reference_code = str(reference["scenario_code"].dropna().iloc[0]) if not reference.empty else ""
        metadata = CaseMetadata({
            "template_schema_version": "legacy", "case_id": "legacy-unresolved",
            "dataset_id": "legacy-unresolved", "dataset_name": "Legacy MRV import",
            "default_reference_scenario": reference_code,
        })
        old_factors = _table(book, "16_EMISSION_FACTORS")
        expected = _table(book, "11_EXPECTED_CH7_MRV")
        return ParsedMRVWorkbook(
            schema, metadata, factor_register=old_factors, expected_case_mrv=expected,
            warnings=("Legacy compatibility mode: case/dataset identity and factor semantics require scientific reconciliation.",),
            **frames,
        )


def parse_mrv_workbook(source: str | Path | BinaryIO) -> ParsedMRVWorkbook:
    schema = detect_mrv_workbook_schema(source)
    parser = MRVWorkbookV2Parser() if schema.workbook_type == "MRV_V2" else LegacyMRVWorkbookAdapter()
    return parser.parse(source)
