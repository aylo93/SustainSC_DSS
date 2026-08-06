"""Versioned schema and parser for SustainSCM MRV completion workbooks.

The workbook is a measurement-completion contract.  Its expected-value and
software-output sheets are never treated as production evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
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
    detected_from: str = ""
    case_id: str | None = None
    dataset_id: str | None = None
    migration_required: bool = False
    validation_messages: tuple[str, ...] = ()

    @property
    def schema_version(self) -> str:
        return self.version

    @property
    def schema_family(self) -> str:
        return self.workbook_type


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
    migration_adapter: str | None = None


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


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


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
        return WorkbookSchemaInfo(
            SCHEMA_VERSION, "MRV_V2", tuple(PRODUCTION_SHEETS.values()), OPTIONAL_SHEETS,
            detected_from="18_CASE_METADATA.template_schema_version",
            case_id=_optional_text(values.get("case_id")),
            dataset_id=_optional_text(values.get("dataset_id")),
            validation_messages=("Matched current workbook metadata and required sheet signature.",),
        )
    if set(LEGACY_REQUIRED_SHEETS).issubset(sheets):
        return WorkbookSchemaInfo(
            "legacy", "LEGACY_CASE_WORKBOOK", LEGACY_REQUIRED_SHEETS, (),
            detected_from="missing 18_CASE_METADATA; matched explicit legacy sheet signature",
            migration_required=True,
            validation_messages=(
                "11_EXPECTED_CH7_MRV maps to 11_EXPECTED_CASE_MRV.",
                "16_EMISSION_FACTORS maps to 16_FACTOR_REGISTER.",
            ),
        )
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
        if schema.workbook_type != "LEGACY_CASE_WORKBOOK":
            raise ValueError("LegacyMRVWorkbookAdapter only accepts recognized legacy workbooks.")
        frames = {key: _table(book, sheet) for key, sheet in PRODUCTION_SHEETS.items()
                  if key not in {"factor_register", "case_metadata"}}
        scenarios = frames["scenarios"]
        reference = frames["base_reference"]
        reference_code = str(reference["scenario_code"].dropna().iloc[0]) if not reference.empty else ""
        metadata = CaseMetadata(_migrated_metadata(source, scenarios, reference_code))
        frames["direct_inputs"] = _map_legacy_evidence(frames["direct_inputs"])
        old_factors = _migrate_legacy_factors(
            _table(book, "16_EMISSION_FACTORS"), scenarios, metadata
        )
        expected = _table(book, "11_EXPECTED_CH7_MRV")
        return ParsedMRVWorkbook(
            schema, metadata, factor_register=old_factors, expected_case_mrv=expected,
            warnings=(
                "Legacy workbook migrated deterministically to the current internal contract.",
                "Deprecated evidence labels were mapped explicitly and remain auditable.",
            ),
            migration_adapter="legacy_case_to_current",
            **frames,
        )


def parse_mrv_workbook(source: str | Path | BinaryIO) -> ParsedMRVWorkbook:
    schema = detect_mrv_workbook_schema(source)
    parser = MRVWorkbookV2Parser() if schema.workbook_type == "MRV_V2" else LegacyMRVWorkbookAdapter()
    return parser.parse(source)


LEGACY_EVIDENCE_MAP = {
    "Direct model output": "DIRECT_MODEL_OUTPUT",
    "Direct SD model output": "DIRECT_MODEL_OUTPUT",
    "Direct DES output / common-MRV mapping": "DIRECT_MODEL_OUTPUT",
    "Derived from documented SD outputs": "DERIVED_FROM_MODEL_OUTPUT",
    "Derived/scaled MRV compatibility value": "DERIVED_FROM_MODEL_OUTPUT",
    "Approved case-specific L4 compatibility mapping": "CASE_SPECIFIC_BRIDGE",
    "L4 SD-to-MRV translation": "CASE_SPECIFIC_BRIDGE",
    "L4 SD-to-MRV bridge": "CASE_SPECIFIC_BRIDGE",
    "L4 SD-to-DES input handoff": "CASE_SPECIFIC_BRIDGE",
    "Temporary BASE-coverage retention": "BASE_RETENTION",
    "Retain BASE — not reported by SD": "BASE_RETENTION",
    "Retain BASE - not reported by SD": "BASE_RETENTION",
}


def _map_legacy_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    labels = result.get("evidence_type", pd.Series(index=result.index, dtype=object)).fillna("").astype(str).str.strip()
    mapped = labels.map(LEGACY_EVIDENCE_MAP)
    unsupported = sorted(labels[(labels != "") & mapped.isna()].unique())
    if unsupported:
        raise ValueError(f"Unsupported legacy evidence labels require explicit migration mapping: {unsupported}")
    result["legacy_evidence_type"] = labels
    result["normalized_evidence_class"] = mapped
    # Migration never expands causal permissions from observed values. Rows
    # outside scope remain source evidence and BASE/derived completion wins.
    result["migration_disposition"] = "AUDIT_IF_OUTSIDE_CAUSAL_SCOPE"
    return result


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")


def _source_digest(source: str | Path | BinaryIO) -> str:
    if isinstance(source, (str, Path)):
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    position = source.tell() if hasattr(source, "tell") else None
    payload = source.read()
    if position is not None:
        source.seek(position)
    return hashlib.sha256(payload).hexdigest()


def _first(frame: pd.DataFrame, column: str, default: str = "") -> str:
    if column not in frame or frame[column].dropna().empty:
        return default
    return str(frame[column].dropna().iloc[0]).strip()


def _migrated_metadata(
    source: str | Path | BinaryIO, scenarios: pd.DataFrame, reference_code: str
) -> dict[str, Any]:
    company = _first(scenarios, "company")
    site = _first(scenarios, "site")
    country = _first(scenarios, "country")
    identity = "-".join(filter(None, (_slug(company), _slug(site), _slug(country)))) or "migrated-case"
    digest = _source_digest(source)[:16]
    timestamps = pd.to_datetime(scenarios.get("evaluation_timestamp"), errors="coerce").dropna()
    emission_sets = scenarios.get("emission_factor_set_id", pd.Series(dtype=object)).dropna().astype(str).str.strip().unique()
    cost_sets = scenarios.get("cost_factor_set_id", pd.Series(dtype=object)).dropna().astype(str).str.strip().unique()
    return {
        "template_schema_version": SCHEMA_VERSION,
        "source_schema_version": "legacy",
        "case_id": identity,
        "dataset_id": f"{identity}-{digest}",
        "dataset_name": f"{company or identity} migrated MRV dataset",
        "company": company,
        "site_or_network": site,
        "country": country,
        "reporting_period_start": timestamps.min() if not timestamps.empty else None,
        "reporting_period_end": timestamps.max() if not timestamps.empty else None,
        "functional_unit": _first(scenarios, "functional_unit"),
        "default_reference_scenario": reference_code,
        "default_emission_factor_set_id": emission_sets[0] if len(emission_sets) == 1 else "",
        "default_cost_factor_set_id": cost_sets[0] if len(cost_sets) == 1 else "",
        "approval_status": "Approved" if scenarios.get("approval_status", pd.Series(dtype=object)).astype(str).eq("Approved").all() else "Migration review required",
        "dpp_workbook_required": "Configuration required",
    }


def _migrate_legacy_factors(
    old: pd.DataFrame, scenarios: pd.DataFrame, metadata: CaseMetadata
) -> pd.DataFrame:
    result = pd.DataFrame()
    result["factor_set_id"] = old["factor_code"].map(
        lambda _: metadata.get("default_emission_factor_set_id") or "migrated-factor-set"
    )
    result["factor_code"] = old["factor_code"]
    result["factor_type"] = "EMISSION"
    result["value"] = old["value"]
    result["unit"] = old["unit"]
    result["analytical_role"] = old["analytical_role"]
    known_activity = {
        "EF_ELECTRICITY_CASE": "electricity_kwh",
        "EF_GRID_LOCATION_REFERENCE": "electricity_kwh",
        "EF_DIESEL": "diesel_kwh",
    }
    result["scope"] = old["factor_code"].map(known_activity).fillna("configuration_required")
    result["valid_from"] = metadata.get("reporting_period_start")
    result["valid_to"] = metadata.get("reporting_period_end")
    result["source"] = "Migrated 16_EMISSION_FACTORS"
    result["approval_status"] = "Approved"
    result["notes"] = "Deterministically migrated from the supported legacy factor sheet."
    return result
