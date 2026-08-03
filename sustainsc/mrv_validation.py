"""Schema-based validation for completed seven-column MRV datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MRV_COLUMNS = [
    "variable_name",
    "value",
    "unit",
    "timestamp",
    "scenario_code",
    "source_system",
    "comment",
]


class CompletionValidationError(ValueError):
    """Raised when completed MRV data are unsafe for KPI calculation."""


@dataclass(frozen=True)
class CompletedMRVValidation:
    is_valid: bool
    required_variable_count: int
    scenario_count: int
    report: pd.DataFrame


def load_common_variable_units(dictionary_path: str | Path) -> dict[str, str]:
    """Return the active common MRV contract from the dictionary."""
    dictionary = pd.read_csv(dictionary_path)
    common = dictionary[
        dictionary["common_upload_variable"].astype(str).str.strip().str.lower().isin(
            {"yes", "true", "1"}
        )
    ].copy()
    return dict(zip(common["variable_name"], common["canonical_unit"]))


def select_common_mrv(
    completed: pd.DataFrame, *, dictionary_path: str | Path
) -> pd.DataFrame:
    """Keep only common analytical rows; native/reference rows remain source evidence."""
    required = set(load_common_variable_units(dictionary_path))
    return completed[
        completed["variable_name"].astype(str).str.strip().isin(required)
    ].copy()


def canonicalize_common_mrv_units(
    completed: pd.DataFrame, *, dictionary_path: str | Path
) -> pd.DataFrame:
    """Apply dictionary-owned canonical unit labels to common MRV rows."""
    units = load_common_variable_units(dictionary_path)
    result = completed.copy()
    result["unit"] = result["variable_name"].map(units).fillna(result["unit"])
    return result


def validate_completed_mrv(
    completed: pd.DataFrame,
    *,
    dictionary_path: str | Path,
    raise_on_error: bool = True,
) -> CompletedMRVValidation:
    """Validate completeness, uniqueness, values and canonical units per scenario."""

    required_units = load_common_variable_units(dictionary_path)
    required = set(required_units)
    missing_columns = set(MRV_COLUMNS) - set(completed.columns)
    if missing_columns:
        raise CompletionValidationError(
            f"Completed MRV data are missing columns: {sorted(missing_columns)}"
        )

    findings: list[dict[str, object]] = []
    for scenario_code, scenario in completed.groupby("scenario_code", sort=True):
        names = scenario["variable_name"].astype(str)
        present = set(names)
        duplicate_names = sorted(names[names.duplicated(keep=False)].unique())
        missing = sorted(required - present)
        unknown = sorted(present - required)
        null_names = sorted(scenario.loc[scenario["value"].isna(), "variable_name"].astype(str))
        mismatches = sorted(
            row.variable_name
            for row in scenario.itertuples()
            if row.variable_name in required_units
            and str(row.unit).strip() != str(required_units[row.variable_name]).strip()
        )
        for kind, values in (
            ("missing_variable", missing),
            ("duplicate_variable", duplicate_names),
            ("unknown_variable", unknown),
            ("null_value", null_names),
            ("unit_mismatch", mismatches),
        ):
            for variable in values:
                findings.append(
                    {
                        "scenario_code": scenario_code,
                        "finding": kind,
                        "variable_name": variable,
                        "severity": "Critical",
                    }
                )
        if not any((missing, duplicate_names, unknown, null_names, mismatches)):
            findings.append(
                {
                    "scenario_code": scenario_code,
                    "finding": "complete",
                    "variable_name": "",
                    "severity": "Pass",
                }
            )

    report = pd.DataFrame(
        findings,
        columns=["scenario_code", "finding", "variable_name", "severity"],
    )
    result = CompletedMRVValidation(
        is_valid=not (report["severity"] == "Critical").any() if not report.empty else False,
        required_variable_count=len(required),
        scenario_count=int(completed["scenario_code"].nunique()),
        report=report,
    )
    if raise_on_error and not result.is_valid:
        failures = report[report["severity"] == "Critical"]
        detail = "; ".join(
            f"{row.scenario_code}:{row.finding}:{row.variable_name}"
            for row in failures.itertuples()
        )
        raise CompletionValidationError(f"Completed MRV validation failed: {detail}")
    return result
