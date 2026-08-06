"""Batch runner for the SustainSCM MRV causal completion engine.

The workbook contains multiple scenario rows and scenario_code-keyed input tables.
Each scenario is passed independently to ScenarioCompletionEngine, and the outputs
are combined into a single seven-column MRV CSV for the existing KPI engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib
import zipfile

import pandas as pd

from scenario_completion_engine import ScenarioCompletionEngine, CompletionResult, UPLOAD_COLUMNS
from sustainsc.mrv_schema_v2 import ParsedMRVWorkbook, parse_mrv_workbook
from sustainsc.mrv_validation import validate_completed_mrv

@dataclass
class BatchCompletionResult:
    scenario_results: dict[str, CompletionResult]
    completion_review: pd.DataFrame
    software_upload: pd.DataFrame
    qa_report: pd.DataFrame
    comparison_report: pd.DataFrame
    parsed_workbook: ParsedMRVWorkbook | None = None
    source_filename: str | None = None
    structural_summary: dict[str, object] | None = None
    evidence_outcomes: pd.DataFrame | None = None
    failure_diagnostics: pd.DataFrame | None = None
    l3_permission_diagnostics: pd.DataFrame | None = None
    rule_execution_trace: pd.DataFrame | None = None
    workbook_sha256: str | None = None
    workbook_size: int | None = None
    parser_version: str = "2"
    completion_engine_version: str = "4"

    @property
    def production_qa_report(self) -> pd.DataFrame:
        return self.qa_report

    @property
    def regression_comparison_report(self) -> pd.DataFrame:
        return self.comparison_report

    @property
    def has_critical_failures(self) -> bool:
        if self.qa_report.empty:
            return False
        return bool(((self.qa_report["severity"] == "Critical") & (self.qa_report["status"] == "FAIL")).any())

    @property
    def metadata_resolved(self) -> bool:
        if not self.parsed_workbook:
            return False
        metadata = self.parsed_workbook.metadata
        return all(
            str(metadata.get(field, "")).strip() not in {"", "legacy-unresolved"}
            for field in ("case_id", "dataset_id")
        )

    @property
    def can_commit(self) -> bool:
        return bool(
            self.metadata_resolved
            and self.structural_summary
            and self.structural_summary.get("complete")
            and not self.has_critical_failures
        )

    def export_combined_csv(self, path: str | Path, *, force: bool = False) -> Path:
        if self.has_critical_failures and not force:
            raise ValueError("Critical QA failures block combined CSV export.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.software_upload.loc[:, UPLOAD_COLUMNS].to_csv(path, index=False)
        return path

    def export_scenario_csv_zip(self, path: str | Path, *, force: bool = False) -> Path:
        if self.has_critical_failures and not force:
            raise ValueError("Critical QA failures block scenario CSV export.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for scenario_code, result in self.scenario_results.items():
                content = result.software_upload.loc[:, UPLOAD_COLUMNS].to_csv(index=False)
                archive.writestr(f"{scenario_code}_completed_mrv.csv", content)
        return path

class BatchScenarioCompletionEngine:
    def __init__(self, config_dir: str | Path, *, strict_approval: bool = True) -> None:
        self.engine = ScenarioCompletionEngine(config_dir, strict_approval=strict_approval)

    def complete_batch_from_excel(
        self,
        workbook_path: str | Path,
        *,
        comparison_tolerance: float = 1e-6,
    ) -> BatchCompletionResult:
        workbook_path = Path(workbook_path)
        if not workbook_path.exists():
            raise FileNotFoundError(workbook_path)

        parsed = parse_mrv_workbook(workbook_path)
        self.engine = ScenarioCompletionEngine(
            self.engine.config_dir,
            strict_approval=self.engine.strict_approval,
            config_frames={
                "dictionary": parsed.variable_dictionary,
                "scope": parsed.strategy_scope,
                "overrides": parsed.variable_overrides,
                "rules": parsed.mrv_rules,
                "bridges": parsed.bridge_rules,
                "factor_register": parsed.factor_register,
                "default_factor_set_id": parsed.metadata.get("default_emission_factor_set_id"),
            },
        )
        scenarios = parsed.scenarios
        if "batch_enabled" in scenarios.columns:
            scenarios = scenarios[scenarios["batch_enabled"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})]

        reference = parsed.base_reference
        direct = parsed.direct_inputs
        native = parsed.native_outputs
        assumptions = parsed.assumptions
        expected = parsed.expected_case_mrv if parsed.expected_case_mrv is not None else pd.DataFrame()

        results: dict[str, CompletionResult] = {}
        review_frames: list[pd.DataFrame] = []
        upload_frames: list[pd.DataFrame] = []
        qa_frames: list[pd.DataFrame] = []
        l3_frames: list[pd.DataFrame] = []
        trace_frames: list[pd.DataFrame] = []

        for _, scenario in scenarios.iterrows():
            scenario_code = str(scenario["scenario_code"]).strip()
            result = self.engine.complete(
                scenario=scenario,
                direct_inputs=direct[direct["scenario_code"].astype(str) == scenario_code].drop(columns=["scenario_code"], errors="ignore"),
                native_outputs=native[native["scenario_code"].astype(str) == scenario_code].drop(columns=["scenario_code"], errors="ignore"),
                assumptions=assumptions[assumptions["scenario_code"].astype(str) == scenario_code].drop(columns=["scenario_code"], errors="ignore"),
                reference=reference,
            )
            results[scenario_code] = result
            review = result.completion_review.copy()
            review["run_id"] = result.run_id
            review_frames.append(review)
            upload_frames.append(result.software_upload)
            qa_frames.append(result.qa_report)
            l3_frames.append(result.l3_permission_diagnostics)
            trace_frames.append(result.rule_execution_trace)

        review_all = pd.concat(review_frames, ignore_index=True) if review_frames else pd.DataFrame()
        upload_all = pd.concat(upload_frames, ignore_index=True) if upload_frames else pd.DataFrame(columns=UPLOAD_COLUMNS)
        qa_all = pd.concat(qa_frames, ignore_index=True) if qa_frames else pd.DataFrame()
        l3_all = pd.concat(l3_frames, ignore_index=True) if l3_frames else pd.DataFrame()
        trace_all = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
        if scenarios.empty:
            qa_all = pd.DataFrame([{
                "check_id": "QA_SCENARIO_CONFIGURATION", "severity": "Critical", "status": "FAIL",
                "affected_variable": "", "message": "No enabled scenarios are configured.",
                "required_action": "Configure and enable at least one scenario before completion.",
                "scenario_code": "", "run_id": "batch_configuration",
            }])
        completed_validation = validate_completed_mrv(
            upload_all,
            dictionary=parsed.variable_dictionary,
            raise_on_error=False,
        )
        validation_failures = completed_validation.report[
            completed_validation.report["severity"] == "Critical"
        ]
        if not validation_failures.empty:
            completion_qa = pd.DataFrame(
                {
                    "check_id": "QA_COMPLETED_MRV_SCHEMA",
                    "severity": "Critical",
                    "status": "FAIL",
                    "affected_variable": validation_failures["variable_name"],
                    "message": validation_failures.apply(
                        lambda row: (
                            f"{row['finding']} for scenario {row['scenario_code']}: "
                            f"{row['variable_name']}"
                        ),
                        axis=1,
                    ),
                    "required_action": (
                        "Resolve the completed MRV schema finding before KPI calculation."
                    ),
                    "scenario_code": validation_failures["scenario_code"],
                    "run_id": "batch_schema_validation",
                }
            )
            qa_all = pd.concat([qa_all, completion_qa], ignore_index=True)

        expected_columns = ["scenario_code", "variable_name", "expected_value"]
        expected_values = expected.loc[:, expected_columns] if set(expected_columns).issubset(expected.columns) else pd.DataFrame(columns=expected_columns)
        comparison = upload_all.merge(expected_values, on=["scenario_code", "variable_name"], how="left")
        comparison["expected_value"] = pd.to_numeric(comparison["expected_value"], errors="coerce")
        comparison["absolute_difference"] = (comparison["value"] - comparison["expected_value"]).abs()
        scale = comparison[["value", "expected_value"]].abs().max(axis=1).clip(lower=1.0)
        unit_absolute_tolerance = comparison["unit"].map(
            {"%": 1e-4, "EUR": 1e-2, "tCO2e": 1e-3, "index": 1e-4}
        ).fillna(comparison_tolerance)
        comparison["tolerance"] = unit_absolute_tolerance.combine(
            scale * 1e-9, max
        )
        comparison["relative_difference"] = comparison["absolute_difference"] / scale
        comparison["within_tolerance"] = comparison["absolute_difference"] <= comparison["tolerance"]
        comparison["comparison_status"] = "UNRESOLVED_DIFFERENCE"
        comparison.loc[comparison["expected_value"].isna(), "comparison_status"] = "MISSING_EXPECTED_VALUE"
        comparison.loc[comparison["within_tolerance"], "comparison_status"] = "MATCH"
        rounding = (
            ~comparison["within_tolerance"]
            & comparison["expected_value"].notna()
            & (comparison["relative_difference"] <= 1e-6)
        )
        comparison.loc[rounding, "comparison_status"] = "ROUNDING_ONLY"
        comparison["reason"] = comparison["comparison_status"].map({
            "MATCH": "Current result agrees with the validation baseline within the unit-aware tolerance.",
            "ROUNDING_ONLY": "Difference is limited to numeric representation or display precision.",
            "MISSING_EXPECTED_VALUE": "No historical comparison value is available; production QA is unaffected.",
            "UNRESOLVED_DIFFERENCE": "Historical value differs and requires regression review; production QA is evaluated separately.",
        })
        comparison = comparison.loc[:, [
            "scenario_code", "variable_name", "value", "expected_value",
            "absolute_difference", "relative_difference", "tolerance",
            "within_tolerance", "comparison_status", "reason", "source_system", "comment"
        ]]
        mismatch_rows = comparison[
            comparison["expected_value"].notna()
            & ~comparison["comparison_status"].isin(["MATCH", "ROUNDING_ONLY"])
        ]

        if not qa_all.empty:
            qa_all = qa_all.drop_duplicates(
                subset=["scenario_code", "affected_variable", "check_id"], keep="first"
            ).reset_index(drop=True)

        if "comparison_mode" in expected.columns and not mismatch_rows.empty:
            strict_keys = expected[
                expected["comparison_mode"].astype(str).str.upper() == "STRICT_REGRESSION"
            ][["scenario_code", "variable_name"]]
            strict_mismatches = mismatch_rows.merge(strict_keys, on=["scenario_code", "variable_name"])
            if not strict_mismatches.empty:
                strict_qa = pd.DataFrame({
                    "check_id": "QA_STRICT_REGRESSION", "severity": "Critical", "status": "FAIL",
                    "affected_variable": strict_mismatches["variable_name"],
                    "message": "Completed value differs from a STRICT_REGRESSION expected value.",
                    "required_action": "Reconcile the method or explicitly change the regression mode.",
                    "scenario_code": strict_mismatches["scenario_code"], "run_id": "batch_comparison",
                })
                qa_all = pd.concat([qa_all, strict_qa], ignore_index=True)

        common = parsed.variable_dictionary[
            parsed.variable_dictionary["common_upload_variable"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})
        ]
        required_names = set(common["variable_name"].astype(str))
        duplicate_count = int(upload_all[["scenario_code", "variable_name"]].duplicated().sum())
        null_count = int(upload_all["value"].isna().sum())
        finite_count = int(pd.to_numeric(upload_all["value"], errors="coerce").map(lambda value: pd.notna(value) and abs(float(value)) != float("inf")).sum())
        unknown_count = int((~upload_all["variable_name"].astype(str).isin(required_names)).sum()) if not upload_all.empty else 0
        expected_rows = len(scenarios) * len(required_names)
        structural_summary = {
            "scenario_count": len(scenarios), "required_variable_count": len(required_names),
            "final_row_count": len(upload_all), "expected_row_count": expected_rows,
            "duplicate_pairs": duplicate_count, "null_values": null_count,
            "non_finite_values": len(upload_all) - finite_count, "unknown_variables": unknown_count,
            "rule_level_total": len(review_all),
        }
        structural_summary["complete"] = bool(
            expected_rows > 0 and len(upload_all) == expected_rows and duplicate_count == 0
            and null_count == 0 and finite_count == len(upload_all) and unknown_count == 0
            and len(review_all) == len(upload_all)
        )

        evidence_rows = []
        for scenario_code, scenario_result in results.items():
            selected = set(
                scenario_result.completion_review.loc[
                    scenario_result.completion_review["rule_level"] == "L1", "variable_name"
                ].astype(str)
            )
            rejected = {
                str(row.variable_name): str(row.reason)
                for row in scenario_result.rejected_inputs.itertuples()
                if row.input_type == "direct"
            }
            source = direct[direct["scenario_code"].astype(str) == scenario_code]
            for row in source.itertuples():
                variable = str(row.variable_name)
                evidence_rows.append({
                    "scenario_code": scenario_code, "variable_name": variable,
                    "evidence_type": getattr(row, "evidence_type", ""),
                    "normalized_evidence_class": getattr(row, "normalized_evidence_class", ""),
                    "outcome": "selected_as_L1" if variable in selected else rejected.get(variable, "superseded_or_audit_only"),
                })
        evidence_outcomes = pd.DataFrame(evidence_rows)

        review_lookup = review_all.set_index(["scenario_code", "variable_name"]) if not review_all.empty else pd.DataFrame()
        diagnostic_rows = []
        failures = qa_all[(qa_all["status"].isin(["FAIL", "WARN"]))] if not qa_all.empty else qa_all
        for number, row in enumerate(failures.itertuples(), start=1):
            key = (str(row.scenario_code), str(row.affected_variable))
            review = review_lookup.loc[key] if not review_all.empty and key in review_lookup.index else {}
            diagnostic_rows.append({
                "failure_id": f"MRV-{number:04d}",
                "qa_domain": "production_qa",
                "check_id": row.check_id, "scenario_code": row.scenario_code,
                "variable_name": row.affected_variable,
                "rule_level": review.get("rule_level", "") if hasattr(review, "get") else "",
                "rule_id": review.get("rule_id", "") if hasattr(review, "get") else "",
                "severity": row.severity,
                "blocking": row.severity == "Critical" and row.status == "FAIL",
                "actual_value": review.get("completed_value", None) if hasattr(review, "get") else None,
                "expected_value": None,
                "unit": review.get("unit", "") if hasattr(review, "get") else "",
                "source_system": review.get("source_module", "") if hasattr(review, "get") else "",
                "message": row.message, "required_action": row.required_action,
            })
        failure_diagnostics = pd.DataFrame(diagnostic_rows)

        return BatchCompletionResult(
            scenario_results=results,
            completion_review=review_all,
            software_upload=upload_all,
            qa_report=qa_all,
            comparison_report=comparison,
            parsed_workbook=parsed,
            source_filename=workbook_path.name,
            structural_summary=structural_summary,
            evidence_outcomes=evidence_outcomes,
            failure_diagnostics=failure_diagnostics,
            l3_permission_diagnostics=l3_all,
            rule_execution_trace=trace_all,
            workbook_sha256=hashlib.sha256(workbook_path.read_bytes()).hexdigest(),
            workbook_size=workbook_path.stat().st_size,
        )
