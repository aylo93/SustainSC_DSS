"""Batch runner for the SustainSCM MRV causal completion engine.

The workbook contains multiple scenario rows and scenario_code-keyed input tables.
Each scenario is passed independently to ScenarioCompletionEngine, and the outputs
are combined into a single seven-column MRV CSV for the existing KPI engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import zipfile

import pandas as pd

from scenario_completion_engine import ScenarioCompletionEngine, CompletionResult, UPLOAD_COLUMNS
from sustainsc.mrv_validation import validate_completed_mrv

BATCH_INPUT_SHEETS = {
    "scenarios": "01_SCENARIOS",
    "direct": "02_DIRECT_MRV_INPUT",
    "native": "03_NATIVE_OUTPUTS",
    "assumptions": "04_APPROVED_ASSUMPTIONS",
    "reference": "05_REFERENCE_BASE",
    "expected": "11_EXPECTED_CH7_MRV",
}

@dataclass
class BatchCompletionResult:
    scenario_results: dict[str, CompletionResult]
    completion_review: pd.DataFrame
    software_upload: pd.DataFrame
    qa_report: pd.DataFrame
    comparison_report: pd.DataFrame

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

    @staticmethod
    def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
        frame = pd.read_excel(path, sheet_name=sheet, header=3)
        frame.columns = [str(c).strip() for c in frame.columns]
        return frame.dropna(how="all").copy()

    def complete_batch_from_excel(
        self,
        workbook_path: str | Path,
        *,
        comparison_tolerance: float = 1e-6,
    ) -> BatchCompletionResult:
        workbook_path = Path(workbook_path)
        if not workbook_path.exists():
            raise FileNotFoundError(workbook_path)

        frames = {name: self._read_sheet(workbook_path, sheet) for name, sheet in BATCH_INPUT_SHEETS.items()}
        scenarios = frames["scenarios"]
        if "batch_enabled" in scenarios.columns:
            scenarios = scenarios[scenarios["batch_enabled"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"})]

        reference = frames["reference"]
        direct = frames["direct"]
        native = frames["native"]
        assumptions = frames["assumptions"]
        expected = frames["expected"]

        results: dict[str, CompletionResult] = {}
        review_frames: list[pd.DataFrame] = []
        upload_frames: list[pd.DataFrame] = []
        qa_frames: list[pd.DataFrame] = []

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

        review_all = pd.concat(review_frames, ignore_index=True) if review_frames else pd.DataFrame()
        upload_all = pd.concat(upload_frames, ignore_index=True) if upload_frames else pd.DataFrame(columns=UPLOAD_COLUMNS)
        qa_all = pd.concat(qa_frames, ignore_index=True) if qa_frames else pd.DataFrame()
        completed_validation = validate_completed_mrv(
            upload_all,
            dictionary_path=self.engine.config_dir / "mrv_dictionary.csv",
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

        expected_columns = expected.loc[:, ["scenario_code", "variable_name", "expected_value", "unit"]].rename(
            columns={"unit": "expected_unit"}
        )
        comparison = upload_all.merge(
            expected_columns,
            on=["scenario_code", "variable_name"],
            how="left",
        )
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

        if not qa_all.empty:
            qa_all = qa_all.drop_duplicates(
                subset=["scenario_code", "affected_variable", "check_id"], keep="first"
            ).reset_index(drop=True)

        return BatchCompletionResult(
            scenario_results=results,
            completion_review=review_all,
            software_upload=upload_all,
            qa_report=qa_all,
            comparison_report=comparison,
        )
