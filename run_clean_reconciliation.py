"""Run the authoritative MRV-to-MCDA regression in a fresh database."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from batch_completion_engine import BatchScenarioCompletionEngine
from generate_final_reconciliation import export_active_results
from load_example_data import load_kpis
from sustainsc.config import Base, SessionLocal, engine
from sustainsc.dataset_scope import activate_import_run, assert_scenario_integrity, utc_now_naive
from sustainsc.kpi_engine import run_full_pipeline
from sustainsc.models import EmissionFactor, ImportRun, ImportRunScenario, Measurement, Scenario


def _commit_completion(result) -> int:
    frame = result.software_upload.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    metadata = result.parsed_workbook.metadata
    with SessionLocal() as session:
        load_kpis(session)
        session.query(EmissionFactor).delete()
        for factor in result.parsed_workbook.factor_register.itertuples(index=False):
            if str(getattr(factor, "approval_status", "")).strip().lower() != "approved":
                continue
            session.add(EmissionFactor(
                code=str(factor.factor_code), name=str(factor.factor_code),
                activity_type=str(getattr(factor, "factor_type", "emission")),
                unit=str(factor.unit), value=float(factor.value),
                valid_from=pd.to_datetime(factor.valid_from, errors="coerce").to_pydatetime(),
                valid_to=pd.to_datetime(factor.valid_to, errors="coerce").to_pydatetime(),
                source=str(factor.source), analytical_role=str(factor.analytical_role),
                factor_set_id=str(factor.factor_set_id),
            ))
        run = ImportRun(
            dataset_name=str(metadata.get("dataset_name")),
            case_id=str(metadata.get("case_id")),
            dataset_id=str(metadata.get("dataset_id")),
            schema_version=str(metadata.get("template_schema_version")),
            factor_set_id=str(metadata.get("default_emission_factor_set_id") or "") or None,
            source_filename=result.source_filename,
            import_timestamp=utc_now_naive(), status="importing",
            reference_scenario_code=str(metadata.get("default_reference_scenario")),
            scenario_count=0, measurement_count=0, is_active=False,
        )
        session.add(run)
        session.flush()
        scenario_ids: dict[str, int] = {}
        for code in sorted(frame["scenario_code"].unique()):
            scenario = session.query(Scenario).filter(Scenario.code == code).one_or_none()
            if scenario is None:
                scenario = Scenario(code=code, name=code, description="Clean MRV regression import")
                session.add(scenario)
                session.flush()
            scenario_ids[code] = scenario.id
            session.add(ImportRunScenario(import_run_id=run.id, scenario_id=scenario.id))
        session.flush()
        for row in frame.itertuples():
            session.add(Measurement(
                scenario_id=scenario_ids[row.scenario_code], import_run_id=run.id,
                variable_name=row.variable_name, value=float(row.value), unit=row.unit,
                timestamp=row.timestamp.to_pydatetime(), source_system=row.source_system,
                comment=row.comment,
            ))
        run.scenario_count = len(scenario_ids)
        run.measurement_count = len(frame)
        session.flush()
        assert_scenario_integrity(session, run.id, sorted(scenario_ids))
        activate_import_run(session, run)
        session.commit()
        return int(run.id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("generated/clean_reconciliation"))
    parser.add_argument("--expected-sha256")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    result = BatchScenarioCompletionEngine("config").complete_batch_from_excel(args.workbook)
    if args.expected_sha256 and result.workbook_sha256.lower() != args.expected_sha256.lower():
        raise RuntimeError(
            f"Workbook checksum mismatch: expected {args.expected_sha256}, got {result.workbook_sha256}"
        )
    if result.parsed_workbook.schema.migration_required or not result.can_commit:
        raise RuntimeError("Current-schema completion is not eligible for commit.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.completion_review.to_csv(args.output_dir / "completion_review.csv", index=False)
    result.qa_report.to_csv(args.output_dir / "qa_report.csv", index=False)
    result.l3_permission_diagnostics.to_csv(args.output_dir / "l3_permission_diagnostics.csv", index=False)
    result.rule_execution_trace.to_csv(args.output_dir / "rule_execution_trace.csv", index=False)
    import_run_id = _commit_completion(result)
    run_full_pipeline(import_run_id=import_run_id)
    export_active_results(args.output_dir)
    pd.DataFrame([{
        "uploaded_filename": args.workbook.name,
        "uploaded_sha256": result.workbook_sha256,
        "file_size": result.workbook_size,
        "template_schema_version": result.parsed_workbook.schema.version,
        "case_id": result.parsed_workbook.metadata.get("case_id"),
        "dataset_id": result.parsed_workbook.metadata.get("dataset_id"),
        "parser_version": result.parser_version,
        "completion_engine_version": result.completion_engine_version,
        "import_run_id": import_run_id,
        "calculation_run_id": import_run_id,
    }]).to_csv(args.output_dir / "run_identity.csv", index=False)
    print(f"import_run_id={import_run_id} calculation_run_id={import_run_id}")
    print(f"workbook_sha256={result.workbook_sha256}")


if __name__ == "__main__":
    main()
