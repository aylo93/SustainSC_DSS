"""Persistent import-run scoping for measurements and analytical results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from .config import Base, engine
from .models import ImportRun, ImportRunScenario, Scenario


class DatasetIntegrityError(ValueError):
    pass


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ensure_dataset_schema() -> None:
    """Create run tables and safely add nullable run/factor columns to legacy SQLite DBs."""
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name != "sqlite":
        return
    additions = {
        "sc_import_run": [
            ("case_id", "VARCHAR(255)"),
            ("dataset_id", "VARCHAR(255)"),
            ("schema_version", "VARCHAR(30)"),
        ],
        "sc_measurement": [("import_run_id", "INTEGER")],
        "sc_kpi_result": [("import_run_id", "INTEGER")],
        "sc_kpi_normalized_result": [("import_run_id", "INTEGER")],
        "sc_emission_factor": [
            ("code", "VARCHAR(100)"),
            ("analytical_role", "VARCHAR(100)"),
            ("factor_set_id", "VARCHAR(100)"),
        ],
        "sc_product_batch": [
            ("source_system", "VARCHAR(100)"),
            ("source_reference", "VARCHAR(255)"),
            ("import_run_id", "INTEGER"),
            ("created_at", "DATETIME"),
            ("updated_at", "DATETIME"),
        ],
        "sc_traceability_event": [
            ("event_code", "VARCHAR(120)"),
            ("source_reference", "VARCHAR(255)"),
            ("import_run_id", "INTEGER"),
            ("created_at", "DATETIME"),
            ("updated_at", "DATETIME"),
        ],
    }
    with engine.begin() as connection:
        current = inspect(connection)
        for table_name, columns in additions.items():
            existing = {c["name"] for c in current.get_columns(table_name)}
            for name, sql_type in columns:
                if name not in existing:
                    connection.execute(
                        text(f'ALTER TABLE "{table_name}" ADD COLUMN "{name}" {sql_type}')
                    )


def file_checksum(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_active_import_run(session: Session) -> ImportRun | None:
    return (
        session.query(ImportRun)
        .filter(ImportRun.is_active.is_(True), ImportRun.status == "active")
        .order_by(ImportRun.import_timestamp.desc(), ImportRun.id.desc())
        .first()
    )


def resolve_import_run_id(session: Session, import_run_id: int | None = None) -> int | None:
    if import_run_id is not None:
        return int(import_run_id)
    active = get_active_import_run(session)
    return active.id if active else None


def get_import_run_scenario_ids(
    session: Session, import_run_id: int | None = None
) -> list[int]:
    run_id = resolve_import_run_id(session, import_run_id)
    if run_id is None:
        return []
    from .models import Measurement

    measured_ids = [
        row[0]
        for row in (
            session.query(Measurement.scenario_id)
            .filter(
                Measurement.import_run_id == run_id,
                Measurement.scenario_id.is_not(None),
            )
            .distinct()
            .order_by(Measurement.scenario_id)
            .all()
        )
    ]
    if measured_ids:
        return measured_ids
    return [
        row[0]
        for row in (
            session.query(ImportRunScenario.scenario_id)
            .filter(ImportRunScenario.import_run_id == run_id)
            .order_by(ImportRunScenario.scenario_id)
            .all()
        )
    ]


def get_import_run_scenario_count(
    session: Session, import_run_id: int | None = None
) -> int:
    return len(get_import_run_scenario_ids(session, import_run_id))


def get_import_run_scenario_codes(
    session: Session, import_run_id: int | None = None
) -> list[str]:
    run_id = resolve_import_run_id(session, import_run_id)
    if run_id is None:
        return []
    return [
        row[0]
        for row in (
            session.query(Scenario.code)
            .join(ImportRunScenario, ImportRunScenario.scenario_id == Scenario.id)
            .filter(ImportRunScenario.import_run_id == run_id)
            .order_by(Scenario.code)
            .all()
        )
    ]


def assert_scenario_integrity(
    session: Session, import_run_id: int, source_codes: Iterable[str]
) -> None:
    source = {str(code).strip() for code in source_codes if str(code).strip()}
    active = set(get_import_run_scenario_codes(session, import_run_id))
    if source != active:
        raise DatasetIntegrityError(
            "Import-run scenario membership differs from source; "
            f"unexpected={sorted(active - source)}, missing={sorted(source - active)}"
        )


def activate_import_run(session: Session, import_run: ImportRun) -> None:
    session.query(ImportRun).filter(
        ImportRun.id != import_run.id, ImportRun.is_active.is_(True)
    ).update({"is_active": False, "status": "inactive"}, synchronize_session=False)
    import_run.is_active = True
    import_run.status = "active"
    session.flush()


@dataclass(frozen=True)
class DatasetAudit:
    active_run_id: int | None
    active_scenarios: tuple[str, ...]
    inactive_scenarios: tuple[str, ...]
    orphan_measurements: int
    orphan_kpi_results: int
    orphan_normalized_results: int


def audit_dataset(session: Session) -> DatasetAudit:
    from .models import KPIResult, KPINormalizedResult, Measurement

    active = get_active_import_run(session)
    active_codes = tuple(get_import_run_scenario_codes(session, active.id if active else None))
    inactive_codes = tuple(
        code
        for (code,) in session.query(Scenario.code).order_by(Scenario.code).all()
        if code not in set(active_codes)
    )
    return DatasetAudit(
        active_run_id=active.id if active else None,
        active_scenarios=active_codes,
        inactive_scenarios=inactive_codes,
        orphan_measurements=session.query(Measurement).filter(Measurement.import_run_id.is_(None)).count(),
        orphan_kpi_results=session.query(KPIResult).filter(KPIResult.import_run_id.is_(None)).count(),
        orphan_normalized_results=session.query(KPINormalizedResult)
        .filter(KPINormalizedResult.import_run_id.is_(None))
        .count(),
    )
