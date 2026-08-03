"""Transactional ProductBatch + TraceabilityEvent workbook import."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import logging
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sqlalchemy.orm import Session

from .dataset_scope import get_import_run_scenario_codes, resolve_import_run_id
from .dpp_service import (
    build_dpp_core, dpp_summary_to_mrv_records,
    summarize_dpp_mrv, validate_dpp_core,
)
from .models import (
    Facility, Measurement, Process, Product, ProductBatch, Scenario, TraceabilityEvent,
    TransportLeg,
)

BATCH_SHEET = "01_PRODUCT_BATCHES"
EVENT_SHEET = "02_TRACEABILITY_EVENTS"
BATCH_COLUMNS = (
    "batch_code", "product_code", "scenario_code", "origin_facility_code",
    "production_date", "quantity", "unit", "status", "source_system",
    "source_reference", "notes",
)
EVENT_COLUMNS = (
    "event_code", "batch_code", "event_type", "timestamp", "facility_code",
    "process_code", "transport_leg_code", "quantity", "unit", "source_system",
    "source_reference", "comment",
)
ALLOWED_STATUSES = frozenset(
    {"produced", "in_stock", "shipped", "delivered", "blocked", "released"}
)
logger = logging.getLogger(__name__)


@dataclass
class EntityImportResult:
    rows_read: int = 0
    valid: int = 0
    created: int = 0
    updated: int = 0
    rejected: int = 0


@dataclass
class DPPWorkbookImportResult:
    product_batches: EntityImportResult = field(default_factory=EntityImportResult)
    traceability_events: EntityImportResult = field(default_factory=EntityImportResult)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    references: dict[str, set[str]] = field(default_factory=lambda: {
        "unknown_products": set(), "unknown_scenarios": set(),
        "unknown_facilities": set(), "unknown_processes": set(),
        "unknown_transport_legs": set(),
    })
    ignored_example_rows: int = 0
    summaries: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


class DPPImportValidationError(ValueError):
    def __init__(self, result: DPPWorkbookImportResult):
        super().__init__("; ".join(result.errors))
        self.result = result


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.dropna(how="all").copy()
    result.columns = [str(c).strip().lower() for c in result.columns]
    return result


def _read_sheet_table(book: pd.ExcelFile, sheet_name: str, key_column: str) -> pd.DataFrame:
    """Read a workbook table even when explanatory rows precede its header."""
    raw = pd.read_excel(book, sheet_name=sheet_name, header=None)
    header_row = None
    for index, row in raw.iterrows():
        values = {str(value).strip().lower() for value in row if not pd.isna(value)}
        if key_column in values:
            header_row = index
            break
    if header_row is None:
        raise ValueError(
            f"Sheet {sheet_name} does not contain the required {key_column} header."
        )
    frame = raw.iloc[header_row + 1 :].copy()
    frame.columns = raw.iloc[header_row].tolist()
    return _clean_frame(frame)


def read_dpp_workbook(source: str | Path | bytes | BinaryIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(source, bytes):
        source = BytesIO(source)
    book = pd.ExcelFile(source)
    missing = [name for name in (BATCH_SHEET, EVENT_SHEET) if name not in book.sheet_names]
    if missing:
        raise ValueError("Missing required workbook sheet(s): " + ", ".join(missing))
    return (
        _read_sheet_table(book, BATCH_SHEET, "batch_code"),
        _read_sheet_table(book, EVENT_SHEET, "event_code"),
    )


def _text(value) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _positive(value) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _datetime(value) -> datetime | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.to_pydatetime()


def _require_columns(frame, required, label, result):
    missing = [name for name in required if name not in frame.columns]
    if missing:
        result.errors.append(f"{label} missing required columns: {', '.join(missing)}")


def _maps(session: Session):
    def mapping(model):
        return {obj.code: obj for obj in session.query(model).all()}
    return (
        mapping(Product), mapping(Scenario), mapping(Facility),
        mapping(Process), mapping(TransportLeg),
    )


def _filter_examples(batch_df, event_df, result):
    bmask = batch_df.get("batch_code", pd.Series(index=batch_df.index, dtype=object)).astype(str).str.startswith("EXAMPLE-")
    emask = event_df.get("event_code", pd.Series(index=event_df.index, dtype=object)).astype(str).str.startswith("EXAMPLE-")
    result.ignored_example_rows = int(bmask.sum() + emask.sum())
    if result.ignored_example_rows:
        result.warnings.append("Template example rows were ignored.")
    return batch_df.loc[~bmask].copy(), event_df.loc[~emask].copy()


def import_dpp_workbook(
    session: Session,
    source: str | Path | bytes | BinaryIO,
    *,
    active_import_run_id: int | None = None,
    commit: bool = True,
) -> DPPWorkbookImportResult:
    """Validate and atomically upsert both workbook sheets.

    Master records are never auto-created. Any row error rejects the complete
    workbook, ensuring a batch sheet cannot be committed without its events.
    """
    result = DPPWorkbookImportResult()
    try:
        batch_df, event_df = read_dpp_workbook(source)
        logger.info(
            "DPP workbook parsed: batches=%s events=%s",
            len(batch_df),
            len(event_df),
        )
        result.product_batches.rows_read = len(batch_df)
        result.traceability_events.rows_read = len(event_df)
        _require_columns(batch_df, BATCH_COLUMNS, "Product batches", result)
        _require_columns(event_df, EVENT_COLUMNS, "Traceability events", result)
        if result.errors:
            raise DPPImportValidationError(result)
        batch_df, event_df = _filter_examples(batch_df, event_df, result)

        run_id = resolve_import_run_id(session, active_import_run_id)
        active_codes = set(get_import_run_scenario_codes(session, run_id))
        if run_id is None or not active_codes:
            result.errors.append("No active dataset with scenario membership is available.")
            raise DPPImportValidationError(result)

        products, scenarios, facilities, processes, legs = _maps(session)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        batch_rows: list[dict] = []
        seen_batches: set[str] = set()
        for idx, row in batch_df.iterrows():
            tag = f"Product batch row {idx + 2}"
            code, product_code = _text(row.batch_code), _text(row.product_code)
            scenario_code, facility_code = _text(row.scenario_code), _text(row.origin_facility_code)
            production_date, quantity = _datetime(row.production_date), _positive(row.quantity)
            unit, status = _text(row.unit), (_text(row.status) or "").lower()
            before = len(result.errors)
            if not code:
                result.errors.append(f"{tag}: batch_code is required.")
            elif code in seen_batches:
                result.errors.append(f"Duplicate batch_code: {code}")
            else:
                seen_batches.add(code)
            if not product_code or product_code not in products:
                result.references["unknown_products"].add(product_code or "(blank)")
                result.errors.append(f"Unknown product code: {product_code or '(blank)'}")
            if not scenario_code or scenario_code not in active_codes:
                result.references["unknown_scenarios"].add(scenario_code or "(blank)")
                result.errors.append(f"Scenario not part of active dataset: {scenario_code or '(blank)'}")
            if not facility_code or facility_code not in facilities:
                result.references["unknown_facilities"].add(facility_code or "(blank)")
                result.errors.append(f"Unknown facility: {facility_code or '(blank)'}")
            if production_date is None:
                result.errors.append(f"{tag}: invalid production_date.")
            if quantity is None:
                result.errors.append(f"Quantity must be positive: {code or tag}")
            if not unit:
                result.errors.append(f"{tag}: unit is required.")
            if status not in ALLOWED_STATUSES:
                result.errors.append(f"{tag}: invalid status: {status or '(blank)'}")
            if len(result.errors) == before:
                batch_rows.append(dict(code=code, product=products[product_code],
                    scenario=scenarios[scenario_code], facility=facilities[facility_code],
                    production_date=production_date, quantity=quantity, unit=unit,
                    status=status, source_system=_text(row.source_system),
                    source_reference=_text(row.source_reference), notes=_text(row.notes)))
                result.product_batches.valid += 1
            else:
                result.product_batches.rejected += 1

        event_rows: list[dict] = []
        seen_events: set[tuple] = set()
        known_db_batches = {
            b.batch_code: b for b in session.query(ProductBatch)
            .filter(ProductBatch.import_run_id == run_id).all()
        }
        valid_batch_codes = {r["code"] for r in batch_rows} | set(known_db_batches)
        for idx, row in event_df.iterrows():
            tag = f"Traceability event row {idx + 2}"
            event_code, batch_code = _text(row.event_code), _text(row.batch_code)
            event_type, timestamp = _text(row.event_type), _datetime(row.timestamp)
            facility_code, process_code = _text(row.facility_code), _text(row.process_code)
            leg_code, quantity, unit = _text(row.transport_leg_code), _positive(row.quantity), _text(row.unit)
            key = (event_code,) if event_code else (
                batch_code, (event_type or "").lower(), timestamp, _text(row.source_reference)
            )
            before = len(result.errors)
            if not batch_code or batch_code not in valid_batch_codes:
                result.errors.append(f"Event references unknown batch: {batch_code or '(blank)'}")
            if not event_type:
                result.errors.append(f"{tag}: event_type is required.")
            if timestamp is None:
                result.errors.append(f"{tag}: invalid timestamp.")
            if key in seen_events:
                result.errors.append(f"Duplicate event: {event_code or key}")
            else:
                seen_events.add(key)
            if facility_code and facility_code not in facilities:
                result.references["unknown_facilities"].add(facility_code)
                result.errors.append(f"Unknown facility: {facility_code}")
            if process_code and process_code not in processes:
                result.references["unknown_processes"].add(process_code)
                result.errors.append(f"Unknown process: {process_code}")
            if leg_code and leg_code not in legs:
                result.references["unknown_transport_legs"].add(leg_code)
                result.errors.append(f"Unknown transport leg: {leg_code}")
            raw_quantity = row.quantity
            if not pd.isna(raw_quantity) and quantity is None:
                result.errors.append(f"{tag}: quantity must be positive when supplied.")
            if (quantity is None) != (unit is None):
                result.errors.append(
                    f"{tag}: quantity and unit must either both be supplied or both be blank."
                )
            if len(result.errors) == before:
                event_rows.append(dict(event_code=event_code, batch_code=batch_code,
                    event_type=event_type.lower(), timestamp=timestamp,
                    facility=facilities.get(facility_code), process=processes.get(process_code),
                    leg=legs.get(leg_code), quantity=quantity, unit=unit,
                    source_system=_text(row.source_system),
                    source_reference=_text(row.source_reference), comment=_text(row.comment)))
                result.traceability_events.valid += 1
            else:
                result.traceability_events.rejected += 1

        if result.errors:
            raise DPPImportValidationError(result)

        imported: dict[str, ProductBatch] = {}
        for row in batch_rows:
            obj = session.query(ProductBatch).filter_by(
                batch_code=row["code"], import_run_id=run_id
            ).first()
            if obj is None:
                obj = ProductBatch(batch_code=row["code"], created_at=now)
                session.add(obj)
                result.product_batches.created += 1
            else:
                result.product_batches.updated += 1
            obj.product_id, obj.scenario_id = row["product"].id, row["scenario"].id
            obj.origin_facility_id, obj.production_date = row["facility"].id, row["production_date"]
            obj.quantity, obj.unit, obj.status, obj.notes = row["quantity"], row["unit"], row["status"], row["notes"]
            obj.source_system, obj.source_reference = row["source_system"], row["source_reference"]
            obj.import_run_id, obj.updated_at = run_id, now
            imported[row["code"]] = obj
        session.flush()

        for row in event_rows:
            batch = imported.get(row["batch_code"]) or known_db_batches[row["batch_code"]]
            query = session.query(TraceabilityEvent)
            if row["event_code"]:
                obj = query.filter_by(event_code=row["event_code"], import_run_id=run_id).first()
            else:
                obj = query.filter_by(batch_id=batch.id, event_type=row["event_type"],
                    timestamp=row["timestamp"], source_reference=row["source_reference"]).first()
            if obj is None:
                obj = TraceabilityEvent(batch_id=batch.id, created_at=now)
                session.add(obj)
                result.traceability_events.created += 1
            else:
                result.traceability_events.updated += 1
            obj.event_code, obj.event_type, obj.timestamp = row["event_code"], row["event_type"], row["timestamp"]
            obj.facility_id = row["facility"].id if row["facility"] else None
            obj.process_id = row["process"].id if row["process"] else None
            obj.transport_leg_id = row["leg"].id if row["leg"] else None
            obj.quantity, obj.unit = row["quantity"], row["unit"]
            obj.source_system, obj.source_reference, obj.comment = row["source_system"], row["source_reference"], row["comment"]
            obj.import_run_id, obj.updated_at = run_id, now
        session.flush()

        for code in sorted(imported):
            validation = validate_dpp_core(
                build_dpp_core(session, code, import_run_id=run_id)
            )
            if not validation.is_valid:
                result.errors.extend(f"{code}: {error}" for error in validation.errors)
        if result.errors:
            raise DPPImportValidationError(result)
        for scenario_code in sorted({row["scenario"].code for row in batch_rows}):
            scenario = scenarios[scenario_code]
            summary = summarize_dpp_mrv(
                session, scenarios[scenario_code].id, run_id
            )
            result.summaries[scenario_code] = summary
            records = dpp_summary_to_mrv_records(
                summary, scenario_code=scenario_code, timestamp=now,
                run_id=f"import-run-{run_id}",
            )
            variable_names = [record["variable_name"] for record in records]
            if variable_names:
                (
                    session.query(Measurement)
                    .filter(
                        Measurement.import_run_id == run_id,
                        Measurement.scenario_id == scenario.id,
                        Measurement.variable_name.in_(variable_names),
                    )
                    .delete(synchronize_session=False)
                )
            for record in records:
                session.add(Measurement(
                    variable_name=record["variable_name"], value=record["value"],
                    unit=record["unit"], timestamp=record["timestamp"],
                    scenario_id=scenario.id, import_run_id=run_id,
                    source_system=record["source_system"], comment=record["comment"],
                ))
        if commit:
            session.commit()
        logger.info(
            "DPP workbook committed: import_run_id=%s batches_created=%s "
            "batches_updated=%s events_created=%s events_updated=%s",
            run_id,
            result.product_batches.created,
            result.product_batches.updated,
            result.traceability_events.created,
            result.traceability_events.updated,
        )
        return result
    except Exception:
        session.rollback()
        raise
