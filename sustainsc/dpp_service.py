"""Two-phase Digital Product Passport services for the SustainSCM DSS.

The module creates and validates a batch passport before KPI calculation, derives
scenario-level MRV measurements from validated passports, and enriches passports
afterwards. It is a DPP-ready batch-level prototype, not legal or EU compliance
validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid

from sqlalchemy.orm import Session

from sustainsc.models import (
    KPI,
    KPINormalizedResult,
    KPIResult,
    Measurement,
    ProductBatch,
    Scenario,
    TraceabilityEvent,
)

logger = logging.getLogger(__name__)

PASSPORT_TYPE = "DPP-ready batch-level prototype"
DPP_SCHEMA_VERSION = "1.0"
DPP_SOURCE_SYSTEM = "dpp_generation_module"

# The model has no is_composite field. These are the persisted composite codes;
# UI-only rankings are included defensively if they are ever persisted as KPIs.
COMPOSITE_CODES = frozenset(
    {
        "ENV_INDEX",
        "ECO_INDEX",
        "SOC_INDEX",
        "TECH_INDEX",
        "SUSTAIN_INDEX",
        "SUSTAIN_INDEX_GEOM",
        "SUSTAIN_INDEX_ARITH",
        "WSM",
        "WSM_SCORE",
        "TOPSIS",
        "TOPSIS_SCORE",
    }
)
PRODUCTION_EVENT_TYPES = frozenset({"produced", "production", "manufactured"})
OUTBOUND_EVENT_TYPES = frozenset({"shipped", "shipment", "delivered", "delivery"})


@dataclass(frozen=True)
class DPPValidationResult:
    """Structured prototype completeness validation; not legal compliance."""

    is_valid: bool
    completeness_score: float
    errors: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class DPPPipelineResult:
    """Result of the pre-KPI phase and its optional transactional integration."""

    scenario_code: str
    run_id: str
    summary: dict[str, float]
    measurement_records: list[dict[str, Any]]
    passports: list[dict[str, Any]]
    measurements_written: int
    kpi_pipeline_ran: bool


def is_composite_kpi(code: str | None) -> bool:
    """Return whether a KPI code represents an index or ranking, not a base KPI."""

    normalized = (code or "").strip().upper()
    return (
        normalized in COMPOSITE_CODES
        or normalized.endswith("_INDEX")
        or normalized.startswith("SUSTAIN_INDEX_")
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _batch_by_code(
    session: Session, batch_code: str, import_run_id: int | None = None
) -> ProductBatch:
    query = session.query(ProductBatch).filter_by(batch_code=batch_code)
    if import_run_id is not None:
        query = query.filter(ProductBatch.import_run_id == import_run_id)
    batch = query.first()
    if batch is None:
        raise ValueError(f"Batch not found: {batch_code}")
    return batch


def build_dpp_core(
    session: Session, batch_code: str, import_run_id: int | None = None
) -> dict[str, Any]:
    """Build a KPI-independent DPP core for one batch."""

    batch = _batch_by_code(session, batch_code, import_run_id)
    events = (
        session.query(TraceabilityEvent)
        .filter(TraceabilityEvent.batch_id == batch.id)
        .order_by(TraceabilityEvent.timestamp.asc(), TraceabilityEvent.id.asc())
        .all()
    )

    product = batch.product
    scenario = batch.scenario
    origin = batch.origin_facility
    return {
        "passport_type": PASSPORT_TYPE,
        "schema_version": DPP_SCHEMA_VERSION,
        "dpp_id": f"dpp:batch:{batch.batch_code}",
        "product_identity": {
            "product_code": product.code if product else None,
            "product_name": product.name if product else None,
            "product_fu_unit": product.fu_unit if product else None,
            "product_dpp_ref": product.dpp_ref if product else None,
            "batch_code": batch.batch_code,
            "scenario_code": scenario.code if scenario else None,
            "origin_facility": origin.name if origin else None,
            "origin_facility_code": origin.code if origin else None,
            "origin_location": origin.location if origin else None,
            "origin_facility_type": origin.facility_type if origin else None,
            "production_date": _iso(batch.production_date),
            "quantity": batch.quantity,
            "unit": batch.unit,
            "status": batch.status,
            "notes": batch.notes,
        },
        "traceability_events": [
            {
                "event_type": event.event_type,
                "timestamp": _iso(event.timestamp),
                "facility": event.facility.name if event.facility else None,
                "facility_code": event.facility.code if event.facility else None,
                "process": event.process.name if event.process else None,
                "process_code": event.process.code if event.process else None,
                "transport_leg": event.transport_leg.code if event.transport_leg else None,
                "transport_mode": event.transport_leg.mode if event.transport_leg else None,
                "transport_distance_km": (
                    event.transport_leg.distance_km if event.transport_leg else None
                ),
                "quantity": event.quantity,
                "unit": event.unit,
                "source_system": event.source_system,
                "comment": event.comment,
            }
            for event in events
        ],
    }


def validate_dpp_core(passport: Mapping[str, Any]) -> DPPValidationResult:
    """Validate prototype completeness without asserting regulatory compliance."""

    identity = passport.get("product_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    events = passport.get("traceability_events")
    events = events if isinstance(events, Sequence) and not isinstance(events, (str, bytes)) else []

    checks = [
        ("product code", identity.get("product_code")),
        ("batch code", identity.get("batch_code")),
        ("scenario code", identity.get("scenario_code")),
        ("origin facility", identity.get("origin_facility")),
        ("production date", identity.get("production_date")),
        ("unit", identity.get("unit")),
    ]
    errors = [f"Missing {label}." for label, value in checks if value in (None, "")]

    quantity = identity.get("quantity")
    if quantity is None:
        errors.append("Missing quantity; null quantity contributes zero DPP volume.")
    elif isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
        errors.append("Quantity must be numeric.")
    elif float(quantity) <= 0:
        errors.append("Quantity must be greater than zero.")

    if not events:
        errors.append("At least one traceability event is required.")

    event_types = {
        str(event.get("event_type", "")).strip().lower()
        for event in events
        if isinstance(event, Mapping)
    }
    warnings: list[str] = []
    if not event_types.intersection(PRODUCTION_EVENT_TYPES):
        warnings.append("No explicit production event was found.")
    if events and not event_types.intersection(OUTBOUND_EVENT_TYPES):
        warnings.append("No shipment or delivery event was found.")

    required_count = len(checks) + 2  # quantity + event presence
    passed_count = required_count - len(errors)
    score = max(0.0, min(100.0, passed_count / required_count * 100.0))
    return DPPValidationResult(
        is_valid=not errors,
        completeness_score=round(score, 2),
        errors=errors,
        warnings=warnings,
    )


def summarize_dpp_mrv(
    session: Session, scenario_id: int, import_run_id: int | None = None
) -> dict[str, float]:
    """Summarize validated batch passports into deterministic MRV-ready values."""

    query = session.query(ProductBatch).filter(ProductBatch.scenario_id == scenario_id)
    if import_run_id is not None:
        query = query.filter(ProductBatch.import_run_id == import_run_id)
    batches = query.order_by(ProductBatch.batch_code.asc(), ProductBatch.id.asc()).all()
    valid_count = 0
    total_volume = 0.0
    valid_volume = 0.0
    completeness_total = 0.0
    event_total = 0

    for batch in batches:
        passport = build_dpp_core(
            session, batch.batch_code, import_run_id=import_run_id
        )
        validation = validate_dpp_core(passport)
        quantity = batch.quantity
        numeric_positive = (
            isinstance(quantity, (int, float))
            and not isinstance(quantity, bool)
            and float(quantity) > 0
        )
        if numeric_positive:
            total_volume += float(quantity)
        if validation.is_valid:
            valid_count += 1
            if numeric_positive:
                valid_volume += float(quantity)
        completeness_total += validation.completeness_score
        event_total += len(passport["traceability_events"])

    total_count = len(batches)
    return {
        "dpp_batches_total": float(total_count),
        "dpp_batches_valid": float(valid_count),
        "dpp_volume": total_volume,
        "dpp_valid_volume": valid_volume,
        "dpp_completeness_average": (
            completeness_total / total_count if total_count else 0.0
        ),
        "dpp_traceability_events_total": float(event_total),
    }


def _recognized_mrv_variables(dictionary_path: Path | None = None) -> set[str]:
    path = dictionary_path or Path(__file__).resolve().parent.parent / "config" / "mrv_dictionary.csv"
    if not path.exists():
        logger.warning("MRV dictionary not found at %s", path)
        return set()
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("variable_name", "")).strip()
            for row in csv.DictReader(handle)
            if row.get("variable_name")
        }


def dpp_summary_to_mrv_records(
    summary: Mapping[str, float],
    *,
    scenario_code: str,
    timestamp: datetime,
    run_id: str | None = None,
    dictionary_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert recognized DPP summary measures to the long-format MRV contract."""

    recognized = _recognized_mrv_variables(dictionary_path)
    units = {
        "dpp_batches_total": "count",
        "dpp_batches_valid": "count",
        "dpp_volume": "FU",
        "dpp_valid_volume": "FU",
        "dpp_completeness_average": "%",
        "dpp_traceability_events_total": "count",
    }
    evaluated = int(summary.get("dpp_batches_total", 0))
    valid = int(summary.get("dpp_batches_valid", 0))
    comment = (
        f"DPP schema {DPP_SCHEMA_VERSION}; evaluated batches={evaluated}; "
        f"valid batches={valid}; validation=prototype completeness"
        + (f"; completion run={run_id}" if run_id else "")
    )
    return [
        {
            "variable_name": name,
            "value": float(summary[name]),
            "unit": units[name],
            "timestamp": timestamp,
            "scenario_code": scenario_code,
            "source_system": DPP_SOURCE_SYSTEM,
            "comment": comment,
        }
        for name in units
        if name in summary and name in recognized
    ]


def _latest_raw_kpis_for_product_and_scenario(
    session: Session,
    product_id: int | None,
    scenario_id: int | None,
    import_run_id: int | None = None,
) -> list[dict[str, Any]]:
    if product_id is None or scenario_id is None:
        return []
    query = (
        session.query(KPIResult, KPI)
        .join(KPI, KPI.id == KPIResult.kpi_id)
        .filter(KPIResult.product_id == product_id)
        .filter(KPIResult.scenario_id == scenario_id)
        .order_by(KPI.code.asc(), KPIResult.period_end.desc(), KPIResult.id.desc())
    )
    if import_run_id is not None:
        query = query.filter(KPIResult.import_run_id == import_run_id)
    rows = query.all()
    latest: dict[str, tuple[KPIResult, KPI]] = {}
    for result, kpi in rows:
        if kpi.code not in latest and not is_composite_kpi(kpi.code):
            latest[kpi.code] = (result, kpi)
    return [
        {
            "kpi_code": kpi.code,
            "kpi_name": kpi.name,
            "value": result.value,
            "unit": kpi.unit,
            "period_end": _iso(result.period_end),
            "result_id": result.id,
        }
        for result, kpi in latest.values()
    ]


def _latest_normalized_kpis_for_scenario(
    session: Session,
    scenario_id: int | None,
    import_run_id: int | None = None,
) -> list[dict[str, Any]]:
    if scenario_id is None:
        return []
    query = (
        session.query(KPINormalizedResult, KPI)
        .join(KPI, KPI.id == KPINormalizedResult.kpi_id)
        .filter(KPINormalizedResult.scenario_id == scenario_id)
        .order_by(
            KPI.code.asc(),
            KPINormalizedResult.period_end.desc(),
            KPINormalizedResult.id.desc(),
        )
    )
    if import_run_id is not None:
        query = query.filter(KPINormalizedResult.import_run_id == import_run_id)
    rows = query.all()
    latest: dict[str, tuple[KPINormalizedResult, KPI]] = {}
    for result, kpi in rows:
        if kpi.code not in latest and not is_composite_kpi(kpi.code):
            latest[kpi.code] = (result, kpi)
    return [
        {
            "kpi_code": kpi.code,
            "kpi_name": kpi.name,
            "raw_value": result.raw_value,
            "normalized_value": result.normalized_value,
            "semaforo": result.semaforo,
            "unit": kpi.unit,
            "period_end": _iso(result.period_end),
            "normalization_method": result.normalization_method,
            "result_id": result.id,
        }
        for result, kpi in latest.values()
    ]


def enrich_dpp_with_kpis(
    session: Session,
    passport: dict[str, Any],
    *,
    product_id: int | None,
    scenario_id: int | None,
    include_raw_kpis: bool = True,
    include_normalized_kpis: bool = True,
    import_run_id: int | None = None,
) -> dict[str, Any]:
    """Add decision-support results with explicit non-batch scopes.

    Normalized values and traffic lights are scenario decision-support results,
    not intrinsic physical properties of an individual batch.
    """

    enriched = dict(passport)
    if include_raw_kpis:
        enriched["sustainability_claims"] = {
            "scope": "product_scenario",
            "scope_note": "Results are product-and-scenario level; no batch allocation is applied.",
            "results": _latest_raw_kpis_for_product_and_scenario(
                session, product_id, scenario_id, import_run_id
            ),
        }
    if include_normalized_kpis:
        enriched["decision_support_summary"] = {
            "scope": "scenario",
            "scope_note": (
                "Normalized scores and traffic lights support scenario decisions "
                "and are not batch physical properties."
            ),
            "results": _latest_normalized_kpis_for_scenario(
                session, scenario_id, import_run_id
            ),
        }
    return enriched


def build_dpp_passport(
    session: Session,
    batch_code: str,
    *,
    include_raw_kpis: bool = True,
    include_normalized_kpis: bool = True,
    import_run_id: int | None = None,
) -> dict[str, Any]:
    """Build, validate and optionally enrich a backward-compatible passport."""

    batch = _batch_by_code(session, batch_code, import_run_id)
    passport = build_dpp_core(session, batch_code, import_run_id=import_run_id)
    passport["validation"] = asdict(validate_dpp_core(passport))
    passport = enrich_dpp_with_kpis(
        session,
        passport,
        product_id=batch.product_id,
        scenario_id=batch.scenario_id,
        include_raw_kpis=include_raw_kpis,
        include_normalized_kpis=include_normalized_kpis,
        import_run_id=import_run_id,
    )
    # Legacy aliases remain available to callers while the scoped sections are canonical.
    passport["raw_kpis"] = passport.get("sustainability_claims", {}).get("results", [])
    passport["normalized_kpis"] = passport.get("decision_support_summary", {}).get(
        "results", []
    )
    return passport


def run_scenario_pipeline_with_dpp(
    session: Session,
    scenario_code: str,
    *,
    timestamp: datetime | None = None,
    run_id: str | None = None,
    persist_measurements: bool = True,
    kpi_runner: Callable[[Session, str], None] | None = None,
) -> DPPPipelineResult:
    """Run the transactional pre-KPI DPP phase and optional injected KPI phase.

    The repository's global ``run_full_pipeline`` owns independent sessions, so
    it cannot participate safely in this transaction. Callers may inject a
    session-aware runner; otherwise this function returns the exact MRV records
    that must be committed before invoking the existing global pipeline.
    """

    scenario = session.query(Scenario).filter_by(code=scenario_code).first()
    if scenario is None:
        raise ValueError(f"Scenario not found: {scenario_code}")

    effective_run_id = run_id or str(uuid.uuid4())
    effective_timestamp = timestamp or datetime.now(timezone.utc).replace(tzinfo=None)
    batches = (
        session.query(ProductBatch)
        .filter(ProductBatch.scenario_id == scenario.id)
        .order_by(ProductBatch.batch_code.asc())
        .all()
    )
    passports: list[dict[str, Any]] = []
    critical_errors: list[str] = []
    for batch in batches:
        passport = build_dpp_core(session, batch.batch_code)
        validation = validate_dpp_core(passport)
        passport["validation"] = asdict(validation)
        passports.append(passport)
        if not validation.is_valid:
            critical_errors.extend(
                f"{batch.batch_code}: {error}" for error in validation.errors
            )
    if critical_errors:
        session.rollback()
        raise ValueError("Critical DPP validation failure: " + "; ".join(critical_errors))

    summary = summarize_dpp_mrv(session, scenario.id)
    records = dpp_summary_to_mrv_records(
        summary,
        scenario_code=scenario.code,
        timestamp=effective_timestamp,
        run_id=effective_run_id,
    )
    written = 0
    try:
        if persist_measurements:
            variable_names = [record["variable_name"] for record in records]
            if variable_names:
                (
                    session.query(Measurement)
                    .filter(Measurement.scenario_id == scenario.id)
                    .filter(Measurement.variable_name.in_(variable_names))
                    .delete(synchronize_session=False)
                )
            for record in records:
                session.add(
                    Measurement(
                        scenario_id=scenario.id,
                        variable_name=record["variable_name"],
                        value=record["value"],
                        unit=record["unit"],
                        timestamp=record["timestamp"],
                        source_system=record["source_system"],
                        comment=record["comment"],
                    )
                )
                written += 1
            session.flush()
        if kpi_runner is not None:
            kpi_runner(session, scenario.code)
            passports = [
                enrich_dpp_with_kpis(
                    session,
                    passport,
                    product_id=batch.product_id,
                    scenario_id=batch.scenario_id,
                )
                for passport, batch in zip(passports, batches)
            ]
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Two-phase DPP pipeline failed for scenario %s", scenario_code)
        raise

    return DPPPipelineResult(
        scenario_code=scenario.code,
        run_id=effective_run_id,
        summary=summary,
        measurement_records=records,
        passports=passports,
        measurements_written=written,
        kpi_pipeline_ran=kpi_runner is not None,
    )


def dpp_passport_to_json(passport: Mapping[str, Any]) -> str:
    """Serialize a passport as stable, human-readable JSON."""

    return json.dumps(passport, indent=2, ensure_ascii=False, sort_keys=True)
