from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List
import json

from sqlalchemy.orm import Session

from sustainsc.models import (
    ProductBatch,
    TraceabilityEvent,
    KPIResult,
    KPINormalizedResult,
    KPI,
)

COMPOSITE_CODES = {"ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"}


def _latest_raw_kpis_for_product_and_scenario(session: Session, product_id: int | None, scenario_id: int | None) -> List[dict]:
    if product_id is None or scenario_id is None:
        return []

    rows = (
        session.query(KPIResult, KPI)
        .join(KPI, KPI.id == KPIResult.kpi_id)
        .filter(~KPI.code.in_(COMPOSITE_CODES))
        .filter(KPIResult.product_id == product_id)
        .filter(KPIResult.scenario_id == scenario_id)
        .all()
    )

    latest = {}
    for r, k in rows:
        prev = latest.get(k.code)
        if prev is None:
            latest[k.code] = (r, k)
        else:
            prev_r, _ = prev
            prev_ts = prev_r.period_end or datetime.min
            curr_ts = r.period_end or datetime.min
            if curr_ts >= prev_ts:
                latest[k.code] = (r, k)

    return [
        {
            "kpi_code": k.code,
            "kpi_name": k.name,
            "value": r.value,
            "period_end": r.period_end.isoformat() if r.period_end else None,
        }
        for r, k in sorted(latest.values(), key=lambda x: x[1].code)
    ]


def _latest_normalized_kpis_for_scenario(session: Session, scenario_id: int | None) -> List[dict]:
    if scenario_id is None:
        return []

    rows = (
        session.query(KPINormalizedResult, KPI)
        .join(KPI, KPI.id == KPINormalizedResult.kpi_id)
        .filter(~KPI.code.in_(COMPOSITE_CODES))
        .filter(KPINormalizedResult.scenario_id == scenario_id)
        .all()
    )

    latest = {}
    for r, k in rows:
        prev = latest.get(k.code)
        if prev is None:
            latest[k.code] = (r, k)
        else:
            prev_r, _ = prev
            prev_ts = prev_r.period_end or datetime.min
            curr_ts = r.period_end or datetime.min
            if curr_ts >= prev_ts:
                latest[k.code] = (r, k)

    return [
        {
            "kpi_code": k.code,
            "kpi_name": k.name,
            "raw_value": r.raw_value,
            "normalized_value": r.normalized_value,
            "semaforo": r.semaforo,
            "period_end": r.period_end.isoformat() if r.period_end else None,
        }
        for r, k in sorted(latest.values(), key=lambda x: x[1].code)
    ]


def build_dpp_passport(session: Session, batch_code: str) -> Dict[str, Any]:
    batch = session.query(ProductBatch).filter_by(batch_code=batch_code).first()
    if not batch:
        raise ValueError(f"Batch not found: {batch_code}")

    events = (
        session.query(TraceabilityEvent)
        .filter_by(batch_id=batch.id)
        .order_by(TraceabilityEvent.timestamp.asc())
        .all()
    )

    raw_kpis = _latest_raw_kpis_for_product_and_scenario(
        session,
        product_id=batch.product_id,
        scenario_id=batch.scenario_id,
    )

    normalized_kpis = _latest_normalized_kpis_for_scenario(
        session,
        scenario_id=batch.scenario_id,
    )

    passport = {
        "passport_type": "DPP-ready prototype",
        "product_identity": {
            "product_code": batch.product.code if batch.product else None,
            "product_name": batch.product.name if batch.product else None,
            "batch_code": batch.batch_code,
            "scenario_code": batch.scenario.code if batch.scenario else None,
            "origin_facility": batch.origin_facility.name if batch.origin_facility else None,
            "production_date": batch.production_date.isoformat() if batch.production_date else None,
            "quantity": batch.quantity,
            "unit": batch.unit,
            "status": batch.status,
            "notes": batch.notes,
        },
        "traceability_events": [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "facility": e.facility.name if e.facility else None,
                "process": e.process.name if e.process else None,
                "transport_leg": e.transport_leg.code if e.transport_leg else None,
                "quantity": e.quantity,
                "unit": e.unit,
                "source_system": e.source_system,
                "comment": e.comment,
            }
            for e in events
        ],
        "raw_kpis": raw_kpis,
        "normalized_kpis": normalized_kpis,
    }

    return passport


def dpp_passport_to_json(passport: Dict[str, Any]) -> str:
    return json.dumps(passport, indent=2, ensure_ascii=False)