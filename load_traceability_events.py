from _future_ import annotations

import sys
from pathlib import Path
import pandas as pd

from sustainsc.config import SessionLocal
from sustainsc.models import (
    ProductBatch,
    Facility,
    Process,
    TransportLeg,
    TraceabilityEvent,
)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(how="all")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def get_or_create_facility(session, code: str):
    obj = session.query(Facility).filter_by(code=code).first()
    if obj:
        return obj
    obj = Facility(code=code, name=code, location=None, facility_type=None)
    session.add(obj)
    session.flush()
    return obj


def get_or_create_process(session, code: str):
    obj = session.query(Process).filter_by(code=code).first()
    if obj:
        return obj
    obj = Process(code=code, name=code, description="Auto-created by event loader", facility_id=None)
    session.add(obj)
    session.flush()
    return obj


def get_or_create_transport_leg(session, code: str):
    obj = session.query(TransportLeg).filter_by(code=code).first()
    if obj:
        return obj
    obj = TransportLeg(code=code, name=code, from_facility_id=None, to_facility_id=None, mode=None, distance_km=None)
    session.add(obj)
    session.flush()
    return obj


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_traceability_events.py data/traceability_events.csv")
        sys.exit(1)

    path = Path(sys.argv[1])
    df = read_csv(path)

    required = ["batch_code", "event_type", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "facility_code" not in df.columns:
        df["facility_code"] = None
    if "process_code" not in df.columns:
        df["process_code"] = None
    if "transport_leg_code" not in df.columns:
        df["transport_leg_code"] = None
    if "quantity" not in df.columns:
        df["quantity"] = None
    if "unit" not in df.columns:
        df["unit"] = None
    if "source_system" not in df.columns:
        df["source_system"] = "uploaded_traceability_events"
    if "comment" not in df.columns:
        df["comment"] = ""

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamp values in traceability events file.")

    session = SessionLocal()
    try:
        written = 0

        for _, row in df.iterrows():
            batch_code = str(row["batch_code"]).strip()
            event_type = str(row["event_type"]).strip()

            if not batch_code or not event_type:
                continue

            batch = session.query(ProductBatch).filter_by(batch_code=batch_code).first()
            if not batch:
                raise ValueError(f"Batch not found for event load: {batch_code}. Load batches first.")

            facility = None
            process = None
            leg = None

            if pd.notna(row["facility_code"]) and str(row["facility_code"]).strip():
                facility = get_or_create_facility(session, str(row["facility_code"]).strip())

            if pd.notna(row["process_code"]) and str(row["process_code"]).strip():
                process = get_or_create_process(session, str(row["process_code"]).strip())

            if pd.notna(row["transport_leg_code"]) and str(row["transport_leg_code"]).strip():
                leg = get_or_create_transport_leg(session, str(row["transport_leg_code"]).strip())

            exists = (
                session.query(TraceabilityEvent)
                .filter_by(
                    batch_id=batch.id,
                    event_type=event_type,
                    timestamp=row["timestamp"].to_pydatetime(),
                )
                .first()
            )

            if exists:
                exists.facility_id = facility.id if facility else None
                exists.process_id = process.id if process else None
                exists.transport_leg_id = leg.id if leg else None
                exists.quantity = float(row["quantity"]) if pd.notna(row["quantity"]) else None
                exists.unit = str(row["unit"]).strip() if pd.notna(row["unit"]) else None
                exists.source_system = str(row["source_system"]).strip() if pd.notna(row["source_system"]) else None
                exists.comment = str(row["comment"]).strip() if pd.notna(row["comment"]) else None
            else:
                session.add(
                    TraceabilityEvent(
                        batch_id=batch.id,
                        event_type=event_type,
                        timestamp=row["timestamp"].to_pydatetime(),
                        facility_id=facility.id if facility else None,
                        process_id=process.id if process else None,
                        transport_leg_id=leg.id if leg else None,
                        quantity=float(row["quantity"]) if pd.notna(row["quantity"]) else None,
                        unit=str(row["unit"]).strip() if pd.notna(row["unit"]) else None,
                        source_system=str(row["source_system"]).strip() if pd.notna(row["source_system"]) else None,
                        comment=str(row["comment"]).strip() if pd.notna(row["comment"]) else None,
                    )
                )

            written += 1

        session.commit()
        print(f"Loaded/updated {written} traceability events from {path.name}")
    finally:
        session.close()


if _name_ == "_main_":
    main()