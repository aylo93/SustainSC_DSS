from __future__ import annotations

import datetime

from sustainsc.config import SessionLocal
from sustainsc.models import (
    Scenario,
    Product,
    Facility,
    Process,
    TransportLeg,
    ProductBatch,
    TraceabilityEvent,
)


def get_or_create_scenario(session, code="BASE", name="Baseline"):
    obj = session.query(Scenario).filter_by(code=code).first()
    if obj:
        return obj
    obj = Scenario(code=code, name=name, description="Auto-created for DPP demo", notes="")
    session.add(obj)
    session.flush()
    return obj


def get_or_create_product(session, code="AGG_0_4", name="Aggregate 0-4 mm", fu_unit="t"):
    obj = session.query(Product).filter_by(code=code).first()
    if obj:
        return obj
    obj = Product(code=code, name=name, fu_unit=fu_unit, dpp_ref=None)
    session.add(obj)
    session.flush()
    return obj


def get_or_create_facility(session, code="BARIAY", name="Bariay", location="Holguín", facility_type="quarry"):
    obj = session.query(Facility).filter_by(code=code).first()
    if obj:
        return obj
    obj = Facility(code=code, name=name, location=location, facility_type=facility_type)
    session.add(obj)
    session.flush()
    return obj


def get_or_create_process(session, code="CRUSHING", name="Crushing", facility_id=None):
    obj = session.query(Process).filter_by(code=code).first()
    if obj:
        return obj
    obj = Process(code=code, name=name, description="Demo crushing process", facility_id=facility_id)
    session.add(obj)
    session.flush()
    return obj


def get_or_create_transport_leg(session, code="BARIAY_TO_CLIENT", name="Bariay to Client", from_facility_id=None, to_facility_id=None):
    obj = session.query(TransportLeg).filter_by(code=code).first()
    if obj:
        return obj
    obj = TransportLeg(
        code=code,
        name=name,
        from_facility_id=from_facility_id,
        to_facility_id=to_facility_id,
        mode="truck",
        distance_km=45.0,
    )
    session.add(obj)
    session.flush()
    return obj


def get_or_create_batch(session, batch_code, product_id, scenario_id, origin_facility_id):
    obj = session.query(ProductBatch).filter_by(batch_code=batch_code).first()
    if obj:
        return obj
    obj = ProductBatch(
        batch_code=batch_code,
        product_id=product_id,
        scenario_id=scenario_id,
        origin_facility_id=origin_facility_id,
        production_date=datetime.datetime(2025, 7, 15, 10, 0, 0),
        quantity=300.0,
        unit="t",
        status="produced",
        notes="Demo DPP-ready batch",
    )
    session.add(obj)
    session.flush()
    return obj


def add_event_if_missing(
    session,
    batch_id,
    event_type,
    timestamp,
    facility_id=None,
    process_id=None,
    transport_leg_id=None,
    quantity=None,
    unit=None,
    source_system="demo_seed",
    comment="",
):
    exists = (
        session.query(TraceabilityEvent)
        .filter_by(batch_id=batch_id, event_type=event_type, timestamp=timestamp)
        .first()
    )
    if exists:
        return exists

    obj = TraceabilityEvent(
        batch_id=batch_id,
        event_type=event_type,
        timestamp=timestamp,
        facility_id=facility_id,
        process_id=process_id,
        transport_leg_id=transport_leg_id,
        quantity=quantity,
        unit=unit,
        source_system=source_system,
        comment=comment,
    )
    session.add(obj)
    session.flush()
    return obj


def main():
    session = SessionLocal()
    try:
        scenario = get_or_create_scenario(session)
        product = get_or_create_product(session)
        facility = get_or_create_facility(session)
        process = get_or_create_process(session, facility_id=facility.id)
        leg = get_or_create_transport_leg(session, from_facility_id=facility.id, to_facility_id=facility.id)

        batch = get_or_create_batch(
            session,
            batch_code="BATCH_DEMO_001",
            product_id=product.id,
            scenario_id=scenario.id,
            origin_facility_id=facility.id,
        )

        add_event_if_missing(
            session,
            batch_id=batch.id,
            event_type="produced",
            timestamp=datetime.datetime(2025, 7, 15, 10, 0, 0),
            facility_id=facility.id,
            process_id=process.id,
            quantity=300.0,
            unit="t",
            comment="Batch production completed",
        )

        add_event_if_missing(
            session,
            batch_id=batch.id,
            event_type="quality_checked",
            timestamp=datetime.datetime(2025, 7, 15, 12, 0, 0),
            facility_id=facility.id,
            process_id=process.id,
            quantity=300.0,
            unit="t",
            comment="Quality verification completed",
        )

        add_event_if_missing(
            session,
            batch_id=batch.id,
            event_type="shipped",
            timestamp=datetime.datetime(2025, 7, 16, 8, 0, 0),
            facility_id=facility.id,
            transport_leg_id=leg.id,
            quantity=300.0,
            unit="t",
            comment="Batch shipped to destination",
        )

        session.commit()
        print("DPP demo data created successfully.")
        print(f"Batch code: {batch.batch_code}")
    finally:
        session.close()


if __name__ == "__main__":
    main()