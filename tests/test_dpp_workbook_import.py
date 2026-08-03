from io import BytesIO

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sustainsc.config import Base
from sustainsc.dpp_import import (
    DPPImportValidationError, import_dpp_workbook, read_dpp_workbook,
)
from sustainsc.dpp_service import build_dpp_core
from sustainsc.models import (
    Facility, ImportRun, ImportRunScenario, Product, ProductBatch, Scenario,
    TraceabilityEvent, TransportLeg,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()
    scenario = Scenario(code="ACTIVE", name="Active")
    stale = Scenario(code="STALE", name="Stale")
    db.add_all([
        scenario, stale, Product(code="P-1", name="Product", fu_unit="t"),
        Facility(code="F-1", name="Plant"),
    ])
    db.flush()
    run = ImportRun(
        dataset_name="test", import_timestamp=pd.Timestamp("2026-01-01").to_pydatetime(),
        status="active", scenario_count=1, measurement_count=0, is_active=True,
    )
    db.add(run)
    db.flush()
    db.add(ImportRunScenario(import_run_id=run.id, scenario_id=scenario.id))
    db.commit()
    yield db
    db.close()


def workbook(*, scenario="ACTIVE", event_batch="B-1", examples=False):
    batches = [{
        "batch_code": "B-1", "product_code": "P-1", "scenario_code": scenario,
        "origin_facility_code": "F-1", "production_date": "2026-01-02",
        "quantity": 10, "unit": "t", "status": "produced",
        "source_system": "erp", "source_reference": "order-1", "notes": "",
    }]
    events = [{
        "event_code": "E-1", "batch_code": event_batch, "event_type": "produced",
        "timestamp": "2026-01-02 10:00", "facility_code": "F-1",
        "process_code": "", "transport_leg_code": "", "quantity": 10, "unit": "t",
        "source_system": "mes", "source_reference": "event-1", "comment": "",
    }]
    if examples:
        batches.append({**batches[0], "batch_code": "EXAMPLE-BATCH-001"})
        events.append({**events[0], "event_code": "EXAMPLE-EVT-001",
                       "batch_code": "EXAMPLE-BATCH-001"})
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([{"instructions": "ignored"}]).to_excel(
            writer, sheet_name="00_GUIDE", index=False
        )
        pd.DataFrame(batches).to_excel(
            writer, sheet_name="01_PRODUCT_BATCHES", index=False
        )
        pd.DataFrame(events).to_excel(
            writer, sheet_name="02_TRACEABILITY_EVENTS", index=False
        )
        pd.DataFrame([{"field": "batch_code"}]).to_excel(
            writer, sheet_name="03_DATA_DICTIONARY", index=False
        )
    return output.getvalue()


def test_reader_detects_table_headers_after_documentation_rows():
    output = BytesIO()
    batches = pd.DataFrame([{**{
        "batch_code": "B-1", "product_code": "P-1", "scenario_code": "ACTIVE",
        "origin_facility_code": "F-1", "production_date": "2026-01-02",
        "quantity": 10, "unit": "t", "status": "produced",
        "source_system": "erp", "source_reference": "ref", "notes": "",
    }}])
    events = pd.DataFrame([{**{
        "event_code": "E-1", "batch_code": "B-1", "event_type": "produced",
        "timestamp": "2026-01-02", "facility_code": "F-1", "process_code": "",
        "transport_leg_code": "", "quantity": 10, "unit": "t",
        "source_system": "mes", "source_reference": "ref", "comment": "",
    }}])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        batches.to_excel(writer, sheet_name="01_PRODUCT_BATCHES", index=False, startrow=3)
        events.to_excel(writer, sheet_name="02_TRACEABILITY_EVENTS", index=False, startrow=3)
        pd.DataFrame({"ignored": [1]}).to_excel(writer, sheet_name="00_GUIDE", index=False)
        pd.DataFrame({"ignored": [1]}).to_excel(writer, sheet_name="03_DATA_DICTIONARY", index=False)
    parsed_batches, parsed_events = read_dpp_workbook(output.getvalue())
    assert parsed_batches["batch_code"].tolist() == ["B-1"]
    assert parsed_events["event_code"].tolist() == ["E-1"]


def test_valid_workbook_is_atomic_idempotent_and_ignores_examples(session):
    first = import_dpp_workbook(session, workbook(examples=True))
    assert first.product_batches.created == 1
    assert first.traceability_events.created == 1
    assert first.ignored_example_rows == 2
    assert session.query(ProductBatch).count() == 1
    assert session.query(TraceabilityEvent).count() == 1
    event = session.query(TraceabilityEvent).one()
    assert event.batch.batch_code == "B-1"
    assert event.source_reference == "event-1"

    second = import_dpp_workbook(session, workbook())
    assert second.product_batches.updated == 1
    assert second.traceability_events.updated == 1
    assert session.query(TraceabilityEvent).count() == 1


def test_inactive_scenario_rejected(session):
    with pytest.raises(DPPImportValidationError, match="Scenario not part"):
        import_dpp_workbook(session, workbook(scenario="STALE"))
    assert session.query(ProductBatch).count() == 0


def test_unknown_event_batch_rolls_back_batches(session):
    with pytest.raises(DPPImportValidationError, match="unknown batch"):
        import_dpp_workbook(session, workbook(event_batch="B-404"))
    assert session.query(ProductBatch).count() == 0
    assert session.query(TraceabilityEvent).count() == 0


def test_missing_required_sheet_fails(session):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([{"batch_code": "B"}]).to_excel(
            writer, sheet_name="01_PRODUCT_BATCHES", index=False
        )
    with pytest.raises(ValueError, match="02_TRACEABILITY_EVENTS"):
        import_dpp_workbook(session, output.getvalue())


def test_cuba_workbook_commits_complete_active_dataset_transaction():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, future=True)()
    try:
        scenario = Scenario(code="BASE", name="Cuba baseline")
        facilities = {
            code: Facility(code=code, name=code)
            for code in (
                "BARIAY", "LOS_CALICHES", "PILON", "RIO_SAGUA", "BUENAVENTURA"
            )
        }
        db.add_all([
            scenario,
            Product(code="AGG_0_20", name="Aggregate 0-20", fu_unit="t"),
            *facilities.values(),
        ])
        db.flush()
        db.add(TransportLeg(
            code="BARIAY_TO_Z1",
            name="Bariay to zone 1",
            from_facility_id=facilities["BARIAY"].id,
        ))
        run = ImportRun(
            dataset_name="Cuba regression",
            import_timestamp=pd.Timestamp("2026-08-03").to_pydatetime(),
            status="active",
            scenario_count=1,
            measurement_count=0,
            is_active=True,
        )
        db.add(run)
        db.flush()
        db.add(ImportRunScenario(import_run_id=run.id, scenario_id=scenario.id))
        db.commit()

        fixture = "tests/fixtures/dpp/SustainSCM_DPP_Traceability_CUBA_FILLED.xlsx"
        parsed_batches, parsed_events = read_dpp_workbook(fixture)
        outcome = import_dpp_workbook(db, fixture, active_import_run_id=run.id)

        assert len(parsed_batches) == outcome.product_batches.rows_read == 18
        assert len(parsed_events) == outcome.traceability_events.rows_read == 24
        assert outcome.product_batches.created == 18
        assert outcome.traceability_events.created == 24
        assert outcome.product_batches.rejected == 0
        assert outcome.traceability_events.rejected == 0
        assert db.query(ProductBatch).filter_by(import_run_id=run.id).count() == 18
        assert db.query(TraceabilityEvent).filter_by(import_run_id=run.id).count() == 24
        assert db.query(TraceabilityEvent).filter(TraceabilityEvent.batch_id.is_(None)).count() == 0
        assert outcome.summaries["BASE"]["dpp_batches_total"] == 18
        assert outcome.summaries["BASE"]["dpp_traceability_events_total"] == 24
        assert outcome.summaries["BASE"]["dpp_volume"] == pytest.approx(230400)
        assert outcome.summaries["BASE"]["dpp_valid_volume"] == pytest.approx(230400)
        assert build_dpp_core(
            db, "BATCH_BASE_001", import_run_id=run.id
        )["traceability_events"]

        second = import_dpp_workbook(db, fixture, active_import_run_id=run.id)
        assert second.product_batches.updated == 18
        assert second.traceability_events.updated == 24
        assert db.query(ProductBatch).count() == 18
        assert db.query(TraceabilityEvent).count() == 24
    finally:
        db.close()
