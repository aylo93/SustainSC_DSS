from io import BytesIO

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sustainsc.config import Base
from sustainsc.dpp_import import (
    DPPImportValidationError, import_dpp_workbook, read_dpp_workbook,
)
from sustainsc.models import (
    Facility, ImportRun, ImportRunScenario, Product, ProductBatch, Scenario,
    TraceabilityEvent,
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
