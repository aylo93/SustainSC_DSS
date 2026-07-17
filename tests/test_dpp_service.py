from __future__ import annotations

from datetime import datetime
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sustainsc.config import Base
from sustainsc.dpp_service import (
    DPP_SOURCE_SYSTEM,
    build_dpp_core,
    build_dpp_passport,
    dpp_passport_to_json,
    dpp_summary_to_mrv_records,
    enrich_dpp_with_kpis,
    run_scenario_pipeline_with_dpp,
    summarize_dpp_mrv,
    validate_dpp_core,
)
from sustainsc.models import (
    Facility,
    KPI,
    KPINormalizedResult,
    KPIResult,
    Measurement,
    Product,
    ProductBatch,
    Scenario,
    TraceabilityEvent,
)
from sustainsc.kpi_engine import Ctx, dpp_coverage


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db:
        yield db


def seed_batch(
    session,
    *,
    batch_code: str = "B-001",
    scenario_code: str = "SC-1",
    quantity: float | None = 10.0,
    with_events: bool = True,
):
    scenario = session.query(Scenario).filter_by(code=scenario_code).first()
    if scenario is None:
        scenario = Scenario(code=scenario_code, name=scenario_code)
        session.add(scenario)
    product = session.query(Product).filter_by(code="P-1").first()
    if product is None:
        product = Product(code="P-1", name="Product", fu_unit="t")
        session.add(product)
    facility = session.query(Facility).filter_by(code="F-1").first()
    if facility is None:
        facility = Facility(code="F-1", name="Plant", location="X")
        session.add(facility)
    session.flush()
    batch = ProductBatch(
        batch_code=batch_code,
        product_id=product.id,
        scenario_id=scenario.id,
        origin_facility_id=facility.id,
        production_date=datetime(2026, 1, 1),
        quantity=quantity,
        unit="t",
        status="produced",
    )
    session.add(batch)
    session.flush()
    if with_events:
        # Insert out of chronological order to verify SQL ordering.
        session.add_all(
            [
                TraceabilityEvent(
                    batch_id=batch.id,
                    event_type="shipped",
                    timestamp=datetime(2026, 1, 2),
                    facility_id=facility.id,
                    quantity=quantity,
                    unit="t",
                ),
                TraceabilityEvent(
                    batch_id=batch.id,
                    event_type="produced",
                    timestamp=datetime(2026, 1, 1),
                    facility_id=facility.id,
                    quantity=quantity,
                    unit="t",
                ),
            ]
        )
    session.commit()
    return batch


def test_core_generation_and_chronological_events(session):
    seed_batch(session)
    passport = build_dpp_core(session, "B-001")
    assert passport["passport_type"] == "DPP-ready batch-level prototype"
    assert passport["product_identity"]["product_code"] == "P-1"
    assert [event["event_type"] for event in passport["traceability_events"]] == [
        "produced",
        "shipped",
    ]


def test_unknown_batch_raises_value_error(session):
    with pytest.raises(ValueError, match="Batch not found"):
        build_dpp_core(session, "missing")


def test_missing_optional_relationships_do_not_crash(session):
    batch = seed_batch(session, with_events=False)
    batch.origin_facility_id = None
    batch.scenario_id = None
    session.commit()
    passport = build_dpp_core(session, batch.batch_code)
    assert passport["product_identity"]["origin_facility"] is None
    assert passport["product_identity"]["scenario_code"] is None


def test_validation_complete_passport_is_valid(session):
    seed_batch(session)
    result = validate_dpp_core(build_dpp_core(session, "B-001"))
    assert result.is_valid
    assert result.completeness_score == 100.0
    assert not result.errors


@pytest.mark.parametrize("quantity", [0.0, -1.0])
def test_validation_rejects_non_positive_quantity(session, quantity):
    seed_batch(session, quantity=quantity)
    result = validate_dpp_core(build_dpp_core(session, "B-001"))
    assert not result.is_valid
    assert any("greater than zero" in error for error in result.errors)
    assert 0 <= result.completeness_score <= 100


def test_validation_rejects_missing_product_and_events(session):
    passport = {
        "product_identity": {
            "batch_code": "B",
            "scenario_code": "S",
            "origin_facility": "F",
            "production_date": "2026-01-01",
            "quantity": 1.0,
            "unit": "t",
        },
        "traceability_events": [],
    }
    result = validate_dpp_core(passport)
    assert not result.is_valid
    assert "Missing product code." in result.errors
    assert any("traceability event" in error for error in result.errors)
    assert any("production event" in warning for warning in result.warnings)


def test_summary_counts_valid_volume_and_empty_scenario(session):
    valid = seed_batch(session, batch_code="B-VALID", quantity=10)
    invalid = seed_batch(session, batch_code="B-INVALID", quantity=5, with_events=False)
    summary = summarize_dpp_mrv(session, valid.scenario_id)
    assert summary["dpp_batches_total"] == 2
    assert summary["dpp_batches_valid"] == 1
    assert summary["dpp_volume"] == 15
    assert summary["dpp_valid_volume"] == 10
    assert summary["dpp_completeness_average"] == pytest.approx(
        (
            validate_dpp_core(build_dpp_core(session, valid.batch_code)).completeness_score
            + validate_dpp_core(build_dpp_core(session, invalid.batch_code)).completeness_score
        )
        / 2
    )

    empty = Scenario(code="EMPTY", name="Empty")
    session.add(empty)
    session.commit()
    assert summarize_dpp_mrv(session, empty.id) == {
        "dpp_batches_total": 0.0,
        "dpp_batches_valid": 0.0,
        "dpp_volume": 0.0,
        "dpp_valid_volume": 0.0,
        "dpp_completeness_average": 0.0,
        "dpp_traceability_events_total": 0.0,
    }


def test_measurement_conversion_only_emits_recognized_variables(session):
    summary = {
        "dpp_volume": 10.0,
        "dpp_valid_volume": 8.0,
        "unknown_dpp_value": 99.0,
    }
    records = dpp_summary_to_mrv_records(
        summary,
        scenario_code="SC-1",
        timestamp=datetime(2026, 1, 1),
        run_id="run-1",
    )
    assert {record["variable_name"] for record in records} == {
        "dpp_volume",
        "dpp_valid_volume",
    }
    assert all(record["source_system"] == DPP_SOURCE_SYSTEM for record in records)


def test_kpi_enrichment_scopes_composites_and_latest_tie_breaker(session):
    batch = seed_batch(session)
    base_kpi = KPI(
        code="T6",
        name="DPP coverage",
        dimension="technological",
        decision_level="operational",
        flow="information",
        unit="%",
        is_benefit=True,
        formula_id="dpp_coverage",
    )
    composite = KPI(
        code="SUSTAIN_INDEX_GEOM",
        name="Index",
        dimension="sustainability",
        decision_level="strategic",
        flow="information",
        unit="points",
        is_benefit=True,
        formula_id="direct",
    )
    session.add_all([base_kpi, composite])
    session.flush()
    period = datetime(2026, 2, 1)
    session.add_all(
        [
            KPIResult(
                kpi_id=base_kpi.id,
                scenario_id=batch.scenario_id,
                product_id=batch.product_id,
                period_end=period,
                value=10,
            ),
            KPIResult(
                kpi_id=base_kpi.id,
                scenario_id=batch.scenario_id,
                product_id=batch.product_id,
                period_end=period,
                value=20,
            ),
            KPIResult(
                kpi_id=composite.id,
                scenario_id=batch.scenario_id,
                product_id=batch.product_id,
                period_end=period,
                value=90,
            ),
            KPINormalizedResult(
                kpi_id=base_kpi.id,
                scenario_id=batch.scenario_id,
                period_end=period,
                raw_value=10,
                normalized_value=40,
                semaforo="Amber",
            ),
            KPINormalizedResult(
                kpi_id=base_kpi.id,
                scenario_id=batch.scenario_id,
                period_end=period,
                raw_value=20,
                normalized_value=80,
                semaforo="Green",
            ),
            KPINormalizedResult(
                kpi_id=composite.id,
                scenario_id=batch.scenario_id,
                period_end=period,
                raw_value=90,
                normalized_value=90,
                semaforo="Green",
            ),
        ]
    )
    session.commit()

    enriched = enrich_dpp_with_kpis(
        session,
        build_dpp_core(session, batch.batch_code),
        product_id=batch.product_id,
        scenario_id=batch.scenario_id,
    )
    assert enriched["sustainability_claims"]["scope"] == "product_scenario"
    assert enriched["decision_support_summary"]["scope"] == "scenario"
    assert enriched["sustainability_claims"]["results"][0]["value"] == 20
    assert enriched["decision_support_summary"]["results"][0]["normalized_value"] == 80
    assert {
        row["kpi_code"]
        for row in enriched["sustainability_claims"]["results"]
        + enriched["decision_support_summary"]["results"]
    } == {"T6"}

    no_raw = enrich_dpp_with_kpis(
        session,
        build_dpp_core(session, batch.batch_code),
        product_id=batch.product_id,
        scenario_id=batch.scenario_id,
        include_raw_kpis=False,
    )
    assert "sustainability_claims" not in no_raw
    assert "decision_support_summary" in no_raw


def test_backward_compatibility_and_json(session):
    seed_batch(session)
    passport = build_dpp_passport(session, "B-001")
    assert "raw_kpis" in passport
    assert "normalized_kpis" in passport
    assert json.loads(dpp_passport_to_json(passport))["dpp_id"] == "dpp:batch:B-001"


def test_pipeline_is_idempotent_and_runs_kpi_after_mrv(session):
    batch = seed_batch(session)
    observed = []

    def runner(db, scenario_code):
        observed.append(
            (
                scenario_code,
                db.query(Measurement)
                .filter_by(
                    scenario_id=batch.scenario_id,
                    source_system=DPP_SOURCE_SYSTEM,
                )
                .count(),
            )
        )

    first = run_scenario_pipeline_with_dpp(
        session, "SC-1", timestamp=datetime(2026, 3, 1), kpi_runner=runner
    )
    second = run_scenario_pipeline_with_dpp(
        session, "SC-1", timestamp=datetime(2026, 3, 2), kpi_runner=runner
    )
    assert first.measurements_written == 2
    assert second.measurements_written == 2
    assert "sustainability_claims" in first.passports[0]
    assert "decision_support_summary" in first.passports[0]
    assert observed == [("SC-1", 2), ("SC-1", 2)]
    assert (
        session.query(Measurement)
        .filter_by(scenario_id=batch.scenario_id, source_system=DPP_SOURCE_SYSTEM)
        .count()
        == 2
    )


def test_pipeline_blocks_invalid_dpp_and_rolls_back_runner_failure(session):
    invalid = seed_batch(session, batch_code="B-BAD", with_events=False)
    with pytest.raises(ValueError, match="Critical DPP validation"):
        run_scenario_pipeline_with_dpp(session, "SC-1")
    assert session.query(Measurement).count() == 0

    session.delete(invalid)
    session.commit()
    valid = seed_batch(session, batch_code="B-GOOD")

    def failing_runner(db, scenario_code):
        assert db.query(Measurement).filter_by(scenario_id=valid.scenario_id).count() == 2
        raise RuntimeError("KPI failure")

    with pytest.raises(RuntimeError, match="KPI failure"):
        run_scenario_pipeline_with_dpp(session, "SC-1", kpi_runner=failing_runner)
    assert session.query(Measurement).count() == 0


def test_dpp_coverage_prefers_validated_volume(session):
    batch = seed_batch(session)
    timestamp = datetime(2026, 4, 1)
    session.add_all(
        [
            Measurement(
                scenario_id=batch.scenario_id,
                variable_name="dpp_volume",
                value=100,
                unit="t",
                timestamp=timestamp,
            ),
            Measurement(
                scenario_id=batch.scenario_id,
                variable_name="dpp_valid_volume",
                value=80,
                unit="t",
                timestamp=timestamp,
            ),
            Measurement(
                scenario_id=batch.scenario_id,
                variable_name="shipped_volume_total",
                value=100,
                unit="t",
                timestamp=timestamp,
            ),
        ]
    )
    session.commit()
    assert dpp_coverage(Ctx(session, batch.scenario_id, {})) == 80
