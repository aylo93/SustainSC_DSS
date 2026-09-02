from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from batch_completion_engine import BatchScenarioCompletionEngine
from scenario_completion_engine import ScenarioCompletionEngine
from sustainsc.config import Base
from sustainsc.factor_registry import upsert_approved_emission_factors
from sustainsc.models import EmissionFactor
from sustainsc.mrv_schema_v2 import parse_mrv_workbook


WORKBOOK_ENV = "REEL_V3_WORKBOOK"


def _workbook_path() -> Path:
    configured = os.getenv(WORKBOOK_ENV, "").strip()
    if not configured:
        pytest.skip(f"Set {WORKBOOK_ENV} to run the REEL V3 runtime diagnostic.")
    workbook = Path(configured)
    if not workbook.is_file():
        pytest.skip(f"REEL V3 workbook not found: {workbook}")
    return workbook


def _engine_for_workbook(workbook: Path) -> tuple[ScenarioCompletionEngine, object]:
    parsed = parse_mrv_workbook(workbook)
    engine = ScenarioCompletionEngine(
        "config",
        config_frames={
            "dictionary": parsed.variable_dictionary,
            "scope": parsed.strategy_scope,
            "overrides": parsed.variable_overrides,
            "rules": parsed.mrv_rules,
            "bridges": parsed.bridge_rules,
            "factor_register": parsed.factor_register,
            "default_factor_set_id": parsed.metadata.get("default_emission_factor_set_id"),
        },
    )
    return engine, parsed


def test_reel_transport_factor_authorization_diagnostic() -> None:
    workbook = _workbook_path()
    engine, parsed = _engine_for_workbook(workbook)
    scenario = parsed.scenarios[
        parsed.scenarios["scenario_code"].astype(str).eq("REEL_VSMC_KAIZEN")
    ].iloc[0]
    timestamp = pd.to_datetime(scenario["evaluation_timestamp"])

    diagnostic = engine.diagnose_analytical_factor(
        "REEL_CASE_2026",
        "TRANSPORT_GHG_PER_TKM",
        timestamp,
        expected_factor_type="EMISSION",
        expected_unit="tCO2e/tkm",
        required_role=("transport-scope", "active analytical"),
        required_scope="transport",
    )
    print("REEL_FACTOR_AUTHORIZATION_DIAGNOSTIC")
    print(json.dumps(diagnostic, indent=2, default=str, sort_keys=True))

    assert diagnostic["requested"] == {
        "factor_set_id": "REEL_CASE_2026",
        "factor_code": "TRANSPORT_GHG_PER_TKM",
        "evaluation_timestamp": timestamp,
    }
    assert diagnostic["record"]["factor_type"] == "EMISSION"
    assert diagnostic["record"]["value"] == pytest.approx(0.000102)
    assert diagnostic["record"]["unit"] == "tCO2e/tkm"
    assert diagnostic["record"]["analytical_role"] == "Active analytical factor"
    assert diagnostic["record"]["approval_status"] == "Approved"
    assert str(diagnostic["record"]["valid_from"])[:10] == "2026-01-01"
    assert str(diagnostic["record"]["valid_to"])[:10] == "2035-12-31"
    assert all(diagnostic["conditions"].values()), diagnostic


def test_reel_v3_transport_factor_end_to_end() -> None:
    workbook = _workbook_path()
    result = BatchScenarioCompletionEngine("config").complete_batch_from_excel(workbook)
    critical = result.qa_report[
        result.qa_report["severity"].eq("Critical") & result.qa_report["status"].eq("FAIL")
    ]
    transport_failures = result.qa_report[
        result.qa_report["check_id"].isin(
            ["QA_MRV_RULE_ERROR", "TRANSPORT_GHG_BOUNDARY_MISMATCH"]
        )
        & result.qa_report["status"].eq("FAIL")
    ]
    review = result.completion_review
    non_base_work = review[
        review["scenario_code"].ne("REEL_BASE")
        & review["variable_name"].eq("transport_work_tkm")
    ].set_index("scenario_code")
    non_base_ghg = review[
        review["scenario_code"].ne("REEL_BASE")
        & review["variable_name"].eq("transport_ghg_tco2e")
    ].set_index("scenario_code")
    trace = result.rule_execution_trace[
        result.rule_execution_trace["target_variable"].eq("transport_ghg_tco2e")
    ]

    print("REEL_TRANSPORT_E2E")
    print(
        json.dumps(
            {
                "scenarios": len(result.scenario_results),
                "transport_work_tkm": sorted(non_base_work["completed_value"].unique().tolist()),
                "transport_ghg_tco2e": sorted(non_base_ghg["completed_value"].unique().tolist()),
                "rule_ids": sorted(non_base_ghg["rule_id"].unique().tolist()),
                "completion_levels": sorted(non_base_ghg["rule_level"].unique().tolist()),
                "factor_codes": sorted(trace["factor_codes"].unique().tolist()),
                "critical_failures": len(critical),
                "transport_failures": len(transport_failures),
            },
            indent=2,
            sort_keys=True,
        )
    )

    assert len(result.scenario_results) == 18
    assert len(non_base_work) == len(non_base_ghg) == 17
    assert non_base_work["completed_value"].astype(float).eq(21504.0).all()
    assert non_base_ghg["completed_value"].astype(float).apply(
        lambda value: value == pytest.approx(2.193408)
    ).all()
    assert non_base_ghg["rule_id"].eq("MRV_R_TRANSPORT_GHG_TKM").all()
    assert non_base_ghg["rule_level"].eq("L2").all()
    assert non_base_ghg["provenance"].str.contains(
        "factor_set_id=REEL_CASE_2026", regex=False
    ).all()
    assert non_base_ghg["provenance"].str.contains(
        "factor_code=TRANSPORT_GHG_PER_TKM", regex=False
    ).all()
    assert trace["factor_codes"].eq("TRANSPORT_GHG_PER_TKM").all()
    assert not non_base_ghg["rule_id"].str.contains("DES_VEHICLE_CO2", case=False).any()
    assert critical.empty
    assert transport_failures.empty


def test_reel_transport_factor_is_upserted_without_reseeding() -> None:
    db_engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=db_engine)
    TestSession = sessionmaker(bind=db_engine, future=True)
    register = pd.DataFrame(
        [
            {
                "factor_set_id": "REEL_CASE_2026",
                "factor_code": "TRANSPORT_GHG_PER_TKM",
                "factor_type": "EMISSION",
                "value": 0.000102,
                "unit": "tCO2e/tkm",
                "analytical_role": "Active analytical factor",
                "scope": "Outsourced road freight only / transport KPI; outside production Scope 1-2",
                "valid_from": "2026-01-01",
                "valid_to": "2035-12-31",
                "source": "first import",
                "approval_status": "Approved",
            }
        ]
    )

    with TestSession() as session:
        assert upsert_approved_emission_factors(session, register) == 1
        session.commit()
        first = session.query(EmissionFactor).one()
        first_id = first.id
        assert first.value == pytest.approx(0.000102)
        assert first.factor_set_id == "REEL_CASE_2026"

        register.loc[0, "source"] = "repeat import"
        assert upsert_approved_emission_factors(session, register) == 1
        session.commit()
        rows = session.query(EmissionFactor).all()
        assert len(rows) == 1
        assert rows[0].id == first_id
        assert rows[0].source == "repeat import"
