from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sustainsc.config import Base
from sustainsc.dataset_scope import (
    activate_import_run,
    assert_scenario_integrity,
    audit_dataset,
    get_import_run_scenario_codes,
)
from sustainsc.kpi_engine import (
    compute_total_ghg_tco2e_from_factors,
    get_factor_by_code,
    select_valid_emission_factor,
)
from sustainsc.models import (
    EmissionFactor,
    ImportRun,
    ImportRunScenario,
    Measurement,
    Scenario,
)
from sustainsc.mrv_validation import (
    canonicalize_common_mrv_units,
    select_common_mrv,
    validate_completed_mrv,
)


FIXTURES = Path(__file__).parent / "fixtures" / "cuba_final"


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db:
        yield db


def _create_run(session, name, codes):
    run = ImportRun(
        dataset_name=name,
        source_filename=f"{name}.csv",
        import_timestamp=datetime(2025, 1, 1),
        status="importing",
        scenario_count=len(codes),
        measurement_count=0,
        is_active=False,
    )
    session.add(run)
    session.flush()
    for code in codes:
        scenario = session.query(Scenario).filter_by(code=code).first()
        if scenario is None:
            scenario = Scenario(code=code, name=code)
            session.add(scenario)
            session.flush()
        session.add(ImportRunScenario(import_run_id=run.id, scenario_id=scenario.id))
    session.flush()
    return run


def test_active_run_isolates_previous_scenarios(session):
    old = _create_run(session, "demo", ["BASE", "DEMO_EXTRA"])
    activate_import_run(session, old)
    current = _create_run(session, "industrial", ["BASE", "CURRENT"])
    activate_import_run(session, current)
    assert_scenario_integrity(session, current.id, {"BASE", "CURRENT"})
    session.commit()

    assert get_import_run_scenario_codes(session) == ["BASE", "CURRENT"]
    assert old.is_active is False
    assert current.is_active is True
    audit = audit_dataset(session)
    assert audit.active_scenarios == ("BASE", "CURRENT")
    assert audit.inactive_scenarios == ("DEMO_EXTRA",)


def test_cuba_fixture_exact_active_population():
    expected = set(
        pd.read_csv(FIXTURES / "valid_scenarios_CUBA_24.csv")["scenario_code"]
        .astype(str)
        .str.strip()
    )
    measurements = pd.read_csv(
        FIXTURES / "measurements_FINAL_CH7_ANNUAL_KPI_ALIGNED_MILP_CORRECTED.csv"
    )
    actual = set(measurements["scenario_code"].astype(str).str.strip())
    assert actual == expected
    assert len(actual) == 24
    assert {
        "CIRC_PUSH",
        "INTEGRATED",
        "ENERGY_PUSH",
        "MRV_PUSH",
        "MAINT_PUSH",
    }.isdisjoint(actual)
    assert "MILP_CO2CAP" in actual
    assert "MILP_CO2CAP_940" not in actual
    common = select_common_mrv(
        measurements, dictionary_path=Path("config") / "mrv_dictionary.csv"
    )
    common = canonicalize_common_mrv_units(
        common, dictionary_path=Path("config") / "mrv_dictionary.csv"
    )
    validation = validate_completed_mrv(
        common, dictionary_path=Path("config") / "mrv_dictionary.csv"
    )
    assert validation.required_variable_count == 107
    assert validation.scenario_count == 24
    assert len(common) == 2568


def test_analytical_factor_selection_is_explicit_and_deterministic(session):
    session.add_all(
        [
            EmissionFactor(
                code="EF_GRID_LOCATION_REFERENCE", name="reference",
                activity_type="electricity_kwh", unit="kgCO2e/kWh", value=0.817,
                analytical_role="Reference only",
            ),
            EmissionFactor(
                code="EF_ELECTRICITY_CASE", name="case",
                activity_type="electricity_kwh", unit="kgCO2e/kWh", value=0.6161,
                analytical_role="Active analytical factor",
            ),
            EmissionFactor(
                code="EF_DIESEL", name="diesel",
                activity_type="diesel_kwh", unit="kgCO2e/kWh", value=0.2668,
                analytical_role="Active analytical factor",
            ),
        ]
    )
    scenario = Scenario(code="S", name="S")
    session.add(scenario)
    session.flush()
    session.add_all(
        [
            Measurement(
                variable_name="electricity_kwh", value=1000, unit="kWh",
                timestamp=datetime(2025, 1, 1), scenario_id=scenario.id,
            ),
            Measurement(
                variable_name="diesel_kwh", value=500, unit="kWh",
                timestamp=datetime(2025, 1, 1), scenario_id=scenario.id,
            ),
        ]
    )
    session.commit()

    selected = select_valid_emission_factor(
        session, "electricity_kwh", datetime(2025, 1, 1)
    )
    assert selected.code == "EF_ELECTRICITY_CASE"
    assert get_factor_by_code(session, "EF_GRID_LOCATION_REFERENCE").analytical_role == "Reference only"
    assert compute_total_ghg_tco2e_from_factors(session, scenario.id) == pytest.approx(
        (1000 * 0.6161 + 500 * 0.2668) / 1000
    )


def test_corrected_milp_regression_values():
    measurements = pd.read_csv(
        FIXTURES / "measurements_FINAL_CH7_ANNUAL_KPI_ALIGNED_MILP_CORRECTED.csv"
    )
    totals = measurements[
        measurements["variable_name"].eq("ghg_total_s1s2")
    ].set_index("scenario_code")["value"]
    assert totals["MILP_MIN_COST"] == pytest.approx(744.001, abs=0.001)
    assert totals["MILP_CO2CAP"] == pytest.approx(741.300, abs=0.001)
    assert totals["MILP_MIN_CO2"] == pytest.approx(738.792, abs=0.001)


def test_obsolete_milp_bridge_is_inactive():
    bridges = pd.read_csv(Path("config") / "bridge_rules.csv")
    bridge = bridges.loc[bridges["bridge_rule_id"].eq("BR_MILP_GHG")].iloc[0]
    assert bridge["rule_status"] == "INACTIVE"


def test_dashboard_active_context_is_run_scoped():
    source = Path("kpi_dashboard.py").read_text(encoding="utf-8")
    assert "def load_active_context(import_run_id: int)" in source
    assert "WHERE import_run_id = :import_run_id" in source
    assert "SELECT COUNT(*) FROM sc_scenario" not in source
    assert "load_raw_kpi_results(active_import_run_id)" in source
    assert "load_normalized_results(active_import_run_id)" in source
