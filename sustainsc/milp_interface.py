# sustainsc/milp_interface.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import Scenario, Measurement


def measurement_has_column(col: str) -> bool:
    return col in Measurement.__table__.columns.keys()


def get_or_create_scenario(session: Session, code: str, name: str,
                           description: str = "", notes: str = "") -> int:
    sc = session.query(Scenario).filter_by(code=code).first()
    if sc:
        sc.name = name or sc.name
        sc.description = description or sc.description
        sc.notes = notes or sc.notes
        session.flush()
        return sc.id
    sc = Scenario(code=code, name=name, description=description, notes=notes)
    session.add(sc)
    session.flush()
    return sc.id


def get_scenario_id(session: Session, code: str) -> Optional[int]:
    sc = session.query(Scenario).filter_by(code=code).first()
    return sc.id if sc else None


def latest_snapshot_by_variable(session: Session, scenario_id: int) -> Dict[str, Measurement]:
    """
    One latest measurement per variable_name (latest timestamp) for scenario_id.
    If you later store by facility/product, you can extend the key to include those.
    """
    rows = session.query(Measurement).filter_by(scenario_id=scenario_id).all()
    out: Dict[str, Measurement] = {}
    for m in rows:
        k = m.variable_name
        if k not in out:
            out[k] = m
        else:
            if (m.timestamp or datetime.min) > (out[k].timestamp or datetime.min):
                out[k] = m
    return out


def write_scenario_from_snapshot(
    session: Session,
    base_code: str,
    new_code: str,
    new_name: str,
    overrides: Dict[str, Tuple[float, str]],
    timestamp: datetime,
    description: str = "",
    notes: str = "",
    delete_previous: bool = True,
) -> int:
    """
    Creates/updates scenario `new_code`.
    Copies latest snapshot from BASE into new scenario, then applies overrides.
    """
    base_id = get_scenario_id(session, base_code)
    if base_id is None:
        raise RuntimeError(f"Base scenario '{base_code}' not found. Load scenarios first.")

    new_id = get_or_create_scenario(session, new_code, new_name, description=description, notes=notes)

    if delete_previous:
        session.query(Measurement).filter_by(scenario_id=new_id).delete()

    snap = latest_snapshot_by_variable(session, base_id)

    # 1) copy snapshot
    for var, m in snap.items():
        kwargs = dict(
            variable_name=var,
            value=float(m.value),
            unit=m.unit,
            timestamp=timestamp,
            scenario_id=new_id,
            source_system="milp_adapter",
            comment=f"Copied from {base_code}",
        )

        # copy optional context if exists in model
        for fk in ("product_id", "facility_id", "process_id", "transport_leg_id"):
            if measurement_has_column(fk):
                kwargs[fk] = getattr(m, fk, None)

        session.add(Measurement(**kwargs))

    # 2) apply overrides (replace/add)
    for var, (val, unit) in overrides.items():
        kwargs = dict(
            variable_name=var,
            value=float(val),
            unit=unit,
            timestamp=timestamp,
            scenario_id=new_id,
            source_system="milp_adapter",
            comment=f"Copied from {base_code} + overrides",
        )
        session.add(Measurement(**kwargs))

    session.flush()
    return new_id


def register_demo_milp_scenarios(base_code: str = "BASE") -> None:
    """
    Demo scenarios (simulan salidas MILP):
    - MILP_MIN_COST: reduce costos totales (trade-off típico: CO2 puede subir o bajar según supuestos)
    - MILP_MIN_CO2 : reduce CO2 (más RES, menos diésel)
    - MILP_CO2CAP_300: respeta un cap (aquí lo representamos como mix energético más limpio)
    """
    ts = datetime.fromisoformat("2025-12-01T00:00:00")

    with SessionLocal() as session:
        # Min cost (ejemplo)
        overrides_min_cost = {
            "operating_cost_eur": (4_600_000, "EUR"),
            "logistics_cost_eur": (820_000, "EUR"),
            "fuel_cost_eur": (520_000, "EUR"),
            "electricity_cost_eur": (360_000, "EUR"),
            # ligera eficiencia
            "electricity_kwh": (930_000, "kWh"),
            "diesel_kwh": (430_000, "kWh"),
        }

        # Min CO2 (ejemplo)
        overrides_min_co2 = {
            "renewable_energy_kwh": (520_000, "kWh"),
            "electricity_kwh": (980_000, "kWh"),
            "diesel_kwh": (350_000, "kWh"),
            # costos pueden subir
            "electricity_cost_eur": (470_000, "EUR"),
            "fuel_cost_eur": (420_000, "EUR"),
            "operating_cost_eur": (5_100_000, "EUR"),
        }

        # CO2 cap (ejemplo “cap 300 tCO2e” representado como mix intermedio)
        overrides_cap = {
            "renewable_energy_kwh": (420_000, "kWh"),
            "electricity_kwh": (950_000, "kWh"),
            "diesel_kwh": (380_000, "kWh"),
            "operating_cost_eur": (4_950_000, "EUR"),
            "fuel_cost_eur": (460_000, "EUR"),
            "electricity_cost_eur": (430_000, "EUR"),
        }

        write_scenario_from_snapshot(
            session, base_code, "MILP_MIN_COST", "MILP (demo): Minimize cost",
            overrides_min_cost, timestamp=ts,
            notes="Auto-created by milp_interface demo",
        )
        write_scenario_from_snapshot(
            session, base_code, "MILP_MIN_CO2", "MILP (demo): Minimize CO2",
            overrides_min_co2, timestamp=ts,
            notes="Auto-created by milp_interface demo",
        )
        write_scenario_from_snapshot(
            session, base_code, "MILP_CO2CAP_300", "MILP (demo): CO2 cap = 300 tCO2e",
            overrides_cap, timestamp=ts,
            notes="Auto-created by milp_interface demo",
        )

        session.commit()
        print("MILP demo scenarios registered: MILP_MIN_COST, MILP_MIN_CO2, MILP_CO2CAP_300")


if __name__ == "__main__":
    register_demo_milp_scenarios()
