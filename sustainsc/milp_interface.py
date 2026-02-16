# sustainsc/milp_interface.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import Scenario, Measurement, KPIResult


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def measurement_has_column(colname: str) -> bool:
    return colname in Measurement.__table__.columns.keys()


def get_or_create_scenario(
    session: Session,
    code: str,
    name: str,
    description: str = "",
    notes: str = "",
) -> int:
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
    One Measurement per variable_name (latest timestamp) for a scenario.
    """
    rows = session.query(Measurement).filter_by(scenario_id=scenario_id).all()
    out: Dict[str, Measurement] = {}
    for m in rows:
        key = m.variable_name
        if key not in out:
            out[key] = m
        else:
            if (m.timestamp or datetime.min) > (out[key].timestamp or datetime.min):
                out[key] = m
    return out


def write_scenario_from_snapshot(
    session: Session,
    base_code: str,
    new_code: str,
    new_name: str,
    overrides: Dict[str, Tuple[float, str]],
    description: str = "",
    notes: str = "",
    delete_previous: bool = True,
    timestamp: Optional[datetime] = None,
) -> int:
    """
    1) Create/update new scenario
    2) Copy latest snapshot from BASE scenario
    3) Apply overrides (value, unit)
    """
    base_id = get_scenario_id(session, base_code)
    if base_id is None:
        raise RuntimeError(f"BASE scenario '{base_code}' not found. Load example data first.")

    new_id = get_or_create_scenario(session, new_code, new_name, description=description, notes=notes)

    if delete_previous:
        session.query(Measurement).filter_by(scenario_id=new_id).delete(synchronize_session=False)
        session.query(KPIResult).filter_by(scenario_id=new_id).delete(synchronize_session=False)
        session.flush()

    snap = latest_snapshot_by_variable(session, base_id)
    if not snap:
        raise RuntimeError(f"No measurements found in scenario '{base_code}'. Nothing to copy.")

    ts = timestamp or utc_now_naive()

    for var, m in snap.items():
        if m.value is None:
            continue

        val = float(m.value)
        unit = (m.unit or "").strip()

        if var in overrides:
            ov_val, ov_unit = overrides[var]
            val = float(ov_val)
            if ov_unit:
                unit = ov_unit

        # 👇 construimos kwargs solo con columnas existentes
        kwargs = dict(
            variable_name=var,
            value=float(val),
            unit=unit,
            timestamp=ts,
            scenario_id=new_id,
        )

        if measurement_has_column("source_system"):
            kwargs["source_system"] = "milp_adapter"
        if measurement_has_column("comment"):
            kwargs["comment"] = f"Copied from {base_code} + overrides"

        session.add(Measurement(**kwargs))

    session.flush()
    return new_id


def _get_base_value(snap: Dict[str, Measurement], var: str) -> Optional[float]:
    m = snap.get(var)
    if not m or m.value is None:
        return None
    return float(m.value)


def register_demo_milp_scenarios(base_code: str = "BASE") -> None:
    """
    Demo MILP scenarios (NO solver): creates three new scenarios by copying BASE and applying overrides.
    """
    session = SessionLocal()
    try:
        base_id = get_scenario_id(session, base_code)
        if base_id is None:
            raise RuntimeError(f"Scenario '{base_code}' does not exist in DB.")

        snap = latest_snapshot_by_variable(session, base_id)
        ts = utc_now_naive()

        # base values (safe defaults)
        elec = _get_base_value(snap, "electricity_kwh") or 0.0
        ren = _get_base_value(snap, "renewable_energy_kwh") or 0.0
        diesel = _get_base_value(snap, "diesel_kwh") or 0.0
        op_cost = _get_base_value(snap, "operating_cost_eur") or 0.0
        fuel_cost = _get_base_value(snap, "fuel_cost_eur") or 0.0
        elec_cost = _get_base_value(snap, "electricity_cost_eur") or 0.0

        overrides_min_cost = {
            "operating_cost_eur": (op_cost * 0.95, "EUR"),
            "fuel_cost_eur": (fuel_cost * 0.90, "EUR"),
            "electricity_cost_eur": (elec_cost * 0.97, "EUR"),
            "diesel_kwh": (diesel * 1.05, "kWh"),
            "renewable_energy_kwh": (ren * 0.70, "kWh"),
            "electricity_kwh": (elec * 1.02, "kWh"),
        }

        overrides_min_co2 = {
            "operating_cost_eur": (op_cost * 1.03, "EUR"),
            "fuel_cost_eur": (fuel_cost * 0.75, "EUR"),
            "electricity_cost_eur": (elec_cost * 1.05, "EUR"),
            "diesel_kwh": (diesel * 0.70, "kWh"),
            "renewable_energy_kwh": (max(ren * 1.80, ren + 1.0), "kWh"),
            "electricity_kwh": (elec * 0.98, "kWh"),
        }

        overrides_cap_300 = {
            "operating_cost_eur": (op_cost * 1.06, "EUR"),
            "fuel_cost_eur": (fuel_cost * 0.65, "EUR"),
            "diesel_kwh": (diesel * 0.60, "kWh"),
            "renewable_energy_kwh": (max(ren * 2.00, ren + 1.0), "kWh"),
            "electricity_kwh": (elec * 0.95, "kWh"),
        }

        write_scenario_from_snapshot(
            session, base_code, "MILP_MIN_COST", "MILP (demo): Minimize cost",
            overrides_min_cost, notes="Auto-created by milp_interface demo", timestamp=ts
        )
        write_scenario_from_snapshot(
            session, base_code, "MILP_MIN_CO2", "MILP (demo): Minimize CO2",
            overrides_min_co2, notes="Auto-created by milp_interface demo", timestamp=ts
        )
        write_scenario_from_snapshot(
            session, base_code, "MILP_CO2CAP_300", "MILP (demo): CO2 cap 300",
            overrides_cap_300, notes="Auto-created by milp_interface demo", timestamp=ts
        )

        session.commit()
        print("MILP demo scenarios created: MILP_MIN_COST, MILP_MIN_CO2, MILP_CO2CAP_300")

    finally:
        session.close()


if __name__ == "__main__":
    register_demo_milp_scenarios()
