# sustainsc/vsmc.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import Scenario, Measurement, EmissionFactor, CostFactor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VSM_CSV = DATA_DIR / "vsm_steps.csv"


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s).strip())


def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def get_or_create_scenario(session: Session, code: str, name: str, description: str = "", notes: str = "") -> int:
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


def select_valid_factor(session: Session, model_cls, activity_type: str, ts: datetime):
    q = session.query(model_cls).filter(model_cls.activity_type == activity_type)
    q = q.filter(
        and_(
            (model_cls.valid_from.is_(None) | (model_cls.valid_from <= ts)),
            (model_cls.valid_to.is_(None) | (model_cls.valid_to >= ts)),
        )
    )
    q = q.order_by(model_cls.valid_from.desc().nullslast(), model_cls.id.desc())
    return q.first()


@dataclass
class VSMTotals:
    total_cycle_min: float
    total_wait_min: float
    total_lead_min: float
    va_ratio_pct: float
    total_output_ton: float
    total_electricity_kwh: float
    total_diesel_kwh: float
    total_emissions_tco2e: float
    emissions_intensity_kg_per_ton: Optional[float]
    total_cost_eur: Optional[float]


def compute_vsmc_for_scenario(session: Session, df: pd.DataFrame, scenario_code: str) -> Dict[str, Any]:
    d = df[df["scenario_code"] == scenario_code].copy()
    if d.empty:
        return {"scenario_code": scenario_code, "totals": None, "by_step": pd.DataFrame(), "timestamp": datetime.utcnow()}

    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    ts = d["timestamp"].max()
    if pd.isna(ts):
        ts = datetime.utcnow()
    else:
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

    # Factors: kgCO2e per kWh
    ef_elec = select_valid_factor(session, EmissionFactor, "electricity_kwh", ts)
    ef_diesel = select_valid_factor(session, EmissionFactor, "diesel_kwh", ts)
    elec_ef = float(ef_elec.value) if ef_elec else 0.0
    diesel_ef = float(ef_diesel.value) if ef_diesel else 0.0

    # Cost factors: EUR per kWh (optional)
    cf_elec = select_valid_factor(session, CostFactor, "electricity_kwh", ts)
    cf_diesel = select_valid_factor(session, CostFactor, "diesel_kwh", ts)
    elec_cf = float(cf_elec.value) if cf_elec else None
    diesel_cf = float(cf_diesel.value) if cf_diesel else None

    for c in ["cycle_time_min", "wait_time_min", "output_ton", "energy_kwh", "diesel_kwh"]:
        d[c] = d[c].apply(_safe_float)

    d["emissions_kg"] = d["energy_kwh"] * elec_ef + d["diesel_kwh"] * diesel_ef
    d["emissions_tco2e"] = d["emissions_kg"] / 1000.0

    if elec_cf is not None and diesel_cf is not None:
        d["cost_eur"] = d["energy_kwh"] * elec_cf + d["diesel_kwh"] * diesel_cf
        total_cost = float(d["cost_eur"].sum())
    else:
        d["cost_eur"] = pd.NA
        total_cost = None

    total_cycle = float(d["cycle_time_min"].sum())
    total_wait = float(d["wait_time_min"].sum())
    total_lead = total_cycle + total_wait
    va_ratio = (total_cycle / total_lead * 100.0) if total_lead > 0 else 0.0

    total_output = float(d["output_ton"].sum())
    total_elec = float(d["energy_kwh"].sum())
    total_diesel = float(d["diesel_kwh"].sum())
    total_em_t = float(d["emissions_tco2e"].sum())

    intensity = None
    if total_output > 0:
        intensity = (total_em_t * 1000.0) / total_output  # kgCO2e/ton

    totals = VSMTotals(
        total_cycle_min=total_cycle,
        total_wait_min=total_wait,
        total_lead_min=total_lead,
        va_ratio_pct=va_ratio,
        total_output_ton=total_output,
        total_electricity_kwh=total_elec,
        total_diesel_kwh=total_diesel,
        total_emissions_tco2e=total_em_t,
        emissions_intensity_kg_per_ton=intensity,
        total_cost_eur=total_cost,
    )

    by_step = d[[
        "plant_code", "step_code", "step_name",
        "cycle_time_min", "wait_time_min", "output_ton",
        "energy_kwh", "diesel_kwh", "emissions_tco2e", "cost_eur"
    ]].copy()

    return {"scenario_code": scenario_code, "totals": totals, "by_step": by_step, "timestamp": ts}


def delete_measurements_by_prefix(session: Session, scenario_id: int, prefix: str) -> None:
    session.query(Measurement).filter(
        Measurement.scenario_id == scenario_id,
        Measurement.variable_name.like(f"{prefix}%")
    ).delete(synchronize_session=False)


def delete_all_measurements(session: Session, scenario_id: int) -> None:
    session.query(Measurement).filter(Measurement.scenario_id == scenario_id).delete(synchronize_session=False)


def add_measurement(session: Session, scenario_id: int, var: str, value: float, unit: str, ts: datetime, comment: str) -> None:
    session.add(Measurement(
        variable_name=var,
        value=float(value),
        unit=unit,
        timestamp=ts,
        scenario_id=scenario_id,
        source_system="VSM-C",
        comment=comment,
    ))


def write_vsmc_diagnostics(session: Session, scenario_code: str) -> None:
    if not VSM_CSV.exists():
        raise FileNotFoundError(f"Missing {VSM_CSV}. Create data/vsm_steps.csv first.")

    df = pd.read_csv(VSM_CSV)
    required = {"scenario_code","plant_code","step_code","step_name","cycle_time_min","wait_time_min","output_ton","energy_kwh","diesel_kwh","timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"vsm_steps.csv missing columns: {sorted(missing)}")

    res = compute_vsmc_for_scenario(session, df, scenario_code)
    totals: Optional[VSMTotals] = res["totals"]
    by_step: pd.DataFrame = res["by_step"]
    ts: datetime = res["timestamp"]

    scenario_id = get_or_create_scenario(
        session,
        scenario_code,
        name=f"{scenario_code}",
        description="Scenario (VSM-C diagnostics available in measurements)",
        notes="",
    )

    # Solo borra mediciones derivadas VSM para no borrar MRV/KPIs
    delete_measurements_by_prefix(session, scenario_id, "vsm_")

    if totals is None:
        session.commit()
        return

    # Totales VSM-C
    add_measurement(session, scenario_id, "vsm_total_cycle_time_min", totals.total_cycle_min, "min", ts, "VSM total cycle time")
    add_measurement(session, scenario_id, "vsm_total_wait_time_min", totals.total_wait_min, "min", ts, "VSM total waiting time")
    add_measurement(session, scenario_id, "vsm_total_lead_time_min", totals.total_lead_min, "min", ts, "VSM total lead time")
    add_measurement(session, scenario_id, "vsm_va_ratio_pct", totals.va_ratio_pct, "%", ts, "Value-added ratio proxy")
    add_measurement(session, scenario_id, "vsm_total_emissions_tco2e", totals.total_emissions_tco2e, "tCO2e", ts, "Total VSM emissions")
    if totals.emissions_intensity_kg_per_ton is not None:
        add_measurement(session, scenario_id, "vsm_emissions_intensity_kg_per_ton", totals.emissions_intensity_kg_per_ton, "kgCO2e/ton", ts, "Emissions intensity")
    if totals.total_cost_eur is not None:
        add_measurement(session, scenario_id, "vsm_total_cost_eur", totals.total_cost_eur, "EUR", ts, "Total energy+diesel cost (from cost factors)")

    # Por paso
    for _, r in by_step.iterrows():
        step_code = str(r["step_code"])
        plant = str(r["plant_code"])
        add_measurement(session, scenario_id, f"vsm_step_emissions_tco2e::{step_code}", _safe_float(r["emissions_tco2e"]), "tCO2e", ts, f"Step emissions ({plant})")
        add_measurement(session, scenario_id, f"vsm_step_cycle_time_min::{step_code}", _safe_float(r["cycle_time_min"]), "min", ts, f"Step cycle time ({plant})")
        add_measurement(session, scenario_id, f"vsm_step_wait_time_min::{step_code}", _safe_float(r["wait_time_min"]), "min", ts, f"Step wait time ({plant})")
        if not pd.isna(r.get("cost_eur", pd.NA)):
            add_measurement(session, scenario_id, f"vsm_step_cost_eur::{step_code}", float(r["cost_eur"]), "EUR", ts, f"Step cost ({plant})")

    session.commit()


# -------------------------
# KAIZEN scenario generator
# -------------------------

DEFAULT_STEP_MULTIPLIERS = {
    # Mejora Lean típica (ajústalo a tu tesis)
    "cycle_time_min": 0.90,   # -10%
    "wait_time_min":  0.70,   # -30%
    "energy_kwh":     0.92,   # -8%
    "diesel_kwh":     0.88,   # -12%
    "output_ton":     1.00,   # igual (o 1.03 si quieres +3%)
}

DEFAULT_MRV_MULTIPLIERS = {
    # Variables MRV que sí impactan KPIs
    "water_withdrawn_m3": 0.97,
    "waste_generated_t":  0.95,
    "waste_recovered_t":  1.05,
    "transport_work_tkm": 0.98,
    "logistics_cost_eur": 0.98,
}


def create_kaizen_from_base(
    session: Session,
    base_code: str = "BASE",
    new_code: str = "VSMC_KAIZEN_01",
    step_multipliers: Optional[Dict[str, float]] = None,
    mrv_multipliers: Optional[Dict[str, float]] = None,
    co2_cap_tco2e: Optional[float] = None,
) -> None:
    """
    1) Lee VSM steps de BASE
    2) Aplica multiplicadores (tiempo/energía/diesel)
    3) Crea un NUEVO escenario new_code
    4) Copia snapshot MRV desde BASE y sobre-escribe electricidad_kwh / diesel_kwh / output_qty_fu con totales VSM Kaizen
    5) Escribe diagnósticos VSM-C (vsm_*) para el nuevo escenario
    """

    if not VSM_CSV.exists():
        raise FileNotFoundError(f"Missing {VSM_CSV}. Create data/vsm_steps.csv first.")

    df = pd.read_csv(VSM_CSV)
    base_steps = df[df["scenario_code"] == base_code].copy()
    if base_steps.empty:
        raise ValueError(f"No rows for scenario_code={base_code} in vsm_steps.csv")

    base_steps["timestamp"] = pd.to_datetime(base_steps["timestamp"], errors="coerce")
    ts = base_steps["timestamp"].max()
    if pd.isna(ts):
        ts = datetime.utcnow()
    else:
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

    sm = dict(DEFAULT_STEP_MULTIPLIERS)
    if step_multipliers:
        sm.update(step_multipliers)

    mm = dict(DEFAULT_MRV_MULTIPLIERS)
    if mrv_multipliers:
        mm.update(mrv_multipliers)

    # Build Kaizen steps (in-memory)
    kaizen_steps = base_steps.copy()
    kaizen_steps["scenario_code"] = new_code
    for col, mult in sm.items():
        if col in kaizen_steps.columns:
            kaizen_steps[col] = kaizen_steps[col].apply(_safe_float) * float(mult)

    # Compute totals from Kaizen steps
    # emissions need factors
    ef_elec = select_valid_factor(session, EmissionFactor, "electricity_kwh", ts)
    ef_diesel = select_valid_factor(session, EmissionFactor, "diesel_kwh", ts)
    elec_ef = float(ef_elec.value) if ef_elec else 0.0
    diesel_ef = float(ef_diesel.value) if ef_diesel else 0.0

    tot_output = float(kaizen_steps["output_ton"].apply(_safe_float).sum())
    tot_elec = float(kaizen_steps["energy_kwh"].apply(_safe_float).sum())
    tot_diesel = float(kaizen_steps["diesel_kwh"].apply(_safe_float).sum())
    tot_em_t = (tot_elec * elec_ef + tot_diesel * diesel_ef) / 1000.0

    # Optional: simple CO2 cap scaling (reduce energy+diesel proportionally)
    if co2_cap_tco2e is not None and tot_em_t > 0 and tot_em_t > float(co2_cap_tco2e):
        scale = float(co2_cap_tco2e) / float(tot_em_t)
        kaizen_steps["energy_kwh"] = kaizen_steps["energy_kwh"].apply(_safe_float) * scale
        kaizen_steps["diesel_kwh"] = kaizen_steps["diesel_kwh"].apply(_safe_float) * scale
        tot_elec = float(kaizen_steps["energy_kwh"].sum())
        tot_diesel = float(kaizen_steps["diesel_kwh"].sum())
        tot_em_t = (tot_elec * elec_ef + tot_diesel * diesel_ef) / 1000.0

    # Ensure new scenario exists + overwrite measurements for it
    new_scenario_id = get_or_create_scenario(
        session,
        new_code,
        name="VSM-C Kaizen scenario",
        description="Auto-generated improvement scenario from VSM-C levers.",
        notes=f"Base={base_code}; step_multipliers={sm}; mrv_multipliers={mm}; co2_cap={co2_cap_tco2e}",
    )

    # Copy MRV snapshot from BASE
    base_id = get_scenario_id(session, base_code)
    if base_id is None:
        raise ValueError(f"Base scenario {base_code} not found in DB")

    snapshot = latest_snapshot_by_variable(session, base_id)

    # Clear existing measurements in Kaizen scenario (full rebuild)
    delete_all_measurements(session, new_scenario_id)

    # 1) Insert copied MRV measurements (same timestamp)
    for var, m in snapshot.items():
        val = float(m.value)
        unit = m.unit or ""
        # Apply generic MRV multipliers if variable is in dict
        if var in mm:
            val = val * float(mm[var])
        session.add(Measurement(
            variable_name=var,
            value=float(val),
            unit=unit,
            timestamp=ts,
            scenario_id=new_scenario_id,
            source_system="VSM-C",
            comment=f"Copied from {base_code} (Kaizen multipliers applied where relevant)",
        ))

    session.flush()

    # 2) Override core KPI driver variables with Kaizen totals from VSM steps
    # (FU en tu tesis = toneladas; mantenemos unit como FU por compatibilidad con engine)
    overrides = [
        ("output_qty_fu", tot_output, "FU"),
        ("electricity_kwh", tot_elec, "kWh"),
        ("diesel_kwh", tot_diesel, "kWh"),
    ]
    for var, val, unit in overrides:
        session.add(Measurement(
            variable_name=var,
            value=float(val),
            unit=unit,
            timestamp=ts,
            scenario_id=new_scenario_id,
            source_system="VSM-C",
            comment="Overridden from VSM-C Kaizen totals",
        ))

    session.commit()

    # 3) Escribe diagnósticos VSM-C (vsm_*) para el nuevo escenario usando df combinado (base + kaizen)
    df2 = pd.concat([base_steps, kaizen_steps], ignore_index=True)
    res = compute_vsmc_for_scenario(session, df2, new_code)
    # Re-usa writer: simulamos escribiendo desde df2 sin tocar CSV
    # (Solución simple: escribir directamente vsm_*)
    delete_measurements_by_prefix(session, new_scenario_id, "vsm_")
    totals = res["totals"]
    by_step = res["by_step"]
    if totals:
        add_measurement(session, new_scenario_id, "vsm_total_cycle_time_min", totals.total_cycle_min, "min", ts, "VSM total cycle time")
        add_measurement(session, new_scenario_id, "vsm_total_wait_time_min", totals.total_wait_min, "min", ts, "VSM total waiting time")
        add_measurement(session, new_scenario_id, "vsm_total_lead_time_min", totals.total_lead_min, "min", ts, "VSM total lead time")
        add_measurement(session, new_scenario_id, "vsm_va_ratio_pct", totals.va_ratio_pct, "%", ts, "Value-added ratio proxy")
        add_measurement(session, new_scenario_id, "vsm_total_emissions_tco2e", totals.total_emissions_tco2e, "tCO2e", ts, "Total VSM emissions")
        if totals.emissions_intensity_kg_per_ton is not None:
            add_measurement(session, new_scenario_id, "vsm_emissions_intensity_kg_per_ton", totals.emissions_intensity_kg_per_ton, "kgCO2e/ton", ts, "Emissions intensity")

        for _, r in by_step.iterrows():
            step_code = str(r["step_code"])
            plant = str(r["plant_code"])
            add_measurement(session, new_scenario_id, f"vsm_step_emissions_tco2e::{step_code}", _safe_float(r["emissions_tco2e"]), "tCO2e", ts, f"Step emissions ({plant})")
            add_measurement(session, new_scenario_id, f"vsm_step_cycle_time_min::{step_code}", _safe_float(r["cycle_time_min"]), "min", ts, f"Step cycle time ({plant})")
            add_measurement(session, new_scenario_id, f"vsm_step_wait_time_min::{step_code}", _safe_float(r["wait_time_min"]), "min", ts, f"Step wait time ({plant})")

    session.commit()


def run_all_from_csv() -> None:
    if not VSM_CSV.exists():
        print(f"[WARN] {VSM_CSV} not found. Skipping VSM-C.")
        return

    session = SessionLocal()
    try:
        df = pd.read_csv(VSM_CSV)
        codes = sorted(set(df["scenario_code"].astype(str).tolist()))
        for sc in codes:
            write_vsmc_diagnostics(session, sc)
        print(f"VSM-C diagnostics written for: {codes}")
    finally:
        session.close()


def main(kaizen: bool = False, base_code: str = "BASE", new_code: str = "VSMC_KAIZEN_01", co2_cap: Optional[float] = None) -> None:
    # 1) write diagnostics for all scenarios present in vsm_steps.csv
    run_all_from_csv()

    # 2) optionally create Kaizen scenario
    if kaizen:
        session = SessionLocal()
        try:
            create_kaizen_from_base(session, base_code=base_code, new_code=new_code, co2_cap_tco2e=co2_cap)
            print(f"Kaizen scenario created: {new_code} (from {base_code})")
        finally:
            session.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--kaizen", action="store_true", help="Create VSM-C improvement scenario from BASE")
    p.add_argument("--base", default="BASE")
    p.add_argument("--code", default="VSMC_KAIZEN_01")
    p.add_argument("--co2cap", type=float, default=None, help="Optional CO2 cap (tCO2e) for Kaizen scaling")
    args = p.parse_args()
    main(kaizen=args.kaizen, base_code=args.base, new_code=args.code, co2_cap=args.co2cap)
