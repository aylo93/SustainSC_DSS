# sustainsc/anylogistix_adapter.py
# Importa resultados de AnyLogistix/AnyLogic (CSV long) a sc_measurement,
# creando escenarios automáticamente y evitando duplicados.
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import SessionLocal
from .models import Scenario, Measurement


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_dt(x: object) -> datetime:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return utc_now_naive()
    s = str(x).strip()
    if not s:
        return utc_now_naive()
    # soporta "2025-12-01T00:00:00" y "2025-12-01 00:00:00"
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def get_or_create_scenario(session, code: str, name: Optional[str] = None,
                           description: str = "", notes: str = "") -> int:
    code = (code or "").strip()
    if not code:
        code = "ALX_UNK"

    sc = session.query(Scenario).filter_by(code=code).first()
    if sc:
        # opcional: refrescar metadata
        if name:
            sc.name = name
        if description:
            sc.description = description
        if notes:
            sc.notes = notes
        session.flush()
        return sc.id

    sc = Scenario(
        code=code,
        name=name or code,
        description=description or "",
        notes=notes or "",
    )
    session.add(sc)
    session.flush()
    return sc.id


def upsert_measurement(session, scenario_id: int, variable_name: str, ts: datetime,
                       value: float, unit: str = "", source_system: str = "AnyLogistix",
                       comment: str = "") -> None:
    variable_name = (variable_name or "").strip()
    if not variable_name:
        return

    # Dedupe simple: (scenario_id, variable_name, timestamp)
    existing = (
        session.query(Measurement)
        .filter_by(scenario_id=scenario_id, variable_name=variable_name, timestamp=ts)
        .first()
    )
    if existing:
        existing.value = float(value)
        existing.unit = unit or existing.unit
        existing.source_system = source_system or existing.source_system
        existing.comment = comment or existing.comment
        return

    session.add(
        Measurement(
            scenario_id=scenario_id,
            variable_name=variable_name,
            value=float(value),
            unit=unit or None,
            timestamp=ts,
            source_system=source_system or None,
            comment=comment or None,
        )
    )


@dataclass
class ImportStats:
    scenarios_touched: int = 0
    measurements_written: int = 0


def import_anylogistix_csv(
    csv_path: Path,
    scenario_prefix: str = "",
    source_system: str = "AnyLogistix",
) -> ImportStats:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"scenario_code", "variable_name", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # opcionales
    if "timestamp" not in df.columns:
        df["timestamp"] = utc_now_naive().isoformat()
    if "unit" not in df.columns:
        df["unit"] = ""
    if "comment" not in df.columns:
        df["comment"] = ""
    if "source_system" not in df.columns:
        df["source_system"] = source_system

    # normaliza
    df["scenario_code"] = df["scenario_code"].astype(str).str.strip()
    if scenario_prefix:
        df["scenario_code"] = scenario_prefix + df["scenario_code"]

    df["variable_name"] = df["variable_name"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["comment"] = df["comment"].astype(str)

    # limpia values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    session = SessionLocal()
    stats = ImportStats()
    try:
        touched = set()

        for _, r in df.iterrows():
            sc_code = r["scenario_code"]
            ts = parse_dt(r.get("timestamp"))
            var = r["variable_name"]
            val = float(r["value"])
            unit = r.get("unit", "") or ""
            comm = r.get("comment", "") or ""
            src = r.get("source_system", source_system) or source_system

            sc_id = get_or_create_scenario(
                session,
                code=sc_code,
                name=sc_code,  # puedes cambiar por un nombre más bonito si lo incluyes en el CSV
                description="Imported from AnyLogistix/AnyLogic results",
                notes="Auto-imported by anylogistix_adapter",
            )
            touched.add(sc_id)

            upsert_measurement(
                session,
                scenario_id=sc_id,
                variable_name=var,
                ts=ts,
                value=val,
                unit=unit,
                source_system=src,
                comment=comm,
            )
            stats.measurements_written += 1

        session.commit()
        stats.scenarios_touched = len(touched)
        return stats
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=str, help="Path to AnyLogistix results CSV (long format).")
    p.add_argument("--prefix", type=str, default="", help="Optional prefix for scenario_code, e.g. ALX_.")
    args = p.parse_args()

    st = import_anylogistix_csv(Path(args.csv), scenario_prefix=args.prefix)
    print(f"[OK] Scenarios touched: {st.scenarios_touched}")
    print(f"[OK] Measurements written: {st.measurements_written}")


if __name__ == "__main__":
    main()
