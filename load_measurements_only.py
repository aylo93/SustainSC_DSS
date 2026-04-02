from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from sustainsc.config import SessionLocal
from sustainsc.models import Scenario, Measurement

try:
    from sustainsc.kpi_engine import run_full_pipeline
except Exception:
    run_full_pipeline = None


REQUIRED_COLUMNS = ["variable_name", "value", "unit", "timestamp", "scenario_code"]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(how="all")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def ensure_scenarios(session, df: pd.DataFrame) -> Dict[str, int]:
    sc_map = {s.code: s.id for s in session.query(Scenario).all()}

    for scode in sorted(set(df["scenario_code"].astype(str).str.strip())):
        if not scode:
            continue
        if scode not in sc_map:
            sc = Scenario(
                code=scode,
                name=scode,
                description="auto-created from measurements loader",
                notes="created by load_measurements_only.py",
            )
            session.add(sc)
            session.flush()
            sc_map[scode] = sc.id

    return sc_map


def main():
    parser = argparse.ArgumentParser(description="Load measurements from a CSV file.")
    parser.add_argument("csv_path", help="Path to CSV file, e.g. data/measurements_2025_10.csv")
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="Delete all existing measurements before loading the new file.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run KPI engine + normalization + composite indices after loading.",
    )
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["variable_name"] = df["variable_name"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["scenario_code"] = df["scenario_code"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if df["value"].isna().any():
        bad = df[df["value"].isna()]
        raise ValueError(
            f"Invalid numeric values in 'value'. Bad rows: {bad.index.tolist()[:10]}"
        )

    if df["timestamp"].isna().any():
        bad = df[df["timestamp"].isna()]
        raise ValueError(
            f"Invalid timestamp values. Bad rows: {bad.index.tolist()[:10]}"
        )

    if (df["variable_name"] == "").any():
        raise ValueError("Some rows have empty variable_name.")

    if (df["unit"] == "").any():
        raise ValueError("Some rows have empty unit.")

    if (df["scenario_code"] == "").any():
        raise ValueError("Some rows have empty scenario_code.")

    session = SessionLocal()
    try:
        sc_map = ensure_scenarios(session, df)

        if args.replace_all:
            deleted = session.query(Measurement).delete()
            session.commit()
            print(f"Deleted {deleted} existing measurements.")

        loaded = 0
        for _, row in df.iterrows():
            sid = sc_map[str(row["scenario_code"]).strip()]

            session.add(
                Measurement(
                    variable_name=str(row["variable_name"]).strip(),
                    value=float(row["value"]),
                    unit=str(row["unit"]).strip(),
                    timestamp=row["timestamp"].to_pydatetime(),
                    scenario_id=sid,
                    source_system="csv_monthly",
                    comment=f"loaded from {path.name}",
                    product_id=None,
                    facility_id=None,
                    process_id=None,
                    transport_leg_id=None,
                )
            )
            loaded += 1

        session.commit()
        print(f"Loaded {loaded} measurements from {path.name}")

    finally:
        session.close()

    if args.run_pipeline:
        if run_full_pipeline is None:
            print("WARNING: run_full_pipeline not available. Measurements loaded, but KPIs were not recalculated.")
        else:
            print("Running KPI pipeline...")
            run_full_pipeline(debug_missing=True)