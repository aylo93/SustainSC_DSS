# load_measurements_only.py
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

from sustainsc.config import SessionLocal
from sustainsc.models import Scenario, Measurement

def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python load_measurements_only.py data/measurements_2025_10.csv")
        sys.exit(1)

    path = Path(sys.argv[1])
    df = read_csv(path)

    required = ["variable_name", "value", "unit", "timestamp", "scenario_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    session = SessionLocal()
    try:
        # map scenario_code -> scenario_id
        sc_map = {s.code: s.id for s in session.query(Scenario).all()}
        if "BASE" not in sc_map:
            raise RuntimeError("BASE scenario not found. Run load_example_data.py first.")

        # replace measurements
        session.query(Measurement).delete()
        session.commit()

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Invalid timestamp values in file.")

        for _, row in df.iterrows():
            scode = str(row["scenario_code"]).strip()
            sid = sc_map.get(scode, sc_map["BASE"])

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

        session.commit()
        print(f"Loaded {len(df)} measurements from {path.name}")

    finally:
        session.close()

if __name__ == "__main__":
    main()
