from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from sustainsc.config import SessionLocal
from sustainsc.models import (
    Scenario, Measurement, ImportRun, ImportRunScenario, EmissionFactor,
)
from sustainsc.dataset_scope import (
    activate_import_run,
    assert_scenario_integrity,
    ensure_dataset_schema,
    file_checksum,
    utc_now_naive,
)
from sustainsc.mrv_validation import (
    canonicalize_common_mrv_units,
    select_common_mrv,
    validate_completed_mrv,
)

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
    ensure_dataset_schema()

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

    dictionary_path = Path(__file__).resolve().parent / "config" / "mrv_dictionary.csv"
    df = select_common_mrv(df, dictionary_path=dictionary_path)
    df = canonicalize_common_mrv_units(df, dictionary_path=dictionary_path)
    validation = validate_completed_mrv(df, dictionary_path=dictionary_path)
    print(
        f"Validated {validation.scenario_count} scenarios x "
        f"{validation.required_variable_count} common MRV variables."
    )

    session = SessionLocal()
    import_run_id = None
    try:
        sc_map = ensure_scenarios(session, df)
        factor_sets = sorted(
            {
                value
                for (value,) in session.query(EmissionFactor.factor_set_id)
                .filter(EmissionFactor.factor_set_id.is_not(None))
                .all()
                if value
            }
        )
        import_run = ImportRun(
            dataset_name=path.stem,
            source_filename=path.name,
            import_timestamp=utc_now_naive(),
            status="importing",
            reference_scenario_code=(
                "BASE" if "BASE" in set(df["scenario_code"]) else None
            ),
            scenario_count=int(df["scenario_code"].nunique()),
            measurement_count=len(df),
            checksum=file_checksum(path),
            factor_set_id=",".join(factor_sets) or None,
            is_active=False,
        )
        session.add(import_run)
        session.flush()
        import_run_id = import_run.id
        for code in sorted(set(df["scenario_code"])):
            session.add(
                ImportRunScenario(import_run_id=import_run.id, scenario_id=sc_map[code])
            )
        session.flush()

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
                    import_run_id=import_run.id,
                    source_system="csv_monthly",
                    comment=f"loaded from {path.name}",
                    product_id=None,
                    facility_id=None,
                    process_id=None,
                    transport_leg_id=None,
                )
            )
            loaded += 1

        assert_scenario_integrity(session, import_run.id, set(df["scenario_code"]))
        activate_import_run(session, import_run)
        session.commit()
        print(
            f"Loaded {loaded} measurements from {path.name} "
            f"as active import run {import_run.id}"
        )

    finally:
        session.close()

    if args.run_pipeline:
        if run_full_pipeline is None:
            print("WARNING: run_full_pipeline not available. Measurements loaded, but KPIs were not recalculated.")
        else:
            print("Running KPI pipeline...")
            run_full_pipeline(debug_missing=True, import_run_id=import_run_id)


if __name__ == "__main__":
    main()
