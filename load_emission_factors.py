"""Load an explicit emission-factor register without ambiguous category selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sustainsc.config import SessionLocal
from sustainsc.dataset_scope import ensure_dataset_schema
from sustainsc.kpi_engine import ANALYTICAL_FACTOR_CODES
from sustainsc.models import EmissionFactor


def load_emission_factors_file(path: str | Path, factor_set_id: str = "imported") -> int:
    source = Path(path)
    frame = pd.read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"factor_code", "value", "unit", "analytical_role"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing emission-factor columns: {sorted(missing)}")
    code_to_activity = {code: activity for activity, code in ANALYTICAL_FACTOR_CODES.items()}
    ensure_dataset_schema()
    with SessionLocal() as session:
        count = 0
        for row in frame.to_dict("records"):
            code = str(row["factor_code"]).strip()
            factor = session.query(EmissionFactor).filter_by(code=code).first()
            if factor is None:
                factor = EmissionFactor(code=code)
                session.add(factor)
            factor.name = code
            factor.activity_type = code_to_activity.get(code, "reference_only")
            factor.value = float(row["value"])
            factor.unit = str(row["unit"]).strip()
            factor.analytical_role = str(row["analytical_role"]).strip()
            factor.factor_set_id = factor_set_id
            factor.source = source.name
            count += 1
        session.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--factor-set-id", default="imported")
    args = parser.parse_args()
    count = load_emission_factors_file(args.csv_path, args.factor_set_id)
    print(f"Loaded {count} emission factors.")


if __name__ == "__main__":
    main()
