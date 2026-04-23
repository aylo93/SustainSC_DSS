from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

from sustainsc.config import SessionLocal
from sustainsc.models import Scenario, Product, Facility, ProductBatch


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(how="all")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def get_or_create_scenario(session, code: str):
    obj = session.query(Scenario).filter_by(code=code).first()
    if obj:
        return obj
    obj = Scenario(code=code, name=code, description="Auto-created by batch loader", notes="")
    session.add(obj)
    session.flush()
    return obj


def get_or_create_product(session, code: str, name: str | None = None, fu_unit: str | None = None):
    obj = session.query(Product).filter_by(code=code).first()
    if obj:
        return obj
    obj = Product(code=code, name=name or code, fu_unit=fu_unit, dpp_ref=None)
    session.add(obj)
    session.flush()
    return obj


def get_or_create_facility(session, code: str, name: str | None = None):
    obj = session.query(Facility).filter_by(code=code).first()
    if obj:
        return obj
    obj = Facility(code=code, name=name or code, location=None, facility_type=None)
    session.add(obj)
    session.flush()
    return obj


def load_product_batches_file(path: str | Path) -> int:
    path = Path(path)
    df = read_csv(path)

    required = ["batch_code", "product_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in product_batches.csv: {missing}")

    if "scenario_code" not in df.columns:
        df["scenario_code"] = "BASE"
    if "origin_facility_code" not in df.columns:
        df["origin_facility_code"] = None
    if "production_date" not in df.columns:
        df["production_date"] = None
    if "quantity" not in df.columns:
        df["quantity"] = None
    if "unit" not in df.columns:
        df["unit"] = None
    if "status" not in df.columns:
        df["status"] = None
    if "notes" not in df.columns:
        df["notes"] = None
    if "product_name" not in df.columns:
        df["product_name"] = None
    if "facility_name" not in df.columns:
        df["facility_name"] = None

    df["production_date"] = pd.to_datetime(df["production_date"], errors="coerce")

    session = SessionLocal()
    try:
        written = 0

        for _, row in df.iterrows():
            batch_code = str(row["batch_code"]).strip()
            product_code = str(row["product_code"]).strip()
            scenario_code = str(row["scenario_code"]).strip() if pd.notna(row["scenario_code"]) else "BASE"
            facility_code = str(row["origin_facility_code"]).strip() if pd.notna(row["origin_facility_code"]) and str(row["origin_facility_code"]).strip() else None

            if not batch_code or not product_code:
                continue

            scenario = get_or_create_scenario(session, scenario_code)
            product = get_or_create_product(
                session,
                product_code,
                name=(str(row["product_name"]).strip() if pd.notna(row["product_name"]) and str(row["product_name"]).strip() else product_code),
                fu_unit=(str(row["unit"]).strip() if pd.notna(row["unit"]) and str(row["unit"]).strip() else None),
            )
            facility = None
            if facility_code:
                facility = get_or_create_facility(
                    session,
                    facility_code,
                    name=(str(row["facility_name"]).strip() if pd.notna(row["facility_name"]) and str(row["facility_name"]).strip() else facility_code),
                )

            existing = session.query(ProductBatch).filter_by(batch_code=batch_code).first()
            if existing:
                existing.product_id = product.id
                existing.scenario_id = scenario.id
                existing.origin_facility_id = facility.id if facility else None
                existing.production_date = row["production_date"].to_pydatetime() if pd.notna(row["production_date"]) else None
                existing.quantity = float(row["quantity"]) if pd.notna(row["quantity"]) else None
                existing.unit = str(row["unit"]).strip() if pd.notna(row["unit"]) else None
                existing.status = str(row["status"]).strip() if pd.notna(row["status"]) else None
                existing.notes = str(row["notes"]).strip() if pd.notna(row["notes"]) else None
            else:
                session.add(
                    ProductBatch(
                        batch_code=batch_code,
                        product_id=product.id,
                        scenario_id=scenario.id,
                        origin_facility_id=facility.id if facility else None,
                        production_date=row["production_date"].to_pydatetime() if pd.notna(row["production_date"]) else None,
                        quantity=float(row["quantity"]) if pd.notna(row["quantity"]) else None,
                        unit=str(row["unit"]).strip() if pd.notna(row["unit"]) else None,
                        status=str(row["status"]).strip() if pd.notna(row["status"]) else None,
                        notes=str(row["notes"]).strip() if pd.notna(row["notes"]) else None,
                    )
                )

            written += 1

        session.commit()
        return written
    finally:
        session.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_product_batches.py data/product_batches.csv")
        sys.exit(1)

    written = load_product_batches_file(sys.argv[1])
    print(f"Loaded/updated {written} product batches from {sys.argv[1]}")


if __name__ == "__main__":
    main()