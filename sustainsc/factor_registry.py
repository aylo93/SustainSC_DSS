"""Validated emission-factor persistence for completed MRV workbooks."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from .models import EmissionFactor


REQUIRED_FACTOR_COLUMNS = {
    "factor_set_id",
    "factor_code",
    "factor_type",
    "value",
    "unit",
    "analytical_role",
    "scope",
    "valid_from",
    "valid_to",
    "source",
    "approval_status",
}


def upsert_approved_emission_factors(session: Session, factor_register: pd.DataFrame) -> int:
    """Insert or update approved EMISSION factors without relying on database seeding."""
    missing = REQUIRED_FACTOR_COLUMNS - set(factor_register.columns)
    if missing:
        raise ValueError(f"Missing factor-register columns: {sorted(missing)}")

    upserted = 0
    for row in factor_register.to_dict("records"):
        if str(row.get("approval_status", "")).strip().lower() != "approved":
            continue
        if str(row.get("factor_type", "")).strip().upper() != "EMISSION":
            continue

        code = str(row.get("factor_code", "")).strip()
        factor_set_id = str(row.get("factor_set_id", "")).strip()
        if not code or not factor_set_id:
            raise ValueError("Approved emission factors require factor_set_id and factor_code")

        valid_from = pd.to_datetime(row.get("valid_from"), errors="coerce")
        valid_to = pd.to_datetime(row.get("valid_to"), errors="coerce")
        values = {
            "name": code,
            "activity_type": str(row.get("scope") or code).strip(),
            "unit": str(row.get("unit", "")).strip(),
            "value": float(row["value"]),
            "valid_from": None if pd.isna(valid_from) else valid_from.to_pydatetime(),
            "valid_to": None if pd.isna(valid_to) else valid_to.to_pydatetime(),
            "source": str(row.get("source", "")).strip(),
            "analytical_role": str(row.get("analytical_role", "")).strip(),
            "factor_set_id": factor_set_id,
        }
        factor = session.query(EmissionFactor).filter_by(code=code).first()
        if factor is None:
            session.add(EmissionFactor(code=code, **values))
        else:
            for field, value in values.items():
                setattr(factor, field, value)
        upserted += 1

    session.flush()
    return upserted
