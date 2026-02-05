# load_example_data.py
# SustainSC DSS - Example data loader (Cloud-proof)
# Loads: scenarios, emission factors, cost factors, KPIs, measurements
# Robust against comma/semicolon/tab delimiters + UTF-8 BOM.

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

from sustainsc.config import SessionLocal
from sustainsc.models import (
    Scenario,
    EmissionFactor,
    CostFactor,
    KPI,
    Measurement,
    Product,
    Facility,
    Process,
    TransportLeg,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Preferred filenames (keep these in your repo under /data)
SCENARIOS_CSV = DATA_DIR / "scenarios.csv"
EMISSION_FACTORS_CSV = DATA_DIR / "emission_factors.csv"
COST_FACTORS_CSV = DATA_DIR / "cost_factors.csv"
KPIS_CSV = DATA_DIR / "kpis.csv"
MEASUREMENTS_CSV = DATA_DIR / "measurements.csv"


# -----------------------------
# Helpers
# -----------------------------

def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())

def ffloat(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    return float(s)

def norm_row(row: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        kk = (k or "").strip().lower().replace(" ", "_")
        out[kk] = (v if v is not None else "")
    return out

def detect_delimiter(sample: str) -> str:
    # Try Sniffer first, otherwise heuristic
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        return dialect.delimiter
    except csv.Error:
        counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
        delim = max(counts, key=counts.get)
        return delim if counts[delim] > 0 else ","

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)

        if not sample.strip():
            raise ValueError(f"CSV seems empty: {path}")

        delim = detect_delimiter(sample)
        print(f"[INFO] {path.name} delimiter: {repr(delim)}")

        reader = csv.DictReader(f, delimiter=delim, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"No headers detected in: {path}")

        rows: List[Dict[str, str]] = []
        for raw in reader:
            rows.append(norm_row(raw))
        print(f"[INFO] {path.name} headers: {reader.fieldnames}")
        return rows


# -----------------------------
# Loaders
# -----------------------------

def upsert_scenario(session, code: str, name: str, description: str = "", notes: str = "") -> int:
    code = code.strip()
    existing = session.query(Scenario).filter_by(code=code).first()
    if existing:
        existing.name = name or existing.name
        existing.description = description or existing.description
        existing.notes = notes or existing.notes
        session.flush()
        return existing.id

    sc = Scenario(code=code, name=name, description=description, notes=notes)
    session.add(sc)
    session.flush()
    return sc.id

def load_scenarios(session) -> Dict[str, int]:
    scenario_map: Dict[str, int] = {}

    if SCENARIOS_CSV.exists():
        rows = read_csv_rows(SCENARIOS_CSV)
        for row in rows:
            code = (row.get("code") or "").strip()
            if not code:
                continue
            sid = upsert_scenario(
                session,
                code=code,
                name=(row.get("name") or code).strip(),
                description=(row.get("description") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
            scenario_map[code] = sid
        session.commit()
        print(f"Loaded {len(scenario_map)} scenarios from scenarios.csv.")
    else:
        # Cloud-proof fallback: ensure BASE/S1/S2 exist even without scenarios.csv
        defaults = [
            ("BASE", "Baseline", "Reference scenario", "Default baseline"),
            ("S1", "Low-carbon energy mix", "More renewable energy", "Higher RES share"),
            ("S2", "Efficiency + digital", "Efficiency improvements", "Lower energy intensity"),
        ]
        for code, name, desc, notes in defaults:
            sid = upsert_scenario(session, code, name, desc, notes)
            scenario_map[code] = sid
        session.commit()
        print("scenarios.csv not found. Ensured default scenarios: BASE, S1, S2.")

    # Refresh map from DB (guarantee IDs)
    for sc in session.query(Scenario).all():
        scenario_map[sc.code] = sc.id

    return scenario_map

def load_emission_factors(session) -> None:
    if not EMISSION_FACTORS_CSV.exists():
        print(f"[WARN] Missing {EMISSION_FACTORS_CSV.name}; skipping emission factors.")
        return

    rows = read_csv_rows(EMISSION_FACTORS_CSV)
    count = 0

    for row in rows:
        activity_type = (row.get("activity_type") or row.get("variable_name") or "").strip()
        if not activity_type:
            continue

        name = (row.get("name") or row.get("factor_name") or activity_type).strip()
        unit = (row.get("unit") or "").strip()
        value = ffloat(row.get("value") or row.get("factor_value"))
        if value is None:
            continue

        valid_from = parse_dt(row["valid_from"]) if row.get("valid_from") else None
        valid_to = parse_dt(row["valid_to"]) if row.get("valid_to") else None
        source = (row.get("source") or "").strip()

        existing = (
            session.query(EmissionFactor)
            .filter_by(activity_type=activity_type, value=float(value), valid_from=valid_from, valid_to=valid_to)
            .first()
        )
        if existing:
            existing.name = name or existing.name
            existing.unit = unit or existing.unit
            existing.source = source or existing.source
        else:
            session.add(EmissionFactor(
                name=name,
                activity_type=activity_type,
                unit=unit,
                value=float(value),
                valid_from=valid_from,
                valid_to=valid_to,
                source=source,
            ))
        count += 1

    session.commit()
    print(f"Loaded {count} emission factors.")

def load_cost_factors(session) -> None:
    if not COST_FACTORS_CSV.exists():
        print(f"[WARN] Missing {COST_FACTORS_CSV.name}; skipping cost factors.")
        return

    rows = read_csv_rows(COST_FACTORS_CSV)
    count = 0

    for row in rows:
        activity_type = (row.get("activity_type") or row.get("variable_name") or "").strip()
        if not activity_type:
            continue

        name = (row.get("name") or row.get("factor_name") or activity_type).strip()
        unit = (row.get("unit") or "").strip()
        value = ffloat(row.get("value") or row.get("factor_value"))
        if value is None:
            continue

        valid_from = parse_dt(row["valid_from"]) if row.get("valid_from") else None
        valid_to = parse_dt(row["valid_to"]) if row.get("valid_to") else None
        source = (row.get("source") or "").strip()

        existing = (
            session.query(CostFactor)
            .filter_by(activity_type=activity_type, value=float(value), valid_from=valid_from, valid_to=valid_to)
            .first()
        )
        if existing:
            existing.name = name or existing.name
            existing.unit = unit or existing.unit
            existing.source = source or existing.source
        else:
            session.add(CostFactor(
                name=name,
                activity_type=activity_type,
                unit=unit,
                value=float(value),
                valid_from=valid_from,
                valid_to=valid_to,
                source=source,
            ))
        count += 1

    session.commit()
    print(f"Loaded {count} cost factors.")

def load_kpis(session) -> None:
    if not KPIS_CSV.exists():
        print(f"[WARN] Missing {KPIS_CSV.name}; cannot load KPIs.")
        return

    rows = read_csv_rows(KPIS_CSV)
    count = 0

    for row in rows:
        code = (row.get("code") or "").strip()
        if not code:
            continue

        is_benefit_raw = (row.get("is_benefit") or "0").strip().lower()
        is_benefit = is_benefit_raw in ("1", "true", "yes", "y")

        payload = dict(
            code=code,
            name=(row.get("name") or code).strip(),
            description=(row.get("description") or "").strip(),
            dimension=(row.get("dimension") or "").strip(),
            decision_level=(row.get("decision_level") or "").strip(),
            unit=(row.get("unit") or "").strip(),
            is_benefit=is_benefit,
            formula_id=(row.get("formula_id") or "").strip(),
            protocol_notes=(row.get("protocol_notes") or "").strip(),
        )

        flow = (row.get("flow") or "").strip()
        if hasattr(KPI, "flow") and flow:
            payload["flow"] = flow

        existing = session.query(KPI).filter_by(code=code).first()
        if existing:
            for k, v in payload.items():
                setattr(existing, k, v)
        else:
            session.add(KPI(**payload))

        count += 1

    session.commit()
    print(f"Loaded {count} KPIs.")

def _resolve_optional_fk_ids(session, row: Dict[str, str]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    product_id = facility_id = process_id = transport_leg_id = None

    pcode = (row.get("product_code") or "").strip()
    if pcode:
        p = session.query(Product).filter_by(code=pcode).first()
        product_id = p.id if p else None

    fcode = (row.get("facility_code") or "").strip()
    if fcode:
        fac = session.query(Facility).filter_by(code=fcode).first()
        facility_id = fac.id if fac else None

    prcode = (row.get("process_code") or "").strip()
    if prcode:
        pr = session.query(Process).filter_by(code=prcode).first()
        process_id = pr.id if pr else None

    tcode = (row.get("transport_leg_code") or "").strip()
    if tcode:
        leg = session.query(TransportLeg).filter_by(code=tcode).first()
        transport_leg_id = leg.id if leg else None

    return product_id, facility_id, process_id, transport_leg_id

def load_measurements(session, scenario_map: Dict[str, int]) -> None:
    if not MEASUREMENTS_CSV.exists():
        print(f"[WARN] Missing {MEASUREMENTS_CSV.name}; cannot load measurements.")
        return

    rows = read_csv_rows(MEASUREMENTS_CSV)
    count = 0

    for row in rows:
        variable_name = (row.get("variable_name") or row.get("activity_type") or "").strip()
        if not variable_name:
            continue

        value = ffloat(row.get("value"))
        unit = (row.get("unit") or "").strip()
        ts_raw = (row.get("timestamp") or "").strip()
        scenario_code = (row.get("scenario_code") or "BASE").strip()

        if value is None or not unit or not ts_raw:
            continue

        ts = parse_dt(ts_raw)

        scenario_id = scenario_map.get(scenario_code)
        if scenario_id is None:
            scenario_id = upsert_scenario(session, scenario_code, scenario_code, "", "auto-created")
            scenario_map[scenario_code] = scenario_id
            session.commit()

        product_id, facility_id, process_id, transport_leg_id = _resolve_optional_fk_ids(session, row)
        source_system = (row.get("source_system") or "").strip()
        comment = (row.get("comment") or "").strip()

        existing = (
            session.query(Measurement)
            .filter_by(variable_name=variable_name, timestamp=ts, scenario_id=scenario_id)
            .first()
        )

        if existing:
            existing.value = float(value)
            existing.unit = unit
            existing.product_id = product_id
            existing.facility_id = facility_id
            existing.process_id = process_id
            existing.transport_leg_id = transport_leg_id
            existing.source_system = source_system or existing.source_system
            existing.comment = comment or existing.comment
        else:
            session.add(Measurement(
                variable_name=variable_name,
                value=float(value),
                unit=unit,
                timestamp=ts,
                scenario_id=scenario_id,
                product_id=product_id,
                facility_id=facility_id,
                process_id=process_id,
                transport_leg_id=transport_leg_id,
                source_system=source_system,
                comment=comment,
            ))

        count += 1

    session.commit()
    print(f"Loaded {count} measurements.")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    session = SessionLocal()
    try:
        scenario_map = load_scenarios(session)
        load_emission_factors(session)
        load_cost_factors(session)
        load_kpis(session)
        load_measurements(session, scenario_map)
    finally:
        session.close()


if __name__ == "__main__":
    main()
