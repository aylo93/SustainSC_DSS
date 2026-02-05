# load_example_data.py
# SustainSC DSS - Example data loader (robust CSV loader)
# Loads: scenarios, emission factors, cost factors, KPIs, measurements
# Handles comma or semicolon delimited CSVs (Excel in some locales uses ';').

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

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

def pick_csv(*filenames: str) -> Path:
    candidates = []
    for fn in filenames:
        p = DATA_DIR / fn
        if p.exists():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Could not find any of these files in {DATA_DIR}: {filenames}")

    # Prefer non-empty files (avoid accidentally selecting an empty CSV)
    for p in candidates:
        try:
            if p.stat().st_size > 10:
                return p
        except Exception:
            pass

    return candidates[0]

# File candidates (include your earlier "ascii" export)
SCENARIOS_CSV = DATA_DIR / "scenarios.csv"  # optional
EMISSION_FACTORS_CSV = pick_csv("emission_factors.csv", "data_emission_factors.csv")
COST_FACTORS_CSV = pick_csv("cost_factors.csv", "data_cost_factors.csv")
KPIS_CSV = pick_csv("kpis.csv", "data_kpis.csv", "data_kpis_ascii.csv")
MEASUREMENTS_CSV = pick_csv("measurements.csv", "data_measurements.csv")

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
    out = {}
    for k, v in row.items():
        kk = (k or "").strip().lower().replace(" ", "_")
        out[kk] = v
    return out

def open_dict_reader(path: Path) -> csv.DictReader:
    """
    Open CSV robustly (supports delimiter ',' or ';' or tab).
    Also supports UTF-8 with BOM via utf-8-sig.
    If Sniffer fails (file very small / ambiguous), use heuristic delimiter choice.
    """
    f = path.open("r", newline="", encoding="utf-8-sig")

    sample = f.read(4096)
    f.seek(0)

    # If file is empty or almost empty, fall back to comma
    if not sample or not sample.strip():
        print(f"[WARN] {path.name} seems empty or unreadable; defaulting delimiter to ','")
        return csv.DictReader(f, delimiter=",", skipinitialspace=True)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        print(f"[INFO] {path.name} delimiter detected: {repr(dialect.delimiter)}")
        return csv.DictReader(f, dialect=dialect, skipinitialspace=True)

    except csv.Error:
        # Heuristic: choose the delimiter that appears most in the sample
        counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
        delim = max(counts, key=counts.get)
        if counts[delim] == 0:
            delim = ","  # final default

        print(f"[WARN] Could not sniff delimiter for {path.name}. Using heuristic delimiter: {repr(delim)}")
        return csv.DictReader(f, delimiter=delim, skipinitialspace=True)

# -----------------------------
# Loaders
# -----------------------------

def load_scenarios(session) -> Dict[str, int]:
    scenario_map: Dict[str, int] = {}

    def upsert_scenario(code: str, name: str, description: str = "", notes: str = "") -> int:
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

    if SCENARIOS_CSV.exists():
        reader = open_dict_reader(SCENARIOS_CSV)
        count = 0
        for raw in reader:
            row = norm_row(raw)
            code = (row.get("code") or "").strip()
            if not code:
                continue
            sid = upsert_scenario(
                code=code,
                name=(row.get("name") or code),
                description=(row.get("description") or ""),
                notes=(row.get("notes") or ""),
            )
            scenario_map[code] = sid
            count += 1
        session.commit()
        print(f"Loaded {count} scenarios.")
    else:
        base = session.query(Scenario).filter_by(code="BASE").first()
        if not base:
            base = Scenario(code="BASE", name="Baseline", description="Default baseline scenario", notes="")
            session.add(base)
            session.commit()
        scenario_map["BASE"] = base.id
        print("Scenarios CSV not found. Ensured BASE scenario exists.")

    for sc in session.query(Scenario).all():
        scenario_map[sc.code] = sc.id

    return scenario_map

def load_emission_factors(session) -> None:
    reader = open_dict_reader(EMISSION_FACTORS_CSV)
    print(f"Emission factors headers: {reader.fieldnames}")

    count = 0
    for raw in reader:
        row = norm_row(raw)
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
    reader = open_dict_reader(COST_FACTORS_CSV)
    print(f"Cost factors headers: {reader.fieldnames}")

    count = 0
    for raw in reader:
        row = norm_row(raw)
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
    reader = open_dict_reader(KPIS_CSV)
    print(f"KPIs headers: {reader.fieldnames}")

    count = 0
    for raw in reader:
        row = norm_row(raw)
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
    reader = open_dict_reader(MEASUREMENTS_CSV)
    print(f"Measurements headers: {reader.fieldnames}")

    count = 0
    for raw in reader:
        row = norm_row(raw)

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
            sc = session.query(Scenario).filter_by(code=scenario_code).first()
            if not sc:
                sc = Scenario(code=scenario_code, name=scenario_code, description="", notes="auto-created")
                session.add(sc)
                session.flush()
            scenario_id = sc.id
            scenario_map[scenario_code] = scenario_id

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

def main():
    # ... tu lógica de cargar escenarios/factores/kpis/measurements
    pass

if __name__ == "__main__":
    main()
