# load_example_data.py
from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from sustainsc.config import SessionLocal
from sustainsc.models import Scenario, EmissionFactor, CostFactor, KPI, Measurement


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

SCENARIOS_CSV = DATA_DIR / "scenarios.csv"
EMISSION_FACTORS_CSV = DATA_DIR / "emission_factors.csv"
COST_FACTORS_CSV = DATA_DIR / "cost_factors.csv"
KPIS_CSV = DATA_DIR / "kpis.csv"
MEASUREMENTS_CSV = DATA_DIR / "measurements.csv"


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
    f = path.open("r", newline="", encoding="utf-8-sig")
    sample = f.read(4096)
    f.seek(0)

    if not sample or not sample.strip():
        return csv.DictReader(f, delimiter=",", skipinitialspace=True)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        print(f"[INFO] {path.name} delimiter detected: {repr(dialect.delimiter)}")
        return csv.DictReader(f, dialect=dialect, skipinitialspace=True)
    except csv.Error:
        counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
        delim = max(counts, key=counts.get)
        if counts[delim] == 0:
            delim = ","
        print(f"[WARN] Could not sniff delimiter for {path.name}. Using heuristic delimiter: {repr(delim)}")
        return csv.DictReader(f, delimiter=delim, skipinitialspace=True)


def load_scenarios(session) -> Dict[str, int]:
    scenario_map: Dict[str, int] = {}

    if not SCENARIOS_CSV.exists():
        # asegura BASE
        base = session.query(Scenario).filter_by(code="BASE").first()
        if not base:
            base = Scenario(code="BASE", name="Baseline", description="Default baseline", notes="")
            session.add(base)
            session.commit()
        scenario_map["BASE"] = base.id
        return scenario_map

    reader = open_dict_reader(SCENARIOS_CSV)
    count = 0
    for raw in reader:
        row = norm_row(raw)
        code = (row.get("code") or "").strip()
        if not code:
            continue

        existing = session.query(Scenario).filter_by(code=code).first()
        if existing:
            existing.name = (row.get("name") or existing.name)
            existing.description = (row.get("description") or existing.description)
            existing.notes = (row.get("notes") or existing.notes)
            session.flush()
            scenario_map[code] = existing.id
        else:
            sc = Scenario(
                code=code,
                name=(row.get("name") or code),
                description=(row.get("description") or ""),
                notes=(row.get("notes") or ""),
            )
            session.add(sc)
            session.flush()
            scenario_map[code] = sc.id

        count += 1

    session.commit()
    print(f"Loaded {count} scenarios.")
    return scenario_map


def load_emission_factors(session) -> None:
    if not EMISSION_FACTORS_CSV.exists():
        print("[WARN] emission_factors.csv not found. Skipping.")
        return

    reader = open_dict_reader(EMISSION_FACTORS_CSV)
    count = 0
    for raw in reader:
        row = norm_row(raw)
        activity_type = (row.get("activity_type") or "").strip()
        if not activity_type:
            continue

        value = ffloat(row.get("value"))
        if value is None:
            continue

        ef = EmissionFactor(
            name=(row.get("name") or activity_type),
            activity_type=activity_type,
            unit=(row.get("unit") or "").strip(),
            value=float(value),
            valid_from=parse_dt(row["valid_from"]) if row.get("valid_from") else None,
            valid_to=parse_dt(row["valid_to"]) if row.get("valid_to") else None,
            source=(row.get("source") or "").strip(),
        )
        session.add(ef)
        count += 1

    session.commit()
    print(f"Loaded {count} emission factors.")


def load_cost_factors(session) -> None:
    if not COST_FACTORS_CSV.exists():
        print("[WARN] cost_factors.csv not found. Skipping.")
        return

    reader = open_dict_reader(COST_FACTORS_CSV)
    count = 0
    for raw in reader:
        row = norm_row(raw)
        activity_type = (row.get("activity_type") or "").strip()
        if not activity_type:
            continue

        value = ffloat(row.get("value"))
        if value is None:
            continue

        cf = CostFactor(
            name=(row.get("name") or activity_type),
            activity_type=activity_type,
            unit=(row.get("unit") or "").strip(),
            value=float(value),
            valid_from=parse_dt(row["valid_from"]) if row.get("valid_from") else None,
            valid_to=parse_dt(row["valid_to"]) if row.get("valid_to") else None,
            source=(row.get("source") or "").strip(),
        )
        session.add(cf)
        count += 1

    session.commit()
    print(f"Loaded {count} cost factors.")


def load_kpis(session) -> None:
    if not KPIS_CSV.exists():
        print("[WARN] kpis.csv not found. Skipping.")
        return

    reader = open_dict_reader(KPIS_CSV)
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
            flow=(row.get("flow") or "").strip(),
            unit=(row.get("unit") or "").strip(),
            is_benefit=is_benefit,
            formula_id=(row.get("formula_id") or "").strip(),
            protocol_notes=(row.get("protocol_notes") or "").strip(),
        )

        existing = session.query(KPI).filter_by(code=code).first()
        if existing:
            for k, v in payload.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
        else:
            # filtra campos que existan
            clean = {k: v for k, v in payload.items() if hasattr(KPI, k)}
            session.add(KPI(**clean))

        count += 1

    session.commit()
    print(f"Loaded {count} KPIs.")


def load_measurements(session, scenario_map: Dict[str, int]) -> None:
    if not MEASUREMENTS_CSV.exists():
        print("[WARN] measurements.csv not found. Skipping.")
        return

    reader = open_dict_reader(MEASUREMENTS_CSV)
    count = 0
    for raw in reader:
        row = norm_row(raw)

        variable_name = (row.get("variable_name") or "").strip()
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

        session.add(
            Measurement(
                scenario_id=scenario_id,
                variable_name=variable_name,
                value=float(value),
                unit=unit,
                timestamp=ts,
                source_system=(row.get("source_system") or "").strip(),
                comment=(row.get("comment") or "").strip(),
            )
        )
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
