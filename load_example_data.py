from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Set, Tuple, List

from sustainsc.config import SessionLocal
from sustainsc.models import Scenario, EmissionFactor, CostFactor, KPI, Measurement


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

SCENARIOS_CSV = DATA_DIR / "scenarios.csv"
EMISSION_FACTORS_CSV = DATA_DIR / "emission_factors.csv"
COST_FACTORS_CSV = DATA_DIR / "cost_factors.csv"
KPIS_CSV = DATA_DIR / "kpis.csv"
NORMALIZATION_RULES_CSV = DATA_DIR / "kpi_normalization_rules.csv"
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


def get_expected_base_kpi_codes() -> Set[str]:
    """
    Toma los KPI esperados desde kpi_normalization_rules.csv.
    Para tu caso deben ser 30: E1..E9, EC1..EC8, S1..S6, T1..T7.
    """
    codes: Set[str] = set()

    if not NORMALIZATION_RULES_CSV.exists():
        print("[WARN] kpi_normalization_rules.csv not found. Expected KPI validation skipped.")
        return codes

    reader = open_dict_reader(NORMALIZATION_RULES_CSV)
    for raw in reader:
        row = norm_row(raw)
        code = (row.get("kpi_code") or row.get("code") or "").strip()
        if code:
            codes.add(code)

    return codes


def get_supported_formula_ids() -> Set[str]:
    """
    Lee los formula_id soportados por el motor KPI.
    """
    try:
        from sustainsc.kpi_engine import FORMULAS
        return set(FORMULAS.keys())
    except Exception as e:
        print(f"[WARN] Could not import FORMULAS from sustainsc.kpi_engine: {e}")
        return set()


def load_scenarios(session) -> Dict[str, int]:
    scenario_map: Dict[str, int] = {}

    if not SCENARIOS_CSV.exists():
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


def summarize_kpi_catalog(session, expected_codes: Set[str]) -> None:
    db_rows = session.query(KPI.code, KPI.formula_id).all()
    db_codes = {code for code, _ in db_rows}

    if expected_codes:
        present_expected = sorted(db_codes & expected_codes)
        missing_expected = sorted(expected_codes - db_codes)
        extra_db = sorted(db_codes - expected_codes)

        print(f"[CHECK] Expected base KPI codes from rules: {len(expected_codes)}")
        print(f"[CHECK] Expected base KPI codes present in DB: {len(present_expected)}/{len(expected_codes)}")

        if missing_expected:
            print("[WARN] Missing KPI codes in DB:")
            print("       " + ", ".join(missing_expected))
        else:
            print("[OK] All expected base KPI codes are present in DB.")

        if extra_db:
            print(f"[INFO] Extra KPI codes in DB (not in normalization rules): {len(extra_db)}")
            print("       " + ", ".join(sorted(extra_db)))
    else:
        print(f"[INFO] Total KPI codes in DB: {len(db_codes)}")


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

    expected_codes = get_expected_base_kpi_codes()
    supported_formula_ids = get_supported_formula_ids()

    reader = open_dict_reader(KPIS_CSV)

    created = 0
    updated = 0
    skipped = 0

    csv_codes: Set[str] = set()
    csv_rows_seen = 0
    duplicate_codes: List[str] = []
    unsupported_formulas: List[Tuple[str, str]] = []
    missing_formula_ids: List[str] = []

    for raw in reader:
        row = norm_row(raw)
        code = (row.get("code") or "").strip()
        if not code:
            skipped += 1
            continue

        csv_rows_seen += 1

        if code in csv_codes:
            duplicate_codes.append(code)
        csv_codes.add(code)

        is_benefit_raw = (row.get("is_benefit") or "0").strip().lower()
        is_benefit = is_benefit_raw in ("1", "true", "yes", "y")

        formula_id = (row.get("formula_id") or "").strip()
        if not formula_id:
            missing_formula_ids.append(code)
        elif supported_formula_ids and formula_id not in supported_formula_ids:
            unsupported_formulas.append((code, formula_id))

        payload = dict(
            code=code,
            name=(row.get("name") or code).strip(),
            description=(row.get("description") or "").strip(),
            dimension=(row.get("dimension") or "").strip(),
            decision_level=(row.get("decision_level") or "").strip(),
            flow=(row.get("flow") or "").strip(),
            unit=(row.get("unit") or "").strip(),
            is_benefit=is_benefit,
            formula_id=formula_id,
            protocol_notes=(row.get("protocol_notes") or "").strip(),
        )

        existing = session.query(KPI).filter_by(code=code).first()
        if existing:
            for k, v in payload.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
            updated += 1
        else:
            clean = {k: v for k, v in payload.items() if hasattr(KPI, k)}
            session.add(KPI(**clean))
            created += 1

    session.commit()

    print(f"Loaded KPI rows from CSV: {csv_rows_seen}")
    print(f"Created KPIs: {created}")
    print(f"Updated KPIs: {updated}")
    print(f"Skipped rows without code: {skipped}")

    if duplicate_codes:
        print("[WARN] Duplicate KPI codes found in kpis.csv:")
        print("       " + ", ".join(sorted(set(duplicate_codes))))

    if missing_formula_ids:
        print("[WARN] KPI codes with empty formula_id:")
        print("       " + ", ".join(sorted(set(missing_formula_ids))))

    if unsupported_formulas:
        print("[WARN] KPI rows whose formula_id is not supported by sustainsc.kpi_engine:")
        for code, formula_id in sorted(set(unsupported_formulas)):
            print(f"       {code} -> {formula_id}")

    if expected_codes:
        missing_in_csv = sorted(expected_codes - csv_codes)
        if missing_in_csv:
            print("[WARN] Expected KPI codes missing from kpis.csv:")
            print("       " + ", ".join(missing_in_csv))
        else:
            print(f"[OK] kpis.csv contains all expected base KPI codes ({len(expected_codes)}).")

    summarize_kpi_catalog(session, expected_codes)


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