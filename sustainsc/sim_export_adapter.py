# sustainsc/sim_export_adapter.py
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

import pandas as pd

from sustainsc.config import SessionLocal
from sustainsc.models import Scenario, Measurement

MRV_COLS = {"variable_name", "value", "unit", "timestamp", "scenario_code"}

def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def snake(s: str) -> str:
    s = norm(s).lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def sanitize_scenario_code(code: str) -> str:
    code = norm(code)
    if not code:
        return "SIM_EXPORT"
    # evita espacios raros; respeta si ya es tipo BASE/S1/S2/MILP_...
    code = code.replace(" ", "_")
    return code[:50]  # tu columna code es String(50)

def parse_unit_from_header(col: str) -> Tuple[str, Optional[str]]:
    col = norm(col)
    m = re.search(r"\(([^)]+)\)\s*$", col)
    if m:
        base = col[: m.start()].strip()
        unit = m.group(1).strip()
        return base, unit
    m = re.search(r",\s*([A-Za-z%/0-9\-\_]+)\s*$", col)
    if m:
        base = col[: m.start()].strip()
        unit = m.group(1).strip()
        return base, unit
    return col, None

def get_or_create_scenario(session, code: str, name: Optional[str] = None) -> int:
    code = sanitize_scenario_code(code)
    sc = session.query(Scenario).filter_by(code=code).first()
    if sc:
        if name:
            sc.name = name
        session.flush()
        return sc.id
    sc = Scenario(code=code, name=name or code, description="", notes="auto-created by sim_export_adapter")
    session.add(sc)
    session.flush()
    return sc.id

def read_any_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, engine="openpyxl")

    # CSV
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(p, encoding="utf-8-sig", sep=";")

def normalize_any_export(
    df: pd.DataFrame,
    default_scenario_code: str,
    default_timestamp: Optional[datetime] = None,
    source_system: str = "AnyLogistix/AnyLogic",
    mapping: Optional[Dict[str, Dict[str, str]]] = None,
    scenario_col_candidates: Optional[List[str]] = None,
    time_col_candidates: Optional[List[str]] = None,
) -> pd.DataFrame:
    mapping = mapping or {}
    default_timestamp = default_timestamp or utc_now_naive()
    scenario_col_candidates = scenario_col_candidates or ["scenario", "scenario_name", "scenario_code", "result", "iteration"]
    time_col_candidates = time_col_candidates or ["time", "timestamp", "date", "datetime", "t"]

    # normalize column names (snake)
    orig_cols = list(df.columns)
    colmap = {c: snake(str(c)) for c in orig_cols}
    df = df.rename(columns=colmap)

    cols = set(df.columns)

    # A) already MRV
    if MRV_COLS.issubset(cols):
        out = df.copy()
        out["scenario_code"] = out["scenario_code"].fillna(default_scenario_code).astype(str).map(sanitize_scenario_code)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").fillna(pd.Timestamp(default_timestamp))
        out["source_system"] = source_system
        if "comment" not in out.columns:
            out["comment"] = ""
        return out[list(MRV_COLS) + ["source_system", "comment"]]

    # detect scenario/time columns
    scenario_col = next((snake(c) for c in scenario_col_candidates if snake(c) in cols), None)
    time_col = next((snake(c) for c in time_col_candidates if snake(c) in cols), None)

    # B) long format metric/value
    metric_col = next((snake(c) for c in ["metric", "kpi", "indicator", "name", "statistic"] if snake(c) in cols), None)
    if metric_col and "value" in cols:
        out = pd.DataFrame()
        out["scenario_code"] = df[scenario_col].astype(str).map(sanitize_scenario_code) if scenario_col else sanitize_scenario_code(default_scenario_code)
        out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce") if time_col else pd.Timestamp(default_timestamp)
        out["timestamp"] = out["timestamp"].fillna(pd.Timestamp(default_timestamp))
        out["value"] = pd.to_numeric(df["value"], errors="coerce")
        out["unit"] = df["unit"].astype(str) if "unit" in cols else ""

        def map_metric(m: str) -> Tuple[str, str]:
            m = norm(str(m))
            key = m.lower()
            if key in mapping:
                return mapping[key]["var"], mapping[key].get("unit", "") or ""
            base, inferred_unit = parse_unit_from_header(m)
            return f"sim_{snake(base)}", inferred_unit or ""

        mapped = df[metric_col].apply(map_metric)
        out["variable_name"] = [t[0] for t in mapped]
        inferred_units = [t[1] for t in mapped]
        out.loc[out["unit"].eq(""), "unit"] = inferred_units

        out["source_system"] = source_system
        out["comment"] = "Imported from long-format export"
        out = out.dropna(subset=["value"])
        return out[["variable_name", "value", "unit", "timestamp", "scenario_code", "source_system", "comment"]]

    # C) wide format (comparison table)
    id_cols = []
    if scenario_col:
        id_cols.append(scenario_col)
    if time_col:
        id_cols.append(time_col)

    value_cols = [c for c in df.columns if c not in id_cols]

    metric_cols = []
    for c in value_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            metric_cols.append(c)
    if not metric_cols:
        raise ValueError("No numeric metric columns detected. Export format not recognized.")

    melted = df.melt(
        id_vars=id_cols if id_cols else None,
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"])

    if scenario_col:
        melted["scenario_code"] = melted[scenario_col].astype(str).map(sanitize_scenario_code)
    else:
        melted["scenario_code"] = sanitize_scenario_code(default_scenario_code)

    if time_col:
        melted["timestamp"] = pd.to_datetime(melted[time_col], errors="coerce").fillna(pd.Timestamp(default_timestamp))
    else:
        melted["timestamp"] = pd.Timestamp(default_timestamp)

    def map_metric_col(m: str) -> Tuple[str, str]:
        key = str(m).lower()
        if key in mapping:
            return mapping[key]["var"], mapping[key].get("unit", "") or ""
        return f"sim_{snake(str(m))}", ""

    mapped = melted["metric"].apply(map_metric_col)
    melted["variable_name"] = [t[0] for t in mapped]
    melted["unit"] = [t[1] for t in mapped]
    melted["source_system"] = source_system
    melted["comment"] = "Imported from wide-format export"
    return melted[["variable_name", "value", "unit", "timestamp", "scenario_code", "source_system", "comment"]]

def upsert_measurements(session, df_mrv: pd.DataFrame) -> int:
    written = 0
    for _, r in df_mrv.iterrows():
        sc_code = sanitize_scenario_code(str(r["scenario_code"]))
        scenario_id = get_or_create_scenario(session, sc_code, name=sc_code)

        ts = pd.to_datetime(r["timestamp"], errors="coerce")
        if pd.isna(ts):
            ts = pd.Timestamp(utc_now_naive())
        ts_dt = ts.to_pydatetime()

        var = str(r["variable_name"]).strip()
        val = float(r["value"])
        unit = str(r.get("unit", "") or "").strip()
        src = str(r.get("source_system", "") or "").strip()
        cmt = str(r.get("comment", "") or "").strip()

        existing = (
            session.query(Measurement)
            .filter_by(variable_name=var, timestamp=ts_dt, scenario_id=scenario_id)
            .first()
        )
        if existing:
            existing.value = val
            existing.unit = unit
            existing.source_system = src or existing.source_system
            existing.comment = cmt or existing.comment
        else:
            session.add(
                Measurement(
                    variable_name=var,
                    value=val,
                    unit=unit,
                    timestamp=ts_dt,
                    scenario_id=scenario_id,
                    source_system=src,
                    comment=cmt,
                )
            )
        written += 1

    session.commit()
    return written
