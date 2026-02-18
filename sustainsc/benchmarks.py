# sustainsc/benchmarks.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Puedes usar JSON (recomendado, porque conserva mejor strings como "≥" y rangos)
DEFAULT_JSON = DATA_DIR / "kpi_benchmarks_semaforo.json"
DEFAULT_CSV = DATA_DIR / "kpi_benchmarks_semaforo.csv"


def load_benchmarks() -> pd.DataFrame:
    """
    Loads benchmark rules into a DataFrame with at least:
    code, direction, benchmark_method, green_rule, amber_rule, red_rule,
    industry_dependent, baseline_required, notes
    """
    if DEFAULT_JSON.exists():
        rows = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
        df = pd.DataFrame(rows)
        return df

    if DEFAULT_CSV.exists():
        return pd.read_csv(DEFAULT_CSV)

    return pd.DataFrame()


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _parse_absolute_thresholds(rule_text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Parses simple rules like:
      "≥80%"  -> min=80
      "<50%"  -> max=50 (exclusive)
      "50–79%" or "50-79" -> min=50, max=79
      "4–5" -> min=4 max=5
    Returns (min_value, max_value, op_hint)
    """
    if not rule_text:
        return None, None, None

    s = rule_text.strip()
    s = s.replace("–", "-").replace("—", "-").replace(" ", "")

    # >=
    m = re.search(r"(>=|≥)(-?\d+(\.\d+)?)", s)
    if m:
        return float(m.group(2)), None, ">="

    # <=
    m = re.search(r"(<=|≤)(-?\d+(\.\d+)?)", s)
    if m:
        return None, float(m.group(2)), "<="

    # < number
    m = re.search(r"<(-?\d+(\.\d+)?)", s)
    if m:
        return None, float(m.group(1)), "<"

    # > number
    m = re.search(r">(-?\d+(\.\d+)?)", s)
    if m:
        return float(m.group(1)), None, ">"

    # range a-b
    m = re.search(r"(-?\d+(\.\d+)?)-(-?\d+(\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(3)), "range"

    return None, None, None


def semaforo_label(
    value: Optional[float],
    base_value: Optional[float],
    direction: str,
    benchmark_method: str,
    green_rule: str,
    amber_rule: str,
    red_rule: str,
    baseline_required: int | bool,
) -> str:
    """
    Returns: "Green" | "Amber" | "Red" | "Missing" | "Need BASE"
    Strategy:
      - If method is absolute_thresholds/maturity_scale -> parse thresholds from rules and evaluate value.
      - If method is relative_vs_baseline* -> compute %Δ vs BASE and apply generic bands:
            Green: improvement >= 3%
            Amber: within ±0–3%
            Red: worsening
        (Esto coincide con varias reglas del JSON: E2/E3/E9, etc.)
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Missing"

    baseline_required = int(baseline_required) if baseline_required is not None else 0
    if baseline_required == 1:
        if base_value is None or (isinstance(base_value, float) and pd.isna(base_value)):
            return "Need BASE"

    method = (benchmark_method or "").lower().strip()
    dirn = (direction or "").lower().strip()

    # ---------------------------
    # Relative vs baseline
    # ---------------------------
    if "relative_vs_baseline" in method:
        # %Δ: positive = increase, negative = decrease
        if base_value in (None, 0) or (isinstance(base_value, float) and pd.isna(base_value)) or base_value == 0:
            return "Need BASE"

        pct = (float(value) - float(base_value)) / float(base_value) * 100.0

        # improvement sign depends on direction
        # lower_better => improvement when pct < 0
        # higher_better => improvement when pct > 0
        if dirn == "lower_better":
            if pct <= -3.0:
                return "Green"
            elif -3.0 < pct < 0.0:
                return "Amber"
            else:
                return "Red"
        else:
            if pct >= 3.0:
                return "Green"
            elif 0.0 <= pct < 3.0:
                return "Amber"
            else:
                return "Red"

    # ---------------------------
    # Absolute thresholds / maturity scale
    # ---------------------------
    # Idea: Green rule defines min or a range; Amber is mid; Red is low.
    gmin, gmax, gop = _parse_absolute_thresholds(green_rule or "")
    amin, amax, aop = _parse_absolute_thresholds(amber_rule or "")
    rmin, rmax, rop = _parse_absolute_thresholds(red_rule or "")

    v = float(value)

    # try green check
    if gop == ">=" and gmin is not None and v >= gmin:
        return "Green"
    if gop == "<=" and gmax is not None and v <= gmax:
        return "Green"
    if gop == "range" and gmin is not None and gmax is not None and (gmin <= v <= gmax):
        return "Green"
    if gop == ">" and gmin is not None and v > gmin:
        return "Green"
    if gop == "<" and gmax is not None and v < gmax:
        return "Green"

    # amber check
    if aop == "range" and amin is not None and amax is not None and (amin <= v <= amax):
        return "Amber"
    if aop == ">=" and amin is not None and v >= amin:
        return "Amber"
    if aop == "<=" and amax is not None and v <= amax:
        return "Amber"
    if aop == "<" and amax is not None and v < amax:
        return "Amber"
    if aop == ">" and amin is not None and v > amin:
        return "Amber"

    # if not green/amber and we have red defined, call it red; else default amber
    if (red_rule or "").strip():
        return "Red"
    return "Amber"
