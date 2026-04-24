# sustainsc/normalization.py
# Normalization module for SustainSC DSS
# Normalizes raw KPI values to 0-100 scale and determines semaforo status
# FIXED: implements BASE lookup for relative_vs_base_pct and clears old normalized results.

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import KPIResult, KPINormalizedResult, KPI, Scenario


def load_normalization_rules() -> pd.DataFrame:
    """Load KPI normalization rules from CSV file."""
    try:
        df = pd.read_csv("data/kpi_normalization_rules.csv")
    except FileNotFoundError:
        print("WARNING: kpi_normalization_rules.csv not found. Normalization skipped.")
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in ["kpi_code", "context_id", "dimension", "direction", "norm_method"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in [
        "weight", "lower_ref", "upper_ref", "green_threshold",
        "amber_threshold", "baseline_required"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _to_bool01(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "required"}


def _clamp_0_100(x: Optional[float]) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    return max(0.0, min(100.0, float(x)))


def _get_base_scenario(session: Session, preferred_code: str = "BASE") -> Optional[Scenario]:
    sc = session.query(Scenario).filter(Scenario.code == preferred_code).first()
    if sc:
        return sc

    sc = session.query(Scenario).filter(Scenario.code == "BASE").first()
    if sc:
        return sc

    return (
        session.query(Scenario)
        .filter(Scenario.code.ilike("%BASE%"))
        .order_by(Scenario.code.asc())
        .first()
    )


def _get_baseline_value(
    session: Session,
    kpi_id: int,
    result_period_end,
    base_scenario_id: Optional[int],
    same_period: bool = False,
) -> Optional[float]:
    if base_scenario_id is None:
        return None

    q = session.query(KPIResult).filter(
        KPIResult.kpi_id == kpi_id,
        KPIResult.scenario_id == base_scenario_id,
    )

    if same_period and result_period_end is not None:
        q = q.filter(KPIResult.period_end == result_period_end)

    base_result = q.order_by(KPIResult.period_end.desc(), KPIResult.id.desc()).first()
    return float(base_result.value) if base_result else None


def normalize_value(
    raw_value: float,
    rule: pd.Series,
    baseline_value: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    if pd.isna(raw_value):
        return None, "Missing"

    direction = str(rule.get("direction", "higher_better")).strip()
    norm_method = str(rule.get("norm_method", "absolute_continuous")).strip()

    lower_ref = rule.get("lower_ref", 0)
    upper_ref = rule.get("upper_ref", 100)
    green_threshold = rule.get("green_threshold", 80)
    amber_threshold = rule.get("amber_threshold", 50)
    baseline_required = _to_bool01(rule.get("baseline_required", 0))

    lower_ref = 0.0 if pd.isna(lower_ref) else float(lower_ref)
    upper_ref = 100.0 if pd.isna(upper_ref) else float(upper_ref)
    green_threshold = 80.0 if pd.isna(green_threshold) else float(green_threshold)
    amber_threshold = 50.0 if pd.isna(amber_threshold) else float(amber_threshold)

    normalized = None
    semaforo = "Missing"

    try:
        raw_value = float(raw_value)

        if norm_method == "absolute_continuous":
            if upper_ref == lower_ref:
                normalized = 100.0 if raw_value == upper_ref else 0.0
            elif direction == "higher_better":
                if raw_value <= lower_ref:
                    normalized = 0.0
                elif raw_value >= upper_ref:
                    normalized = 100.0
                else:
                    normalized = ((raw_value - lower_ref) / (upper_ref - lower_ref)) * 100.0
            else:
                if raw_value >= upper_ref:
                    normalized = 0.0
                elif raw_value <= lower_ref:
                    normalized = 100.0
                else:
                    normalized = ((upper_ref - raw_value) / (upper_ref - lower_ref)) * 100.0

        elif norm_method == "relative_vs_base_pct":
            if baseline_required and baseline_value is None:
                return None, "Need BASE"

            if baseline_value is None:
                return None, "Need BASE"

            baseline_value = float(baseline_value)

            if baseline_value == 0:
                normalized = 100.0 if raw_value == 0 else 0.0
            else:
                pct_change = ((raw_value - baseline_value) / baseline_value) * 100.0

                if direction == "higher_better":
                    if pct_change >= upper_ref:
                        normalized = 100.0
                    elif pct_change <= lower_ref:
                        normalized = 0.0
                    else:
                        normalized = ((pct_change - lower_ref) / (upper_ref - lower_ref)) * 100.0
                else:
                    if pct_change <= lower_ref:
                        normalized = 100.0
                    elif pct_change >= upper_ref:
                        normalized = 0.0
                    else:
                        normalized = ((upper_ref - pct_change) / (upper_ref - lower_ref)) * 100.0

        else:
            print(f"WARNING: Unknown norm_method '{norm_method}' for KPI {rule.get('kpi_code')}.")
            return None, "Missing"

        normalized = _clamp_0_100(normalized)

        if normalized is not None:
            if normalized >= green_threshold:
                semaforo = "Green"
            elif normalized >= amber_threshold:
                semaforo = "Amber"
            else:
                semaforo = "Red"

    except Exception as e:
        print(f"Error normalizing KPI {rule.get('kpi_code')}: {e}")
        semaforo = "Error"

    return normalized, semaforo


def run_normalization(
    context_id: str = "aggregates",
    base_scenario_code: str = "BASE",
    clear_existing: bool = True,
    same_period_baseline: bool = False,
):
    print(f"Starting KPI normalization for context: {context_id}...")

    rules_df = load_normalization_rules()
    if rules_df.empty:
        print("No normalization rules found. Skipping normalization.")
        return

    if "context_id" not in rules_df.columns:
        print("WARNING: context_id column not found in normalization rules. Using all rules.")
        context_rules = rules_df.copy()
    else:
        context_rules = rules_df[rules_df["context_id"].astype(str).str.strip() == context_id].copy()

        if context_rules.empty:
            available = sorted(rules_df["context_id"].dropna().astype(str).str.strip().unique().tolist())
            print(f"WARNING: No normalization rules found for context '{context_id}'. Available contexts: {available}")
            if available:
                fallback = available[0]
                print(f"Using fallback context: {fallback}")
                context_rules = rules_df[rules_df["context_id"].astype(str).str.strip() == fallback].copy()

    if context_rules.empty:
        print("No applicable normalization rules found. Skipping normalization.")
        return

    print(f"Found {len(context_rules)} normalization rules.")

    with SessionLocal() as session:
        try:
            if clear_existing:
                session.query(KPINormalizedResult).delete(synchronize_session=False)
                session.flush()

            base_scenario = _get_base_scenario(session, preferred_code=base_scenario_code)
            base_scenario_id = base_scenario.id if base_scenario else None

            if base_scenario:
                print(f"Using BASE scenario for relative normalization: {base_scenario.code} (id={base_scenario.id})")
            else:
                print("WARNING: No BASE scenario found. Relative normalization will return Need BASE.")

            kpi_results = (
                session.query(KPIResult)
                .join(KPI, KPI.id == KPIResult.kpi_id)
                .filter(~KPI.code.in_(["ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"]))
                .all()
            )

            if not kpi_results:
                print("No KPI results found to normalize.")
                return

            normalized_count = 0

            for result in kpi_results:
                kpi_code = result.kpi.code if result.kpi else None
                if not kpi_code:
                    continue

                rule_df = context_rules[context_rules["kpi_code"].astype(str).str.strip() == str(kpi_code).strip()]
                if rule_df.empty:
                    continue

                rule = rule_df.iloc[0]
                norm_method = str(rule.get("norm_method", "")).strip()
                baseline_required = _to_bool01(rule.get("baseline_required", 0))

                baseline_value = None
                if norm_method == "relative_vs_base_pct" or baseline_required:
                    baseline_value = _get_baseline_value(
                        session=session,
                        kpi_id=result.kpi_id,
                        result_period_end=result.period_end,
                        base_scenario_id=base_scenario_id,
                        same_period=same_period_baseline,
                    )

                normalized_value, semaforo = normalize_value(
                    raw_value=result.value,
                    rule=rule,
                    baseline_value=baseline_value,
                )

                normalized_result = KPINormalizedResult(
                    kpi_id=result.kpi_id,
                    scenario_id=result.scenario_id,
                    period_end=result.period_end,
                    raw_value=result.value,
                    normalized_value=normalized_value,
                    semaforo=semaforo,
                    lower_ref=rule.get("lower_ref"),
                    upper_ref=rule.get("upper_ref"),
                    baseline_value=baseline_value,
                    normalization_method=rule.get("norm_method"),
                    notes=(
                        f"Normalized using {rule.get('norm_method')} method for context {context_id}; "
                        f"BASE={base_scenario.code if base_scenario else None}"
                    ),
                )

                session.add(normalized_result)
                normalized_count += 1

            session.commit()
            print(f"Normalization completed. Processed {normalized_count} KPI results.")

        except Exception as e:
            session.rollback()
            print(f"Error during normalization: {e}")
            raise


if __name__ == "__main__":
    run_normalization()
