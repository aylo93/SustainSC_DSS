# sustainsc/normalization.py
# SustainSC DSS - KPI normalization module
# Final version:
# - Implements BASE lookup for relative_vs_base_pct
# - Clears old normalized results before recalculation
# - Uses centered relative normalization:
#     BASE-equivalent performance = neutral score (50)
#     improvement vs BASE -> 50 to 100
#     deterioration vs BASE -> 50 to 0

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import KPIResult, KPINormalizedResult, KPI, Scenario


COMPOSITE_CODES = {"ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"}


def load_normalization_rules() -> pd.DataFrame:
    """Load KPI normalization rules from CSV file."""
    try:
        df = pd.read_csv("data/kpi_normalization_rules.csv")
    except FileNotFoundError:
        print("WARNING: data/kpi_normalization_rules.csv not found. Normalization skipped.")
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in ["kpi_code", "context_id", "dimension", "direction", "norm_method"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in [
        "weight",
        "lower_ref",
        "upper_ref",
        "green_threshold",
        "amber_threshold",
        "baseline_required",
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
    """
    Find the reference BASE scenario.

    Priority:
    1. Exact preferred_code
    2. Exact BASE
    3. Any scenario containing BASE
    """
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
    base_scenario_id: Optional[int],
    result_period_end=None,
    same_period: bool = False,
) -> Optional[float]:
    """
    Retrieve BASE KPI value for the same KPI.

    By default, this does not require the same period_end because the DSS compares
    2025/2030/2035 scenario outputs against one BASE reference.
    """
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


def _centered_relative_score(
    raw_value: float,
    baseline_value: float,
    direction: str,
    lower_ref: float,
    upper_ref: float,
) -> Optional[float]:
    """
    Centered relative normalization.

    Interpretation:
    - raw == BASE -> 50
    - better than BASE -> 50..100
    - worse than BASE -> 50..0

    upper_ref defines the % improvement/deterioration band depending on direction.
    lower_ref is also supported, but if it is 0, the function mirrors upper_ref.
    """
    if baseline_value == 0:
        if raw_value == 0:
            return 50.0
        return 100.0 if direction == "higher_better" and raw_value > 0 else 0.0

    pct_change = ((raw_value - baseline_value) / baseline_value) * 100.0

    # Avoid division by zero in old rule tables where lower_ref may be 0
    improvement_band = abs(upper_ref) if abs(upper_ref) > 1e-9 else 10.0
    deterioration_band = abs(lower_ref) if abs(lower_ref) > 1e-9 else improvement_band

    if abs(pct_change) < 1e-12:
        return 50.0

    if direction == "higher_better":
        if pct_change > 0:
            return 50.0 + min(50.0, (pct_change / improvement_band) * 50.0)
        return 50.0 - min(50.0, (abs(pct_change) / deterioration_band) * 50.0)

    # lower_better
    if pct_change < 0:
        return 50.0 + min(50.0, (abs(pct_change) / improvement_band) * 50.0)
    return 50.0 - min(50.0, (pct_change / deterioration_band) * 50.0)


def normalize_value(
    raw_value: float,
    rule: pd.Series,
    baseline_value: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """
    Normalize a raw KPI value to a 0-100 score and assign traffic-light status.
    """
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

            else:  # lower_better
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

            normalized = _centered_relative_score(
                raw_value=raw_value,
                baseline_value=float(baseline_value),
                direction=direction,
                lower_ref=lower_ref,
                upper_ref=upper_ref,
            )

        else:
            print(f"WARNING: Unknown norm_method '{norm_method}' for KPI {rule.get('kpi_code')}.")
            return None, "Missing"

        normalized = _clamp_0_100(normalized)

        if normalized is None:
            return None, "Missing"

        if normalized >= green_threshold:
            semaforo = "Green"
        elif normalized >= amber_threshold:
            semaforo = "Amber"
        else:
            semaforo = "Red"

        return normalized, semaforo

    except Exception as e:
        print(f"Error normalizing KPI {rule.get('kpi_code')}: {e}")
        return None, "Error"


def run_normalization(
    context_id: str = "aggregates",
    base_scenario_code: str = "BASE",
    clear_existing: bool = True,
    same_period_baseline: bool = False,
):
    """
    Run KPI normalization.

    Parameters
    ----------
    context_id:
        Context id from kpi_normalization_rules.csv. If it is not found, the
        function falls back to the first available context.
    base_scenario_code:
        Scenario code used as reference for relative normalization.
    clear_existing:
        Delete old normalized results before recalculating.
    same_period_baseline:
        If True, the BASE result must have the same period_end as the scenario.
        Keep False for comparing 2030/2035 scenarios against the BASE reference.
    """
    print(f"Starting KPI normalization for context: {context_id}...")

    rules_df = load_normalization_rules()
    if rules_df.empty:
        print("No normalization rules found. Skipping normalization.")
        return

    if "context_id" not in rules_df.columns:
        print("WARNING: context_id column not found. Using all normalization rules.")
        context_rules = rules_df.copy()
    else:
        context_rules = rules_df[rules_df["context_id"].astype(str).str.strip() == context_id].copy()

        if context_rules.empty:
            available = sorted(rules_df["context_id"].dropna().astype(str).str.strip().unique().tolist())
            print(f"WARNING: No rules found for context '{context_id}'. Available contexts: {available}")
            if available:
                fallback_context = available[0]
                print(f"Using fallback context: {fallback_context}")
                context_rules = rules_df[
                    rules_df["context_id"].astype(str).str.strip() == fallback_context
                ].copy()

    if context_rules.empty:
        print("No applicable normalization rules found. Skipping normalization.")
        return

    with SessionLocal() as session:
        try:
            if clear_existing:
                session.query(KPINormalizedResult).delete(synchronize_session=False)
                session.flush()

            base_scenario = _get_base_scenario(session, preferred_code=base_scenario_code)
            base_scenario_id = base_scenario.id if base_scenario else None

            if base_scenario:
                print(f"Using BASE scenario: {base_scenario.code} (id={base_scenario.id})")
            else:
                print("WARNING: No BASE scenario found. Relative KPIs will show Need BASE.")

            kpi_results = (
                session.query(KPIResult)
                .join(KPI, KPI.id == KPIResult.kpi_id)
                .filter(~KPI.code.in_(list(COMPOSITE_CODES)))
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

                rule_df = context_rules[
                    context_rules["kpi_code"].astype(str).str.strip() == str(kpi_code).strip()
                ]

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
                        base_scenario_id=base_scenario_id,
                        result_period_end=result.period_end,
                        same_period=same_period_baseline,
                    )

                normalized_value, semaforo = normalize_value(
                    raw_value=result.value,
                    rule=rule,
                    baseline_value=baseline_value,
                )

                session.add(
                    KPINormalizedResult(
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
                            f"Normalized using {rule.get('norm_method')} for context {context_id}; "
                            f"BASE={base_scenario.code if base_scenario else None}; "
                            f"centered_relative=True"
                        ),
                    )
                )

                normalized_count += 1

            session.commit()
            print(f"Normalization completed. Processed {normalized_count} KPI results.")

        except Exception as e:
            session.rollback()
            print(f"Error during normalization: {e}")
            raise


if __name__ == "__main__":
    run_normalization()