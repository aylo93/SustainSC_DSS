# sustainsc/normalization.py
# Normalization module for SustainSC DSS
# Normalizes raw KPI values to 0-100 scale and determines semaforo status

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session
from typing import Optional

from .config import SessionLocal
from .models import KPIResult, KPINormalizedResult, KPI, Scenario


def load_normalization_rules() -> pd.DataFrame:
    """Load KPI normalization rules from CSV file."""
    try:
        df = pd.read_csv("data/kpi_normalization_rules.csv")
        return df
    except FileNotFoundError:
        print("WARNING: kpi_normalization_rules.csv not found. Normalization skipped.")
        return pd.DataFrame()


def normalize_value(raw_value: float, rule: pd.Series) -> Tuple[Optional[float], str]:
    """
    Normalize a raw KPI value to 0-100 scale based on normalization rule.

    Returns:
        Tuple of (normalized_value, semaforo)
    """
    if pd.isna(raw_value):
        return None, "Missing"

    direction = rule.get('direction', 'higher_better')
    norm_method = rule.get('norm_method', 'absolute_continuous')
    lower_ref = rule.get('lower_ref')
    upper_ref = rule.get('upper_ref')
    green_threshold = rule.get('green_threshold', 80)
    amber_threshold = rule.get('amber_threshold', 50)
    baseline_required = rule.get('baseline_required', 0)

    # For methods requiring baseline, we need baseline_value
    # For now, assume we have it or handle gracefully
    baseline_value = None  # TODO: implement baseline lookup

    normalized = None
    semaforo = "Missing"

    try:
        if norm_method == 'absolute_continuous':
            # Linear interpolation between lower_ref and upper_ref
            if direction == 'higher_better':
                if raw_value <= lower_ref:
                    normalized = 0
                elif raw_value >= upper_ref:
                    normalized = 100
                else:
                    normalized = ((raw_value - lower_ref) / (upper_ref - lower_ref)) * 100
            else:  # lower_better
                if raw_value >= upper_ref:
                    normalized = 0
                elif raw_value <= lower_ref:
                    normalized = 100
                else:
                    normalized = ((upper_ref - raw_value) / (upper_ref - lower_ref)) * 100

        elif norm_method == 'relative_vs_base_pct':
            # Percentage change from baseline
            if baseline_value is None or baseline_required:
                return None, "Need BASE"

            if baseline_value == 0:
                # Handle division by zero
                normalized = 100 if raw_value == 0 else 0
            else:
                pct_change = ((raw_value - baseline_value) / baseline_value) * 100

                if direction == 'higher_better':
                    # For higher_better, positive change is good
                    if pct_change >= upper_ref:
                        normalized = 100
                    elif pct_change <= lower_ref:
                        normalized = 0
                    else:
                        normalized = ((pct_change - lower_ref) / (upper_ref - lower_ref)) * 100
                else:  # lower_better
                    # For lower_better, negative change is good
                    if pct_change <= lower_ref:
                        normalized = 100
                    elif pct_change >= upper_ref:
                        normalized = 0
                    else:
                        normalized = ((upper_ref - pct_change) / (upper_ref - lower_ref)) * 100

        # Clamp normalized value to 0-100
        if normalized is not None:
            normalized = max(0, min(100, normalized))

        # Determine semaforo
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


def run_normalization(context_id: str = "aggregates_ton"):
    """Run the KPI normalization process."""
    print(f"Starting KPI normalization for context: {context_id}...")

    # Load normalization rules
    rules_df = load_normalization_rules()
    if rules_df.empty:
        print("No normalization rules found. Skipping normalization.")
        return

    # Filter rules by context_id
    context_rules = rules_df[rules_df['context_id'] == context_id]
    if context_rules.empty:
        print(f"No normalization rules found for context {context_id}. Skipping normalization.")
        return

    print(f"Found {len(context_rules)} normalization rules for context {context_id}")

    # Create database session
    with SessionLocal() as session:
        try:
            # Get all KPI results
            kpi_results = session.query(KPIResult).all()

            if not kpi_results:
                print("No KPI results found to normalize.")
                return

            normalized_count = 0

            for result in kpi_results:
                # Find the normalization rule for this KPI
                kpi_code = result.kpi.code if result.kpi else None
                if not kpi_code:
                    continue

                rule = context_rules[context_rules['kpi_code'] == kpi_code]
                if rule.empty:
                    continue  # Skip KPIs not in this context

                rule = rule.iloc[0]

                # Normalize the value
                normalized_value, semaforo = normalize_value(result.value, rule)

                # Create normalized result record
                normalized_result = KPINormalizedResult(
                    kpi_id=result.kpi_id,
                    scenario_id=result.scenario_id,
                    period_end=result.period_end,
                    raw_value=result.value,
                    normalized_value=normalized_value,
                    semaforo=semaforo,
                    lower_ref=rule.get('lower_ref'),
                    upper_ref=rule.get('upper_ref'),
                    baseline_value=None,  # TODO: implement baseline lookup
                    normalization_method=rule.get('norm_method'),
                    notes=f"Normalized using {rule.get('norm_method')} method for context {context_id}"
                )

                session.add(normalized_result)
                normalized_count += 1

            # Commit all changes
            session.commit()
            print(f"Normalization completed. Processed {normalized_count} KPI results for context {context_id}.")

        except Exception as e:
            session.rollback()
            print(f"Error during normalization: {e}")
            raise


if __name__ == "__main__":
    run_normalization()