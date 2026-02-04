# generate_demo_measurements.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

MONTHS = ["2025-10-01T00:00:00", "2025-11-01T00:00:00", "2025-12-01T00:00:00"]

BASE = {
    "output_qty_fu": 6000,
    "electricity_kwh": 1_000_000,
    "renewable_energy_kwh": 250_000,
    "diesel_kwh": 500_000,
    "water_withdrawn_m3": 12_000,
    "waste_generated_t": 800,
    "waste_recovered_t": 520,
    "material_total_t": 9_000,
    "material_circular_t": 1_800,

    "operating_cost_eur": 5_000_000,
    "fuel_cost_eur": 600_000,
    "electricity_cost_eur": 400_000,
    "maintenance_cost_eur": 350_000,
    "operating_hours_h": 50_000,
    "logistics_cost_eur": 900_000,
    "transport_work_tkm": 45_000_000,

    "lti_count": 3,
    "hours_worked_h": 2_500_000,
    "training_hours_h": 12_000,
    "employees_avg": 1200,
    "employees_total": 1200,
    "suggestions_count": 900,
    "community_incidents_count": 2,
    "substantiated_complaints_count": 4,
    "employees_health_checks": 1050,

    "sustainable_sales_share": 0.22,
    "customer_survey_score": 0.68,
    "contract_renewal_rate": 0.75,

    "digital_processes_count": 18,
    "core_processes_total": 24,
    "assets_connected": 320,
    "assets_total": 500,
    "mrv_points_active_valid": 140,
    "mrv_points_required": 200,
    "dpp_volume": 4800,
    "shipped_volume_total": 6000,
    "data_completeness": 0.92,
    "data_consistency": 0.88,
    "data_accuracy": 0.85,
    "ot_it_integration_score": 3.5,
    "decisions_supported": 14,
    "decisions_total": 20,

    "oee_availability": 0.86,
    "oee_performance": 0.90,
    "oee_quality": 0.97,

    "investment_cost_eur": 1_200_000,
    "incremental_benefits_eur": 240_000,
}

SCENARIO_ADJUST = {
    "BASE": {},
    "ALT1": {
        "electricity_kwh": 0.90,
        "diesel_kwh": 0.80,
        "renewable_energy_kwh": 1.60,
        "electricity_cost_eur": 0.90,
        "fuel_cost_eur": 0.80,
        "operating_cost_eur": 0.96,
        "E7_hint": None,
    },
    "ALT2": {
        "maintenance_cost_eur": 0.85,
        "data_completeness": 1.04,
        "data_consistency": 1.05,
        "data_accuracy": 1.06,
        "ot_it_integration_score": 1.20,
        "decisions_supported": 1.15,
        "digital_processes_count": 1.10,
        "assets_connected": 1.15,
        "mrv_points_active_valid": 1.20,
    },
}

MONTH_FACTOR = {
    "2025-10-01T00:00:00": 0.96,
    "2025-11-01T00:00:00": 1.00,
    "2025-12-01T00:00:00": 1.04,
}

FRACTION_VARS = {"sustainable_sales_share", "customer_survey_score", "contract_renewal_rate",
                 "data_completeness", "data_consistency", "data_accuracy",
                 "oee_availability", "oee_performance", "oee_quality"}

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def make_rows(month_ts: str, scenario_code: str):
    rows = []
    month_mult = MONTH_FACTOR[month_ts]
    adj = SCENARIO_ADJUST[scenario_code]

    for var, base_val in BASE.items():
        val = float(base_val)

        # Apply month multiplier to "volume-like" metrics (not to indices/scores)
        if var not in FRACTION_VARS and var not in {"ot_it_integration_score"}:
            val *= month_mult

        # Apply scenario adjustment multipliers if present
        if var in adj:
            val *= float(adj[var])

        # Clamp fractions
        if var in FRACTION_VARS:
            val = clamp01(val)

        rows.append({
            "variable_name": var,
            "value": val,
            "unit": "unit",
            "timestamp": month_ts,
            "scenario_code": scenario_code
        })

    # Units map (optional; makes your data look more professional)
    unit_map = {
        "output_qty_fu": "FU",
        "electricity_kwh": "kWh",
        "renewable_energy_kwh": "kWh",
        "diesel_kwh": "kWh",
        "water_withdrawn_m3": "m3",
        "waste_generated_t": "t",
        "waste_recovered_t": "t",
        "material_total_t": "t",
        "material_circular_t": "t",

        "operating_cost_eur": "EUR",
        "fuel_cost_eur": "EUR",
        "electricity_cost_eur": "EUR",
        "maintenance_cost_eur": "EUR",
        "operating_hours_h": "h",
        "logistics_cost_eur": "EUR",
        "transport_work_tkm": "tkm",

        "lti_count": "count",
        "hours_worked_h": "h",
        "training_hours_h": "h",
        "employees_avg": "count",
        "employees_total": "count",
        "suggestions_count": "count",
        "community_incidents_count": "count",
        "substantiated_complaints_count": "count",
        "employees_health_checks": "count",

        "sustainable_sales_share": "fraction",
        "customer_survey_score": "fraction",
        "contract_renewal_rate": "fraction",

        "digital_processes_count": "count",
        "core_processes_total": "count",
        "assets_connected": "count",
        "assets_total": "count",
        "mrv_points_active_valid": "count",
        "mrv_points_required": "count",
        "dpp_volume": "FU",
        "shipped_volume_total": "FU",
        "data_completeness": "fraction",
        "data_consistency": "fraction",
        "data_accuracy": "fraction",
        "ot_it_integration_score": "score",
        "decisions_supported": "count",
        "decisions_total": "count",

        "oee_availability": "fraction",
        "oee_performance": "fraction",
        "oee_quality": "fraction",

        "investment_cost_eur": "EUR",
        "incremental_benefits_eur": "EUR",
    }

    for r in rows:
        r["unit"] = unit_map.get(r["variable_name"], "unit")

    return rows

def main():
    for month_ts in MONTHS:
        rows = []
        for sc in ["BASE", "ALT1", "ALT2"]:
            rows.extend(make_rows(month_ts, sc))

        df = pd.DataFrame(rows)
        fname = month_ts[:7].replace("-", "_")  # YYYY_MM
        out = OUT_DIR / f"measurements_{fname}.csv"
        df.to_csv(out, index=False)
        print("Wrote:", out)

if __name__ == "__main__":
    main()
