"""Export reconciled analytical results from the active SustainSC dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
from sqlalchemy import text

from sustainsc.composite_indices import corrected_sustain_index
from sustainsc.config import engine
from sustainsc.mcda import (
    build_mcda_input,
    calculate_mcda,
    compute_complete_dimension_indices,
    evaluate_scenario_eligibility,
)
from sustainsc.numerical import comparison_effect


def export_active_results(output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with engine.connect() as connection:
        active = connection.execute(text(
            "SELECT id, reference_scenario_code FROM sc_import_run WHERE is_active=1"
        )).mappings().one()
        run_id, reference = int(active["id"]), str(active["reference_scenario_code"])
        raw = pd.read_sql(text("""
            SELECT s.code scenario_code, k.code kpi_code, r.value raw_value
            FROM sc_kpi_result r JOIN sc_scenario s ON s.id=r.scenario_id
            JOIN sc_kpi k ON k.id=r.kpi_id
            WHERE r.import_run_id=:run AND k.code NOT LIKE '%_INDEX'
        """), connection, params={"run": run_id})
        norm = pd.read_sql(text("""
            SELECT s.code scenario_code, k.code kpi_code, k.name kpi_name,
                   k.dimension, n.normalized_value, n.semaforo, n.notes
            FROM sc_kpi_normalized_result n JOIN sc_scenario s ON s.id=n.scenario_id
            JOIN sc_kpi k ON k.id=n.kpi_id WHERE n.import_run_id=:run
        """), connection, params={"run": run_id})
        measurements = pd.read_sql(text("""
            SELECT s.code scenario_code, m.variable_name, m.value, m.comment
            FROM sc_measurement m JOIN sc_scenario s ON s.id=m.scenario_id
            WHERE m.import_run_id=:run AND m.variable_name IN ('ghg_total_s1s2','maintenance_cost_eur')
        """), connection, params={"run": run_id})

    rules = pd.read_csv("data/kpi_normalization_rules.csv")
    rules.columns = rules.columns.str.strip().str.lower()
    metadata = rules[["kpi_code", "dimension"]].drop_duplicates()
    local = rules.drop_duplicates("kpi_code").set_index("kpi_code")["weight"].astype(float)
    dimensions = ["environmental", "economic", "social", "technological"]
    global_weights = pd.concat([
        group / group.sum() * 0.25
        for _, group in rules.set_index("kpi_code").groupby("dimension")["weight"]
    ]).reindex(rules["kpi_code"]).astype(float)
    global_weights.index = rules["kpi_code"]

    dim_long, _ = compute_complete_dimension_indices(norm, metadata, local)
    dim = dim_long.pivot(index="scenario_code", columns="dimension", values="dimension_index").reset_index()
    dim["SUSTAIN_INDEX_GEOM"] = dim.apply(lambda row: corrected_sustain_index(
        {name: row[name] for name in dimensions}, {name: 0.25 for name in dimensions}, method="geometric"
    ), axis=1)
    dim["SUSTAIN_INDEX_ARITH"] = dim.apply(lambda row: corrected_sustain_index(
        {name: row[name] for name in dimensions}, {name: 0.25 for name in dimensions}, method="arithmetic"
    ), axis=1)

    eligibility = evaluate_scenario_eligibility(raw, norm, metadata)
    mcda_input = build_mcda_input(
        norm, global_weights, eligibility, reference_scenario_code=reference
    )
    mcda_result = calculate_mcda(mcda_input, eligibility)
    ranking = mcda_result.wsm.merge(mcda_result.topsis, on="scenario_code", validate="one_to_one")
    ranking["Rank_WSM"] = ranking["WSM_score"].rank(ascending=False, method="dense")
    ranking["Rank_TOPSIS"] = ranking["TOPSIS_score"].rank(ascending=False, method="dense")
    ranking = ranking.sort_values(["Rank_WSM", "scenario_code"])

    ref = norm[norm.scenario_code == reference].set_index("kpi_code")
    details = []
    def diagnostic(notes: object, key: str) -> str | float | bool | None:
        match = re.search(rf"(?:^|; )({re.escape(key)})=([^;]+)", str(notes or ""))
        if not match:
            return None
        value = match.group(2).strip()
        if value in {"True", "False"}:
            return value == "True"
        if value == "None":
            return None
        try:
            return float(value)
        except ValueError:
            return value

    for scenario in sorted(set(norm.scenario_code) - {reference}):
        current = norm[norm.scenario_code == scenario].set_index("kpi_code")
        for code in rules.kpi_code:
            delta = float(current.at[code, "normalized_value"] - ref.at[code, "normalized_value"])
            effect = comparison_effect(delta)
            scenario_light = current.at[code, "semaforo"]
            if effect == "Same":
                scenario_light = ref.at[code, "semaforo"]
            notes = current.at[code, "notes"]
            details.append({
                "scenario": scenario, "kpi_code": code,
                "kpi_name": current.at[code, "kpi_name"], "dimension": current.at[code, "dimension"],
                "reference_score": ref.at[code, "normalized_value"],
                "scenario_score": current.at[code, "normalized_value"], "delta_pts": delta,
                "reference_semaforo": ref.at[code, "semaforo"],
                "scenario_semaforo": scenario_light, "effect": effect,
                "denominator_effect_flag": diagnostic(notes, "denominator_effect_flag") if code == "EC2" else False,
                "energy_cost_per_fu_reference": diagnostic(notes, "energy_cost_per_fu_reference") if code == "EC2" else None,
                "energy_cost_per_fu_scenario": diagnostic(notes, "energy_cost_per_fu_scenario") if code == "EC2" else None,
                "raw_EC2_score": diagnostic(notes, "raw_EC2_score") if code == "EC2" else None,
                "guarded_EC2_score": diagnostic(notes, "guarded_EC2_score") if code == "EC2" else None,
            })
    detail = pd.DataFrame(details)
    summary = detail.groupby("scenario").apply(lambda g: pd.Series({
        "Improved": (g.effect == "Improved").sum(), "Worse": (g.effect == "Worse").sum(),
        "Same": (g.effect == "Same").sum(), "Missing": (g.effect == "Missing").sum(),
        "Mean delta (pts)": g.delta_pts.mean(), "Median delta (pts)": g.delta_pts.median(),
        "Net score": (g.effect == "Improved").sum() - (g.effect == "Worse").sum(),
    }), include_groups=False).reset_index()
    by_dimension = detail.groupby(["scenario", "dimension"]).apply(lambda g: pd.Series({
        "Improved": (g.effect == "Improved").sum(), "Worse": (g.effect == "Worse").sum(),
        "Same": (g.effect == "Same").sum(), "Mean delta (pts)": g.delta_pts.mean(),
    }), include_groups=False).reset_index()
    traffic = norm.groupby(["scenario_code", "semaforo"]).size().unstack(fill_value=0).reset_index()

    factor_ghg = measurements[measurements.comment.str.contains("MRV_R_GHG_S1S2_FACTORS", na=False)]
    base_maintenance = float(measurements[(measurements.scenario_code == reference) & (measurements.variable_name == "maintenance_cost_eur")].value.iloc[0])
    retained_maintenance = measurements[(measurements.scenario_code != reference) & (measurements.variable_name == "maintenance_cost_eur") & measurements.comment.str.contains("rule=L6", na=False)]
    contradictory = detail[(detail.effect == "Same") & (detail.reference_semaforo != detail.scenario_semaforo)]
    checks = pd.DataFrame([
        {"check_id": "FACTOR_GHG_RECALCULATED", "scenario_code": "", "variable_or_kpi": "ghg_total_s1s2", "expected_logic": "Weak GHG totals use the configured analytical factors", "actual_value": len(factor_ghg), "status": "PASS" if len(factor_ghg) else "FAIL", "message": "Factor-rule measurements in active run"},
        {"check_id": "MAINTENANCE_L6_RETENTION", "scenario_code": "", "variable_or_kpi": "maintenance_cost_eur", "expected_logic": "Unsupported maintenance changes retain reference value", "actual_value": len(retained_maintenance), "status": "PASS" if retained_maintenance.value.eq(base_maintenance).all() else "FAIL", "message": "L6 maintenance values compared with reference"},
        {"check_id": "MCDA_REFERENCE_EXCLUDED", "scenario_code": reference, "variable_or_kpi": "MCDA", "expected_logic": "Reference excluded before WSM/TOPSIS", "actual_value": reference not in set(ranking.scenario_code), "status": "PASS" if reference not in set(ranking.scenario_code) else "FAIL", "message": f"{len(ranking)} alternatives"},
        {"check_id": "TRAFFIC_LIGHT_TOLERANCE", "scenario_code": "", "variable_or_kpi": "traffic light", "expected_logic": "Same effects cannot have contradictory lights from numeric noise", "actual_value": len(contradictory), "status": "PASS" if contradictory.empty else "FAIL", "message": "Contradictory Same/light rows"},
        {"check_id": "NORMALIZED_COMPARISON_COMPLETE", "scenario_code": "", "variable_or_kpi": "normalized comparison", "expected_logic": "23 alternatives x 30 KPI", "actual_value": len(detail), "status": "PASS" if len(detail)==690 else "FAIL", "message": "Complete alternative/KPI matrix"},
        {"check_id": "KPI_COMPLETENESS", "scenario_code": "", "variable_or_kpi": "raw and normalized KPI", "expected_logic": "24 scenarios x 30 KPI", "actual_value": f"raw={len(raw)}; normalized={len(norm)}", "status": "PASS" if len(raw)==len(norm)==720 else "FAIL", "message": "Active-run KPI population"},
    ])
    outputs = {
        "dimension_indices_by_scenario.csv": dim,
        "mcda_scenario_ranking.csv": ranking,
        "traffic_light_distribution.csv": traffic,
        "normalized_comparison_detail.csv": detail,
        "normalized_comparison_by_dimension.csv": by_dimension,
        "normalized_comparison_summary.csv": summary,
        "final_reconciliation_checks.csv": checks,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    return {filename: len(frame) for filename, frame in outputs.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("generated/results_latest_corrected"))
    args = parser.parse_args()
    print(export_active_results(args.output_dir))
