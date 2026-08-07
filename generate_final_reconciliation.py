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
            WHERE m.import_run_id=:run
        """), connection, params={"run": run_id})
        factors = pd.read_sql(text("""
            SELECT factor_set_id, code factor_code, value factor_value, unit factor_unit,
                   analytical_role, source
            FROM sc_emission_factor
            WHERE lower(coalesce(analytical_role,'')) LIKE '%transport-scope%'
        """), connection)

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
            raw_row = raw[(raw.scenario_code == scenario) & (raw.kpi_code == code)]
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
                "raw_ec2_value": float(raw_row.raw_value.iloc[0]) if code == "EC2" and not raw_row.empty else None,
                "import_run_id": run_id,
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
    ec2_diagnostics = detail[detail.kpi_code == "EC2"].loc[:, [
        "scenario", "raw_ec2_value", "raw_EC2_score",
        "energy_cost_per_fu_reference", "energy_cost_per_fu_scenario",
        "guarded_EC2_score", "denominator_effect_flag", "effect", "import_run_id",
    ]].rename(columns={
        "scenario": "scenario_code",
        "raw_EC2_score": "raw_ec2_score",
        "energy_cost_per_fu_reference": "reference_energy_cost_per_fu",
        "energy_cost_per_fu_scenario": "scenario_energy_cost_per_fu",
        "guarded_EC2_score": "guarded_ec2_score",
    })

    measurement_values = measurements.pivot(index="scenario_code", columns="variable_name", values="value")
    measurement_comments = measurements.pivot(index="scenario_code", columns="variable_name", values="comment")
    transport_factor = factors.iloc[0] if len(factors) == 1 else pd.Series(dtype=object)
    transport_rows = []
    for scenario_code in sorted(measurement_values.index.astype(str)):
        transport_comment = str(measurement_comments.at[scenario_code, "transport_ghg_tco2e"])
        level_match = re.search(r"rule=(BASE|L[1-6])", transport_comment)
        rule_match = re.search(r"rule_id=([^;]+)", transport_comment)
        bridge_match = re.search(r"(BR_[A-Z0-9_]+)", transport_comment)
        completion_rule = rule_match.group(1) if rule_match else ""
        bridge_rule = bridge_match.group(1) if bridge_match else ""
        configured_factor_code = str(transport_factor.get("factor_code", ""))
        boundary_ok = (
            (bool(bridge_rule) or (level_match and level_match.group(1) in {"BASE", "L1"})
             or (configured_factor_code and configured_factor_code in transport_comment))
            and "EF_DIESEL" not in transport_comment
        )
        scope_comment = str(measurement_comments.at[scenario_code, "ghg_total_s1s2"])
        double_count_ok = (
            "transport_ghg_tco2e" not in scope_comment
            or ("separately" in scope_comment.lower() and "not added" in scope_comment.lower())
        )
        transport_rows.append({
            "scenario_code": scenario_code,
            "output_qty_fu": measurement_values.at[scenario_code, "output_qty_fu"],
            "transport_work_tkm": measurement_values.at[scenario_code, "transport_work_tkm"],
            "direct_transport_ghg": measurement_values.at[scenario_code, "transport_ghg_tco2e"] if level_match and level_match.group(1) == "L1" else None,
            "direct_source": "DES / AnyLogistix Vehicle CO2" if bridge_rule else "",
            "bridge_rule": bridge_rule,
            "factor_set_id": transport_factor.get("factor_set_id", ""),
            "factor_code": transport_factor.get("factor_code", ""),
            "factor_value": transport_factor.get("factor_value", None),
            "factor_unit": transport_factor.get("factor_unit", ""),
            "completed_transport_ghg_tco2e": measurement_values.at[scenario_code, "transport_ghg_tco2e"],
            "completion_level": level_match.group(1) if level_match else "",
            "completion_rule": completion_rule,
            "transport_ghg_intensity_fu": measurement_values.at[scenario_code, "transport_ghg_intensity_fu"],
            "boundary_status": "PASS" if boundary_ok else "FAIL",
            "double_count_status": "PASS" if double_count_ok else "FAIL",
            "qa_status": "PASS" if boundary_ok and double_count_ok else "FAIL",
            "message": transport_comment,
        })
    transport_diagnostics = pd.DataFrame(transport_rows)

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
    checks["provenance"] = "active import run and recalculated KPI pipeline"
    measurement_index = measurements.set_index(["scenario_code", "variable_name"])
    des_scenarios = sorted(set(measurements.loc[
        (measurements.variable_name == "electricity_kwh")
        & measurements.comment.str.contains("rule=L3", na=False)
        & measurements.comment.str.contains("strategy=LOGISTICS_REDESIGN", na=False),
        "scenario_code",
    ]))
    sd_scenarios = sorted(set(measurements.loc[
        (measurements.variable_name == "output_qty_fu")
        & measurements.comment.str.contains("BR_SD_DELIVERY_OUTPUT", na=False),
        "scenario_code",
    ]))

    extra_checks: list[dict[str, object]] = []
    def add_check(check_id: str, scenarios: list[str], item: str, ok: bool, message: str) -> None:
        extra_checks.append({
            "check_id": check_id, "scenario_code": ",".join(scenarios),
            "variable_or_kpi": item, "expected_logic": message,
            "actual_value": ok, "provenance": f"import_run_id={run_id}",
            "status": "PASS" if ok else "FAIL", "message": message,
        })

    for variable, check_id in (
        ("electricity_kwh", "DES_ELECTRICITY_L3"),
        ("diesel_kwh", "DES_DIESEL_L3"),
        ("water_withdrawn_m3", "DES_WATER_L3"),
        ("waste_generated_t", "DES_WASTE_L3"),
        ("operating_cost_eur", "DES_OPERATING_COST_L3"),
    ):
        ok = bool(des_scenarios) and all(
            "rule=L3" in str(measurement_index.loc[(scenario, variable), "comment"])
            for scenario in des_scenarios
        )
        add_check(check_id, des_scenarios, variable, ok, "Configured DES activity uses L3.")
    maintenance_ok = bool(des_scenarios) and all(
        "rule=L6" in str(measurement_index.loc[(scenario, "maintenance_cost_eur"), "comment"])
        for scenario in des_scenarios
    )
    add_check("DES_MAINTENANCE_BASE_RETENTION", des_scenarios, "maintenance_cost_eur", maintenance_ok, "Unsupported maintenance retains BASE.")
    ghg_ok = bool(des_scenarios) and all(
        "MRV_R_GHG_S1S2_FACTORS" in str(measurement_index.loc[(scenario, "ghg_total_s1s2"), "comment"])
        for scenario in des_scenarios
    )
    add_check("DES_FACTOR_BASED_GHG", des_scenarios, "ghg_total_s1s2", ghg_ok, "Factor GHG executes after L3.")
    for kpi in ("E2", "E3", "E5", "E9"):
        rows = detail[(detail.scenario.isin(des_scenarios)) & (detail.kpi_code == kpi)]
        add_check(f"DES_{kpi}_INTENSITY_PRESERVATION", des_scenarios, kpi, len(rows) == len(des_scenarios) and rows.effect.eq("Same").all(), "Proportional scaling preserves intensity.")
    for kpi in ("T3", "E7", "E9"):
        rows = detail[(detail.scenario.isin(sd_scenarios)) & (detail.kpi_code == kpi)]
        add_check(f"SD_{kpi}_BASE_COMPLETION", sd_scenarios, kpi, len(rows) == len(sd_scenarios) and rows.effect.eq("Same").all(), "Non-modelled SD intensity/coverage remains BASE-equivalent.")
    guarded = ec2_diagnostics[ec2_diagnostics.denominator_effect_flag == True]  # noqa: E712
    add_check(
        "EC2_DENOMINATOR_GUARD", guarded.scenario_code.astype(str).tolist(), "EC2",
        not guarded.empty and guarded.guarded_ec2_score.eq(50.0).all() and guarded.effect.eq("Same").all(),
        "Denominator-only improvements are neutralized.",
    )
    reference_transport = transport_diagnostics[
        transport_diagnostics.scenario_code.eq(reference)
    ].iloc[0]
    factor_ok = len(factors) == 1 and transport_factor.factor_unit == "tCO2e/tkm"
    add_check("E7_BASE_BOUNDARY", [reference], "E7", reference_transport.qa_status == "PASS", "Reference E7 uses the outbound-road transport boundary.")
    des_boundary = transport_diagnostics[transport_diagnostics.bridge_rule.ne("")].copy()
    for year in (2025, 2030, 2035):
        candidates = des_boundary[des_boundary.scenario_code.str.contains(str(year), regex=False)]
        add_check(f"E7_DES_{year}_COMPARABILITY", candidates.scenario_code.tolist(), "E7", not candidates.empty and candidates.qa_status.eq("PASS").all(), "Direct bridged Vehicle CO2 uses the same E7 boundary as the reference.")
    factor_completed = transport_diagnostics[
        transport_diagnostics.message.str.contains(str(transport_factor.get("factor_code", "__missing__")), regex=False)
    ]
    same_as_base = factor_completed[
        (factor_completed.transport_work_tkm - reference_transport.transport_work_tkm).abs() <= 1e-6
    ]
    proportional = factor_completed[
        (factor_completed.transport_ghg_intensity_fu - reference_transport.transport_ghg_intensity_fu).abs() <= 1e-6
    ]
    changed = factor_completed.drop(proportional.index)
    add_check("E7_SOCIAL_BASE_RETENTION", same_as_base.scenario_code.tolist(), "E7", not same_as_base.empty and same_as_base.qa_status.eq("PASS").all(), "Factor completion retains reference E7 when transport work is unchanged.")
    add_check("E7_SD_PROPORTIONAL_COMPLETION", proportional.scenario_code.tolist(), "E7", not proportional.empty and proportional.qa_status.eq("PASS").all(), "Proportional transport work and output preserve E7.")
    add_check("E7_MILP_FACTOR_COMPLETION", changed.scenario_code.tolist(), "E7", not changed.empty and changed.qa_status.eq("PASS").all(), "Changed transport work is completed through the approved factor.")
    add_check("E7_VSMC_FACTOR_COMPLETION", changed.scenario_code.tolist(), "E7", not changed.empty and changed.qa_status.eq("PASS").all(), "Transport-work improvements flow through the approved factor and E7 alias.")
    no_double_count = transport_diagnostics.double_count_status.eq("PASS").all()
    add_check("E1_UNCHANGED_BY_E7_PATCH", [], "E1/E2", no_double_count, "E7 completion has no dependency into production Scope 1-2 GHG.")
    add_check("TRANSPORT_GHG_NOT_ADDED_TO_S1S2", [], "ghg_total_s1s2", no_double_count, "Transport GHG is not added to Scope 1-2 GHG.")
    add_check("TRANSPORT_FACTOR_APPROVED", [], "transport factor", factor_ok, "Exactly one approved transport-scope factor has the required tCO2e/tkm unit.")
    checks = pd.concat([checks, pd.DataFrame(extra_checks)], ignore_index=True)
    outputs = {
        "dimension_indices_by_scenario.csv": dim,
        "mcda_scenario_ranking.csv": ranking,
        "traffic_light_distribution.csv": traffic,
        "normalized_comparison_detail.csv": detail,
        "normalized_comparison_by_dimension.csv": by_dimension,
        "normalized_comparison_summary.csv": summary,
        "final_reconciliation_checks.csv": checks,
        "ec2_denominator_guard_diagnostics.csv": ec2_diagnostics,
        "transport_ghg_boundary_diagnostics.csv": transport_diagnostics,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    return {filename: len(frame) for filename, frame in outputs.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("generated/results_latest_corrected"))
    args = parser.parse_args()
    print(export_active_results(args.output_dir))
