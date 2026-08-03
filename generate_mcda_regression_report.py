"""Generate auditable current-dataset sustainability and MCDA artifacts."""

from pathlib import Path

import pandas as pd
import plotly.express as px

from sustainsc.config import engine
from sustainsc.mcda import (
    DIMENSION_ORDER,
    build_mcda_input,
    calculate_mcda,
    compute_complete_dimension_indices,
    evaluate_scenario_eligibility,
)

OUTPUT_DIR = Path("generated") / "mcda_regression"


def main() -> None:
    catalog = pd.read_sql(
        """
        SELECT code AS kpi_code, dimension
        FROM sc_kpi
        WHERE code NOT IN ('ENV_INDEX','ECO_INDEX','SOC_INDEX','TECH_INDEX','SUSTAIN_INDEX')
        ORDER BY code
        """,
        engine,
    )
    raw = pd.read_sql(
        """
        SELECT s.code AS scenario_code, k.code AS kpi_code, r.value AS raw_value
        FROM sc_kpi_result r
        JOIN sc_scenario s ON s.id = r.scenario_id
        JOIN sc_kpi k ON k.id = r.kpi_id
        WHERE k.code NOT IN ('ENV_INDEX','ECO_INDEX','SOC_INDEX','TECH_INDEX','SUSTAIN_INDEX')
        """,
        engine,
    )
    normalized = pd.read_sql(
        """
        SELECT s.code AS scenario_code, k.code AS kpi_code,
               n.normalized_value
        FROM sc_kpi_normalized_result n
        JOIN sc_scenario s ON s.id = n.scenario_id
        JOIN sc_kpi k ON k.id = n.kpi_id
        """,
        engine,
    )
    mrv = pd.read_sql(
        """
        SELECT s.code AS scenario_code,
               COUNT(DISTINCT m.variable_name) AS completed_mrv_count
        FROM sc_measurement m
        JOIN sc_scenario s ON s.id = m.scenario_id
        GROUP BY s.code
        """,
        engine,
    )
    rules = pd.read_csv("data/kpi_normalization_rules.csv")
    rules = rules[rules["context_id"].astype(str).str.strip() == "aggregates_ton"].copy()
    local_weights = rules.drop_duplicates("kpi_code").set_index("kpi_code")["weight"]
    dimension_weights = pd.Series(0.25, index=DIMENSION_ORDER)
    rules["local_weight"] = rules["weight"] / rules.groupby("dimension")["weight"].transform("sum")
    rules["global_weight"] = rules["local_weight"] * rules["dimension"].map(dimension_weights)
    global_weights = (
        rules.drop_duplicates("kpi_code").set_index("kpi_code")["global_weight"].reindex(catalog["kpi_code"])
    )

    eligibility = evaluate_scenario_eligibility(raw, normalized, catalog)
    dimension_long, incomplete_dimensions = compute_complete_dimension_indices(
        normalized, catalog, local_weights
    )
    dimension_wide = dimension_long.pivot(
        index="scenario_code", columns="dimension", values="dimension_index"
    ).reset_index()
    mcda_input = build_mcda_input(normalized, global_weights, eligibility)
    result = calculate_mcda(mcda_input, eligibility)

    report = (
        eligibility.merge(mrv, on="scenario_code", how="left")
        .merge(dimension_wide, on="scenario_code", how="left")
        .merge(result.wsm, on="scenario_code", how="left")
        .merge(result.topsis, on="scenario_code", how="left")
    )
    report["eligible"] = report["mcda_eligible"]
    report["exclusion_reason"] = report["reason"]
    report = report[
        [
            "scenario_code",
            "completed_mrv_count",
            "raw_kpi_count",
            "normalized_kpi_count",
            *DIMENSION_ORDER,
            "WSM_score",
            "TOPSIS_score",
            "eligible",
            "exclusion_reason",
        ]
    ].sort_values("scenario_code")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "scenario_regression_report.csv", index=False)
    eligibility.to_csv(OUTPUT_DIR / "mcda_eligibility.csv", index=False)
    incomplete_dimensions.to_csv(OUTPUT_DIR / "incomplete_dimensions.csv", index=False)
    pd.DataFrame(
        {
            "diagnostic": list(result.diagnostics),
            "value": [str(value) for value in result.diagnostics.values()],
        }
    ).to_csv(OUTPUT_DIR / "mcda_diagnostics.csv", index=False)

    profile = dimension_long.copy()
    px.bar(
        profile,
        x="scenario_code",
        y="dimension_index",
        color="dimension",
        barmode="group",
        category_orders={"dimension": list(DIMENSION_ORDER)},
        title="Validated complete dimension profiles",
    ).write_html(OUTPUT_DIR / "dimension_profiles.html")
    px.bar(
        result.wsm.sort_values("WSM_score"),
        x="WSM_score",
        y="scenario_code",
        orientation="h",
        title="Validated WSM ranking",
    ).write_html(OUTPUT_DIR / "wsm_ranking.html")
    px.bar(
        result.topsis.sort_values("TOPSIS_score"),
        x="TOPSIS_score",
        y="scenario_code",
        orientation="h",
        title="Validated TOPSIS ranking",
    ).write_html(OUTPUT_DIR / "topsis_ranking.html")
    print(report.to_string(index=False))
    print(f"Wrote regression artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
