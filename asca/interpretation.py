"""Deterministic, descriptive decision support for ASCA screening results.

This module does not generate scientific evidence or alter metamodel outputs. It
formats existing ASCA evaluations, constructs a separately validated in-memory
BASE counterfactual, and explains the resulting routes and mathematical deltas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from sustainsc.ui.chart_theme import ranking_chart_height
from sustainsc.ui.theme import PRIMARY, WARNING

from .agent import ASCAEngine, ASCAEvaluation
from .core import ASCASuggestion, FORMAL_BOUNDS


TARGET_LABELS = {
    "milp_total_cost": "Total system cost",
    "milp_total_co2_t": "Total GHG emissions",
    "milp_energy_kwh": "Energy use",
    "milp_transport_work_tkm": "Transport work",
    "milp_capacity_util": "Capacity utilization",
    "des_throughput_rate_mean": "Throughput rate",
    "des_service_pct_mean": "Service level",
    "des_mean_lead_time_mean": "Mean lead time",
    "des_mean_wait_mean": "Mean waiting time",
    "des_logistics_cost_mean": "Logistics cost",
    "des_transport_co2_t_mean": "Transport CO2e",
    "sd_ghg_2030_t": "GHG emissions in 2030",
    "sd_ghg_2035_t": "GHG emissions in 2035",
    "sd_cum_ghg_t": "Cumulative GHG emissions",
    "sd_cum_output": "Cumulative output",
    "sd_oee_2035": "OEE in 2035",
    "sd_digital_2035": "Digital maturity in 2035",
}

RULE_LABELS = {
    "QL_RENEW_LOW": "low renewable-energy use",
    "QL_RENEW_HIGH": "high renewable-energy use",
    "QL_DIGITAL_LOW": "limited digital maturity",
    "QL_DIGITAL_MOD": "moderate digital maturity",
    "QL_DIGITAL_HIGH": "advanced digital maturity",
    "QL_CIRC_LOW": "limited circularity maturity",
    "QL_CIRC_MOD": "moderate circularity maturity",
    "QL_CIRC_HIGH": "advanced circularity maturity",
    "QL_SOCIAL_LOW": "limited social/workforce maturity",
    "QL_SOCIAL_MOD": "moderate social/workforce maturity",
    "QL_SOCIAL_HIGH": "advanced social/workforce maturity",
    "QL_OEE_LOW": "low OEE",
    "QL_OEE_MOD": "moderate OEE",
    "QL_OEE_HIGH": "high OEE",
    "QL_LOGISTICS_HIGH": "high logistics exposure",
    "QL_LOGISTICS_LOW": "low logistics exposure",
    "QL_DEMAND_VAR_HIGH": "volatile demand variability",
    "QL_DEMAND_VAR_LOW": "stable demand variability",
    "QL_DEMAND_LOAD_HIGH": "high demand load",
    "QL_DEMAND_LOAD_LOW": "low demand load",
    "QL_RESOURCE_HIGH": "high resource intensity",
    "QL_RESOURCE_LOW": "low resource intensity",
}

ALLOWED_ROUTES = {"SURROGATE_SCREENING", "SURROGATE_EXPLORATORY"}
COMPARISON_EXPORT_COLUMNS = [
    "module",
    "target",
    "unit",
    "validation_status",
    "base_prediction",
    "scenario_prediction",
    "absolute_change",
    "relative_change_pct",
    "route",
]


def humanize_target(target: str) -> str:
    """Return a stable user-facing label without altering registry identifiers."""
    return TARGET_LABELS.get(target, target.replace("_", " ").title())


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def format_prediction(value: Any, unit: str) -> str:
    """Format a prediction for the UI while preserving native values in exports."""
    if not _finite(value):
        return "Withheld"
    number = float(value)
    if unit == "EUR/y":
        return f"€{number / 1_000_000:.2f} million/year" if abs(number) >= 1_000_000 else f"€{number:,.0f}/year"
    if unit == "kWh/y":
        return f"{number / 1_000:,.1f} MWh/year" if abs(number) >= 1_000 else f"{number:,.0f} kWh/year"
    if unit == "fraction":
        return f"{number * 100:.2f}%"
    if unit == "%":
        return f"{number:.2f}%"
    if unit == "FU/min":
        return f"{number:.4f} FU/min"
    if unit == "min":
        return f"{number:.2f} min"
    if unit == "t CO2e/y":
        return f"{number:,.2f} t CO2e/year"
    if unit.startswith("t CO2e ("):
        return f"{number:,.2f} {unit}"
    if unit.startswith("FU ("):
        return f"{number / 1_000_000:.2f} million FU" if abs(number) >= 1_000_000 else f"{number:,.0f} FU"
    if unit in {"t·km/y", "tÂ·km/y"}:
        return f"{number:,.0f} t·km/year"
    return f"{number:,.3f} {unit}".strip()


def format_change(value: Any, unit: str) -> str:
    """Format a signed mathematical delta without assigning desirability."""
    if not _finite(value):
        return "Not available"
    number = float(value)
    if unit == "fraction":
        return f"{number * 100:+.2f} percentage points"
    if unit == "%":
        return f"{number:+.2f} percentage points"
    sign = "+" if number > 0 else ""
    return sign + format_prediction(number, unit)


def _level(feature: str, value: float) -> str:
    lo, hi = FORMAL_BOUNDS[feature]
    position = (float(value) - lo) / (hi - lo)
    if position < 1 / 3:
        return "low"
    if position > 2 / 3:
        return "high"
    return "moderate"


def _intensity_label(value: float, strategy: str) -> str:
    if strategy == "BASE" or abs(value) < 1e-12:
        return "baseline"
    if value <= 0.35:
        return "low"
    if value <= 0.65:
        return "moderate"
    return "high"


def summarize_configuration(
    evaluation: ASCAEvaluation,
    suggestion: ASCASuggestion | None = None,
) -> str:
    """Describe the evaluated configuration from its actual structured fields."""
    row = evaluation.model_row
    strategy = str(row["strategy"])
    intensity = float(row["lambda_intensity"])
    summary = (
        f"ASCA interpreted the request as a {row['size_class']}-sized "
        f"{row['archetype']} {str(row['archetype_name']).lower()}. The selected "
        f"counterfactual evaluates the {strategy} strategy at "
        f"{_intensity_label(intensity, strategy)} intervention intensity "
        f"(λ = {intensity:.2f}). The configured current state combines "
        f"{_level('renewable_share', row['renewable_share'])} renewable-energy "
        f"penetration, {_level('zD', row['zD'])} digital/MRV maturity, "
        f"{_level('zC', row['zC'])} circularity maturity, "
        f"{_level('zS', row['zS'])} social maturity, "
        f"{_level('distance_mult', row['distance_mult'])} logistics-distance "
        f"exposure and OEE of {float(row['oee']) * 100:.1f}%."
    )
    if suggestion is not None:
        cues = [
            RULE_LABELS[entry.split(":", 1)[0]]
            for entry in suggestion.trace
            if entry.split(":", 1)[0] in RULE_LABELS
        ]
        if cues:
            summary += " Explicit language cues applied: " + ", ".join(cues) + "."
    return summary


def strategy_priority_interpretation(evaluation: ASCAEvaluation) -> str:
    """Keep the tested strategy distinct from the diagnostic priority."""
    tested = str(evaluation.model_row["strategy"])
    priority = str(evaluation.model_row["priority_strategy"])
    if tested == priority:
        return (
            f"Strategy being evaluated: {tested}. Diagnostic priority: {priority}. "
            "The diagnostic signal is aligned with the requested counterfactual."
        )
    return (
        f"Strategy being evaluated: {tested}. Diagnostic priority: {priority}. "
        f"The requested/tested {tested} strategy is retained for counterfactual "
        f"evaluation. The diagnostic configuration identifies {priority} as a broader "
        f"priority, so a separate {priority} counterfactual may be useful for comparison."
    )


def summarize_routing(predictions: pd.DataFrame) -> dict[str, int]:
    """Count actual output routes rather than assuming registry totals."""
    routes = predictions["route"].astype(str)
    screening = int(routes.eq("SURROGATE_SCREENING").sum())
    exploratory = int(routes.eq("SURROGATE_EXPLORATORY").sum())
    parent = int(len(routes) - screening - exploratory)
    return {
        "validated_screening": screening,
        "exploratory": exploratory,
        "parent_model_required": parent,
    }

def build_base_counterfactual(
    engine: ASCAEngine,
    selected: ASCAEvaluation,
) -> ASCAEvaluation | None:
    """Evaluate an in-memory BASE companion through the same builder and router.

    The primary company/current-state factors are copied exactly. Only ``strategy``
    and ``lambda_intensity`` are changed before the normal ASCA evaluation path.
    No empirical REEL_BASE data or production database state is used.
    """
    if not selected.domain.surrogate_allowed:
        return None
    parameters = {name: float(selected.model_row[name]) for name in FORMAL_BOUNDS}
    base = engine.evaluate(
        archetype=str(selected.model_row["archetype"]),
        size_class=str(selected.model_row["size_class"]),
        strategy="BASE",
        lambda_intensity=0.0,
        parameters=parameters,
        suggestion=None,
    )
    base.model_row["scenario_id"] = f"{selected.model_row['scenario_id']}-BASE-CF"
    return base


def _empty_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *COMPARISON_EXPORT_COLUMNS,
            "indicator",
            "change_direction",
            "interpretation_availability",
            "base_route",
        ]
    )


def compare_with_base(
    selected: ASCAEvaluation,
    base: ASCAEvaluation | None,
) -> pd.DataFrame:
    """Return paired mathematical deltas only when both evaluations allow them."""
    if (
        base is None
        or not selected.domain.surrogate_allowed
        or not base.domain.surrogate_allowed
    ):
        return _empty_comparison()

    selected_frame = selected.predictions.rename(
        columns={"prediction": "scenario_prediction"}
    )
    base_frame = base.predictions[
        ["module", "target", "prediction", "route"]
    ].rename(columns={"prediction": "base_prediction", "route": "base_route"})
    paired = selected_frame.merge(base_frame, on=["module", "target"], how="inner")
    paired = paired[
        paired["route"].isin(ALLOWED_ROUTES)
        & paired["base_route"].isin(ALLOWED_ROUTES)
        & paired["scenario_prediction"].notna()
        & paired["base_prediction"].notna()
    ].copy()
    if paired.empty:
        return _empty_comparison()

    paired["absolute_change"] = (
        paired["scenario_prediction"].astype(float)
        - paired["base_prediction"].astype(float)
    )
    paired["relative_change_pct"] = np.where(
        paired["base_prediction"].astype(float).abs() > 1e-12,
        paired["absolute_change"] / paired["base_prediction"].astype(float) * 100,
        np.nan,
    )
    paired["indicator"] = paired["target"].map(humanize_target)
    paired["change_direction"] = np.select(
        [
            np.isclose(paired["absolute_change"], 0.0, rtol=1e-9, atol=1e-12),
            paired["absolute_change"] > 0,
        ],
        ["No material change", "Increase"],
        default="Decrease",
    )
    paired["interpretation_availability"] = np.where(
        paired["validation_status"].eq("PASS"),
        "Validated screening",
        "Exploratory only",
    )
    return paired.reset_index(drop=True)


def comparison_export_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return native-unit comparison columns for CSV export."""
    if comparison.empty:
        return pd.DataFrame(columns=COMPARISON_EXPORT_COLUMNS)
    return comparison[COMPARISON_EXPORT_COLUMNS].copy()


def comparison_display_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return a user-formatted table without changing raw comparison values."""
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "Module",
                "Indicator",
                "Validation status",
                "BASE",
                "Selected scenario",
                "Absolute change",
                "Relative change (%)",
                "Unit",
                "Interpretation availability",
            ]
        )
    display = pd.DataFrame(
        {
            "Module": comparison["module"],
            "Indicator": comparison["indicator"],
            "Validation status": comparison["validation_status"],
            "BASE": [
                format_prediction(value, unit)
                for value, unit in zip(comparison["base_prediction"], comparison["unit"])
            ],
            "Selected scenario": [
                format_prediction(value, unit)
                for value, unit in zip(comparison["scenario_prediction"], comparison["unit"])
            ],
            "Absolute change": [
                format_change(value, unit)
                for value, unit in zip(comparison["absolute_change"], comparison["unit"])
            ],
            "Relative change (%)": comparison["relative_change_pct"].map(
                lambda value: f"{float(value):+.2f}%" if _finite(value) else "Not available"
            ),
            "Unit": comparison["unit"],
            "Interpretation availability": comparison[
                "interpretation_availability"
            ],
        }
    )
    return display


def build_relative_change_figure(comparison: pd.DataFrame):
    """Plot dimensionless mathematical changes using the shared SustainSCM theme."""
    plot = comparison[comparison["relative_change_pct"].notna()].copy()
    if plot.empty:
        return None
    plot = plot.sort_values(["relative_change_pct", "indicator"])
    plot["BASE value"] = [
        format_prediction(value, unit)
        for value, unit in zip(plot["base_prediction"], plot["unit"])
    ]
    plot["Scenario value"] = [
        format_prediction(value, unit)
        for value, unit in zip(plot["scenario_prediction"], plot["unit"])
    ]
    plot["Absolute change"] = [
        format_change(value, unit)
        for value, unit in zip(plot["absolute_change"], plot["unit"])
    ]
    categories = plot["indicator"].tolist()
    figure = px.bar(
        plot,
        x="relative_change_pct",
        y="indicator",
        orientation="h",
        color="validation_status",
        color_discrete_map={"PASS": PRIMARY, "CONDITIONAL": WARNING},
        category_orders={"indicator": categories},
        labels={
            "relative_change_pct": "Mathematical change vs BASE (%)",
            "indicator": "Indicator",
            "validation_status": "Validation status",
        },
        custom_data=[
            "module",
            "BASE value",
            "Scenario value",
            "Absolute change",
            "unit",
            "validation_status",
        ],
        title="Relative change vs same-configuration BASE",
        template="sustainscm",
    )
    figure.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>Module: %{customdata[0]}"
            "<br>BASE: %{customdata[1]}<br>Selected: %{customdata[2]}"
            "<br>Absolute change: %{customdata[3]}"
            "<br>Relative change: %{x:+.2f}%<br>Unit: %{customdata[4]}"
            "<br>Validation: %{customdata[5]}<extra></extra>"
        )
    )
    figure.add_vline(x=0, line_width=1, line_color="#71828A")
    figure.update_layout(
        height=ranking_chart_height(len(plot)),
        margin={"l": 210, "r": 35, "t": 75, "b": 55},
    )
    return figure


def _result_row(evaluation: ASCAEvaluation, target: str) -> pd.Series | None:
    rows = evaluation.predictions[evaluation.predictions["target"].eq(target)]
    return None if rows.empty else rows.iloc[0]


def _result_sentence(evaluation: ASCAEvaluation, target: str) -> str:
    row = _result_row(evaluation, target)
    label = humanize_target(target)
    if row is None:
        return f"{label} is not present in the model registry."
    if not _finite(row["prediction"]):
        return (
            f"{label} is intentionally withheld; the {row['module']} parent model "
            "is required for this output."
        )
    value = format_prediction(row["prediction"], str(row["unit"]))
    if row["validation_status"] == "CONDITIONAL":
        return f"{label} has an exploratory estimate of {value} (CONDITIONAL)."
    return f"{label} has a validated screening estimate of {value}."


def build_milp_interpretation(evaluation: ASCAEvaluation) -> str:
    """Build a deterministic MILP summary from actual routed outputs."""
    targets = [
        "milp_total_cost",
        "milp_capacity_util",
        "milp_energy_kwh",
        "milp_total_co2_t",
        "milp_transport_work_tkm",
    ]
    return " ".join(_result_sentence(evaluation, target) for target in targets)


def build_des_interpretation(evaluation: ASCAEvaluation) -> str:
    """Build a deterministic DES operational summary."""
    targets = [
        "des_throughput_rate_mean",
        "des_service_pct_mean",
        "des_mean_lead_time_mean",
        "des_mean_wait_mean",
        "des_logistics_cost_mean",
        "des_transport_co2_t_mean",
    ]
    return " ".join(_result_sentence(evaluation, target) for target in targets)


def build_sd_interpretation(evaluation: ASCAEvaluation) -> str:
    """Build a deterministic long-term System Dynamics summary."""
    targets = [
        "sd_ghg_2030_t",
        "sd_ghg_2035_t",
        "sd_cum_ghg_t",
        "sd_cum_output",
    ]
    sentences = [_result_sentence(evaluation, target) for target in targets]
    oee = _result_row(evaluation, "sd_oee_2035")
    if oee is not None and _finite(oee["prediction"]):
        sentences.append(
            "Configured current-state OEE is "
            f"{float(evaluation.model_row['oee']) * 100:.2f}%; the 2035 screening "
            f"estimate is {format_prediction(oee['prediction'], str(oee['unit']))}."
        )
    else:
        sentences.append(_result_sentence(evaluation, "sd_oee_2035"))
    digital = _result_row(evaluation, "sd_digital_2035")
    if digital is not None and _finite(digital["prediction"]):
        sentences.append(
            "Configured current-state digital maturity is "
            f"{float(evaluation.model_row['zD']) * 100:.2f}%; the 2035 screening "
            f"estimate is {format_prediction(digital['prediction'], str(digital['unit']))}."
        )
    else:
        sentences.append(_result_sentence(evaluation, "sd_digital_2035"))
    return " ".join(sentences)


def module_display_frame(evaluation: ASCAEvaluation, module: str) -> pd.DataFrame:
    """Format one module without combining heterogeneous values on an axis."""
    frame = evaluation.predictions[evaluation.predictions["module"].eq(module)].copy()
    return pd.DataFrame(
        {
            "Indicator": frame["target"].map(humanize_target),
            "Screening value": [
                format_prediction(value, unit)
                for value, unit in zip(frame["prediction"], frame["unit"])
            ],
            "Validation status": frame["validation_status"],
            "Route": frame["route"],
        }
    )


def build_key_interpretations(
    evaluation: ASCAEvaluation,
    comparison: pd.DataFrame,
) -> list[str]:
    """Identify traceable patterns without asserting unsupported desirability."""
    messages: list[str] = []
    counts = summarize_routing(evaluation.predictions)
    messages.append(
        f"The {evaluation.model_row['strategy']} counterfactual provides "
        f"{counts['validated_screening']} validated screening outputs and "
        f"{counts['exploratory']} exploratory outputs inside the current routing."
    )
    relative = comparison[comparison["relative_change_pct"].notna()].copy()
    if not relative.empty:
        leading = relative.assign(
            magnitude=relative["relative_change_pct"].abs()
        ).nlargest(2, "magnitude")
        fragments = [
            f"{row.indicator}: {row.change_direction.lower()} of "
            f"{abs(float(row.relative_change_pct)):.2f}%"
            for row in leading.itertuples()
        ]
        messages.append(
            "Largest mathematical changes versus the same-configuration BASE are "
            + "; ".join(fragments)
            + ". These directions do not by themselves imply improvement."
        )
    withheld = evaluation.predictions[evaluation.predictions["prediction"].isna()]
    if not withheld.empty:
        labels = [humanize_target(target) for target in withheld["target"]]
        messages.append(
            "Parent-model confirmation remains mandatory for: " + ", ".join(labels) + "."
        )
    if counts["exploratory"]:
        messages.append(
            "CONDITIONAL results indicate exploratory tendencies only and must not be "
            "treated as validated final evidence."
        )
    return messages[:4]


def build_recommended_action(evaluation: ASCAEvaluation) -> str:
    """Recommend the next traceable workflow action without claiming optimality."""
    if not evaluation.domain.surrogate_allowed:
        return (
            "Execute the relevant MILP, DES or System Dynamics parent model, or extend "
            "the experimental design through a validated adaptive-design process. "
            "Surrogate screening is blocked for this configuration."
        )
    counts = summarize_routing(evaluation.predictions)
    tested = str(evaluation.model_row["strategy"])
    priority = str(evaluation.model_row["priority_strategy"])
    actions = ["Continue rapid screening for outputs routed as PASS"]
    if counts["exploratory"]:
        actions.append("treat CONDITIONAL outputs as exploratory tendencies")
    if counts["parent_model_required"]:
        modules = sorted(
            set(
                evaluation.predictions.loc[
                    ~evaluation.predictions["route"].isin(ALLOWED_ROUTES), "module"
                ].astype(str)
            )
        )
        actions.append("confirm withheld outputs with the " + "/".join(modules) + " parent model")
    if tested != priority:
        actions.append(f"consider a separate {priority} counterfactual alongside {tested}")
    return "; ".join(actions) + "."


def interpretation_payload(
    evaluation: ASCAEvaluation,
    suggestion: ASCASuggestion | None,
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-ready explanation payload separate from the original trace."""
    return {
        "scenario_id": evaluation.model_row["scenario_id"],
        "interpreted_configuration": summarize_configuration(evaluation, suggestion),
        "domain_status": evaluation.domain.status,
        "route_counts": summarize_routing(evaluation.predictions),
        "tested_strategy": evaluation.model_row["strategy"],
        "diagnostic_priority": evaluation.model_row["priority_strategy"],
        "strategy_interpretation": strategy_priority_interpretation(evaluation),
        "key_interpretation": build_key_interpretations(evaluation, comparison),
        "recommended_next_action": build_recommended_action(evaluation),
    }
