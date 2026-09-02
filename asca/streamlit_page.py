from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from .agent import ASCAEngine, ASCAEvaluation
from .core import FORMAL_BOUNDS, SIZE_CLASSES, STRATEGIES
from .interpretation import (
    build_base_counterfactual,
    build_des_interpretation,
    build_key_interpretations,
    build_milp_interpretation,
    build_recommended_action,
    build_relative_change_figure,
    build_sd_interpretation,
    compare_with_base,
    comparison_display_frame,
    comparison_export_frame,
    interpretation_payload,
    module_display_frame,
    strategy_priority_interpretation,
    summarize_configuration,
    summarize_routing,
)


EXAMPLES = {
    "Automotive / ENERGY": "Medium-sized Romanian automotive component supplier with high logistics complexity, low renewable-energy use and moderate digital maturity.",
    "Cable assembly / DIGITAL": "Medium-sized cable-assembly manufacturer with limited digital maturity, traceability gaps and stable demand variability.",
    "Construction materials / ENERGY": "Large non-metallic construction-material producer, energy-intensive, with limited renewable energy and moderate circularity maturity.",
}


def _init_state() -> None:
    st.session_state.setdefault("asca_suggestion", None)


def _render_domain_status(evaluation: ASCAEvaluation) -> None:
    """Display the existing domain decision without changing or clipping it."""
    st.subheader("Applicability-domain status")
    if evaluation.domain.status == "INSIDE_VALIDATED_DOMAIN":
        st.success(
            "INSIDE VALIDATED DOMAIN — this configuration is eligible for "
            "domain-aware surrogate screening. Individual outputs remain subject "
            "to their validation status."
        )
    elif evaluation.domain.status == "NEAR_BOUNDARY":
        st.warning(
            "NEAR BOUNDARY — surrogate results should be treated cautiously; "
            "parent-model confirmation is preferred and predictions remain blocked "
            "in this proof of concept."
        )
    else:
        st.error(
            f"{evaluation.domain.status} — surrogate prediction is blocked. Use the "
            "corresponding parent model or a validated adaptive-design extension."
        )
    if evaluation.domain.violations:
        st.dataframe(pd.DataFrame(evaluation.domain.violations), width="stretch")


def _render_routing_summary(evaluation: ASCAEvaluation) -> None:
    """Render route counts calculated from the actual output DataFrame."""
    counts = summarize_routing(evaluation.predictions)
    validated, exploratory, parent = st.columns(3)
    validated.metric("Validated screening", counts["validated_screening"])
    validated.caption(
        "PASS: rapid in-domain screening; confirm high-impact final decisions with "
        "the parent model."
    )
    exploratory.metric("Exploratory", counts["exploratory"])
    exploratory.caption(
        "CONDITIONAL: tendencies only; parent-model confirmation is required near "
        "decision boundaries or for final conclusions."
    )
    parent.metric("Parent model required", counts["parent_model_required"])
    parent.caption(
        "ASCA deliberately withholds estimates whose validation or domain status "
        "does not support surrogate substitution."
    )


def _render_module_view(
    title: str,
    module: str,
    summary: str,
    evaluation: ASCAEvaluation,
) -> None:
    """Render one module as prose and a unit-aware table, never a mixed-unit axis."""
    with st.expander(title, expanded=False):
        st.write(summary)
        st.dataframe(
            module_display_frame(evaluation, module),
            width="stretch",
            hide_index=True,
        )


def _technical_absolute_figure(evaluation: ASCAEvaluation):
    eligible = evaluation.predictions[
        evaluation.predictions["prediction"].notna()
    ].copy()
    if eligible.empty:
        return None
    figure = px.bar(
        eligible,
        x="target",
        y="prediction",
        color="module",
        pattern_shape="validation_status",
        hover_data=["unit", "validation_status", "holdout_nrmse", "holdout_spearman"],
        title="Technical metamodel outputs — absolute values (mixed units)",
        template="sustainscm",
    )
    figure.update_xaxes(tickangle=-45)
    figure.update_yaxes(title="Native model value — units differ by output")
    return figure


def render_asca_page(
    *,
    assets_dir: str | Path,
    on_configuration: Optional[Callable[[ASCAEvaluation], None]] = None,
    show_title: bool = True,
) -> None:
    """Render ASCA plus a downstream, deterministic interpretation layer.

    ASCA and the metamodel router remain authoritative for configuration, domain
    eligibility, prediction and validation status. This page only explains and
    formats those results and never creates a value for a withheld output.
    """
    _init_state()
    engine = ASCAEngine(assets_dir)

    if show_title:
        st.title("SustainSCM AI Scenario Agent (ASCA)")
        st.caption(
            "Bounded natural-language configuration → domain gate → validated "
            "metamodel routing. ASCA configures and interprets; the validated "
            "metamodels predict."
        )

    environment = engine.router.environment_status()
    if not environment["sklearn_exact_match"]:
        st.error(
            "The metamodel runtime is incompatible: scikit-learn "
            f"{environment['trained_sklearn']} is required, but "
            f"{environment['current_sklearn']} is installed. Predictions are disabled."
        )
        st.stop()

    st.info(
        "Scientific boundary: synthetic configurations are not company observations. "
        "FULL_MODEL_REQUIRED and out-of-domain requests are never silently predicted."
    )

    guide, research_scope = st.columns([1.35, 1], gap="large")
    with guide:
        st.markdown("#### 1 · Describe the configuration")
        st.caption(
            "Choose a Romanian proof-of-concept example or describe a new synthetic "
            "industrial configuration in plain language."
        )
        example_name = st.selectbox(
            "Proof-of-concept starting point",
            ["Custom", *EXAMPLES.keys()],
            index=2,
        )
        default_text = "" if example_name == "Custom" else EXAMPLES[example_name]
        description = st.text_area(
            "Industrial configuration",
            value=default_text,
            height=128,
            key=f"asca_desc_{example_name}",
        )
        interpret = st.button(
            "Interpret description with ASCA",
            type="primary",
            icon="🤖",
            width="stretch",
        )
    with research_scope:
        st.markdown("#### Romanian experimental scope")
        st.write(
            "ASCA uses the 8 Romanian industrial archetypes and the final surrogate "
            "artifacts derived from the thesis experiments."
        )
        scope_a, scope_b = st.columns(2)
        scope_a.metric("Archetypes", "8")
        scope_b.metric("Outputs", "17")
        st.caption(
            "Training design: 136 configurations · independent holdout: 32 · "
            "validation and routing are loaded from the supplied registry."
        )

    if interpret:
        try:
            st.session_state.asca_suggestion = engine.suggest(description)
            st.session_state.pop("asca_evaluation", None)
            st.session_state.pop("asca_base_evaluation", None)
        except Exception as exc:
            st.error(str(exc))

    suggestion = st.session_state.asca_suggestion
    if suggestion is None:
        st.stop()

    st.markdown("### ASCA interpretation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suggested archetype", suggestion.archetype)
    c2.metric("Suggested strategy", suggestion.strategy)
    c3.metric("Intensity", suggestion.intensity_label)
    c4.metric("Archetype text match", f"{suggestion.archetype_similarity:.2f}")
    st.caption(
        "This is the deterministic interpretation of the description. Review the "
        "bounded configuration before running any eligible metamodel."
    )

    with st.expander("2 · Review / override ASCA configuration", expanded=True):
        archetypes = [f"RO-A{i}" for i in range(1, 9)]
        archetype = st.selectbox(
            "Archetype", archetypes, index=archetypes.index(suggestion.archetype)
        )
        size_class = st.selectbox(
            "Size class",
            SIZE_CLASSES,
            index=SIZE_CLASSES.index(suggestion.size_class),
        )
        strategy = st.selectbox(
            "Strategy family",
            STRATEGIES,
            index=STRATEGIES.index(suggestion.strategy),
        )
        lambda_default = 0.0 if strategy == "BASE" else suggestion.lambda_intensity
        intensity = st.slider(
            "Intervention intensity λ",
            0.0,
            1.0,
            float(lambda_default),
            0.05,
            disabled=(strategy == "BASE"),
        )
        parameters: dict[str, float] = {}
        left, right = st.columns(2)
        for index, (feature, (low, high)) in enumerate(FORMAL_BOUNDS.items()):
            host = left if index % 2 == 0 else right
            with host:
                parameters[feature] = st.slider(
                    feature,
                    float(low),
                    float(high),
                    float(suggestion.parameters[feature]),
                    float((high - low) / 100),
                    format="%.4f",
                )

    if st.button(
        "3 · Validate domain and run eligible metamodels",
        type="primary",
        width="stretch",
    ):
        try:
            evaluation = engine.evaluate(
                archetype=archetype,
                size_class=size_class,
                strategy=strategy,
                lambda_intensity=(0.0 if strategy == "BASE" else intensity),
                parameters=parameters,
                suggestion=suggestion,
            )
            st.session_state.asca_evaluation = evaluation
            st.session_state.asca_base_evaluation = build_base_counterfactual(
                engine, evaluation
            )
        except Exception as exc:
            st.error(f"ASCA evaluation failed: {type(exc).__name__}: {exc}")
            st.stop()

    evaluation = st.session_state.get("asca_evaluation")
    if evaluation is None:
        st.stop()
    base_evaluation = st.session_state.get("asca_base_evaluation")
    if evaluation.domain.surrogate_allowed and base_evaluation is None:
        base_evaluation = build_base_counterfactual(engine, evaluation)
        st.session_state.asca_base_evaluation = base_evaluation
    comparison = compare_with_base(evaluation, base_evaluation)

    _render_domain_status(evaluation)

    st.markdown("## ASCA Decision-Support Interpretation")
    st.caption(
        "Deterministic explanation generated only from the structured configuration, "
        "domain decision, validation registry and actual metamodel routes."
    )
    st.markdown("#### What ASCA understood")
    st.write(summarize_configuration(evaluation, suggestion))

    tested, priority = st.columns(2)
    tested.metric("Strategy being evaluated", evaluation.model_row["strategy"])
    priority.metric("Diagnostic priority", evaluation.model_row["priority_strategy"])
    st.info(strategy_priority_interpretation(evaluation))

    st.markdown("#### Screening confidence and routing")
    _render_routing_summary(evaluation)

    st.markdown("## Scenario vs same-configuration BASE")
    st.caption(
        "The BASE companion preserves the same archetype, size and pre-intervention "
        "current-state factors. Only strategy=BASE and λ=0.0 change, and the companion "
        "passes through the same builder, domain gate and output router."
    )
    if base_evaluation is None:
        st.warning(
            "No BASE numerical comparison is available because the selected "
            "configuration is not eligible for surrogate screening."
        )
    elif not base_evaluation.domain.surrogate_allowed:
        st.warning(
            "The BASE companion is outside the validated domain. Paired numerical "
            "comparison is intentionally withheld."
        )
    elif comparison.empty:
        st.info("No outputs have admissible predictions in both the selected and BASE scenarios.")
    else:
        st.caption(f"BASE companion ID: {base_evaluation.model_row['scenario_id']}")
        st.dataframe(
            comparison_display_frame(comparison),
            width="stretch",
            hide_index=True,
        )
        relative_figure = build_relative_change_figure(comparison)
        if relative_figure is not None:
            st.plotly_chart(relative_figure, width="stretch")
            st.caption(
                "Values are mathematical relative changes, not automatic claims of "
                "sustainability improvement or deterioration."
            )

    st.markdown("## Module-by-module interpretation")
    _render_module_view(
        "MILP screening", "MILP", build_milp_interpretation(evaluation), evaluation
    )
    _render_module_view(
        "DES operational screening", "DES", build_des_interpretation(evaluation), evaluation
    )
    _render_module_view(
        "System Dynamics long-term screening",
        "SD",
        build_sd_interpretation(evaluation),
        evaluation,
    )

    st.markdown("#### Key interpretation")
    for message in build_key_interpretations(evaluation, comparison):
        st.markdown(f"- {message}")

    st.markdown("#### Recommended next action")
    st.info(build_recommended_action(evaluation))

    st.markdown("#### Interpretation and uncertainty")
    st.write(
        "PASS indicates validated rapid screening inside the experimental domain, "
        "not exact physical truth. CONDITIONAL indicates an exploratory tendency. "
        "Parent-model-required outputs are intentionally withheld. All results remain "
        "conditional on the applicability-domain gate, and high-impact final decisions "
        "should use the corresponding parent model."
    )

    trace = {
        "scenario": evaluation.scenario_record(),
        "natural_language_rules": suggestion.trace if suggestion else [],
        "domain_status": evaluation.domain.status,
        "environment": evaluation.environment,
        "output_routes": evaluation.predictions[
            ["module", "target", "validation_status", "route"]
        ].to_dict(orient="records"),
    }
    interpretation = interpretation_payload(evaluation, suggestion, comparison)
    scenario_id = evaluation.model_row["scenario_id"]

    st.markdown("## Downloads")
    download_a, download_b = st.columns(2)
    download_a.download_button(
        "Download ASCA trace (JSON)",
        json.dumps(trace, indent=2).encode(),
        file_name=f"{scenario_id}_asca_trace.json",
        mime="application/json",
        width="stretch",
    )
    download_b.download_button(
        "Download metamodel screening (CSV)",
        evaluation.predictions.to_csv(index=False).encode(),
        file_name=f"{scenario_id}_metamodel_screening.csv",
        mime="text/csv",
        width="stretch",
    )
    download_c, download_d = st.columns(2)
    download_c.download_button(
        "Download BASE comparison (CSV)",
        comparison_export_frame(comparison).to_csv(index=False).encode(),
        file_name=f"{scenario_id}_base_comparison.csv",
        mime="text/csv",
        disabled=comparison.empty,
        width="stretch",
    )
    download_d.download_button(
        "Download interpretation (JSON)",
        json.dumps(interpretation, indent=2, ensure_ascii=False).encode(),
        file_name=f"{scenario_id}_interpretation.json",
        mime="application/json",
        width="stretch",
    )

    with st.expander("Technical metamodel outputs — absolute values", expanded=False):
        st.warning(
            "Advanced technical view: the chart below intentionally preserves native "
            "model values with heterogeneous units and must not be read as a common scale."
        )
        scenario_record = evaluation.scenario_record()
        scenario_display = pd.DataFrame(
            {
                "parameter": list(scenario_record),
                "value": [str(value) for value in scenario_record.values()],
            }
        )
        st.dataframe(scenario_display, width="stretch", hide_index=True)
        show_columns = [
            "module",
            "target",
            "prediction",
            "unit",
            "validation_status",
            "route",
            "message",
        ]
        st.dataframe(
            evaluation.predictions[show_columns],
            width="stretch",
            hide_index=True,
        )
        absolute_figure = _technical_absolute_figure(evaluation)
        if absolute_figure is not None:
            st.plotly_chart(absolute_figure, width="stretch")

    with st.expander("Technical validation details", expanded=False):
        validation_columns = [
            "module",
            "target",
            "algorithm",
            "holdout_nrmse",
            "holdout_spearman",
            "validation_status",
            "route",
        ]
        st.dataframe(
            evaluation.predictions[validation_columns],
            width="stretch",
            hide_index=True,
        )

    if on_configuration is not None:
        if st.button(
            "Send validated configuration to SustainSCM callback",
            disabled=not evaluation.domain.surrogate_allowed,
        ):
            on_configuration(evaluation)
            st.success("ASCA evaluation passed to the application callback.")
