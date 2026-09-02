from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from .agent import ASCAEngine, ASCAEvaluation
from .core import FORMAL_BOUNDS, SIZE_CLASSES, STRATEGIES

EXAMPLES = {
    "Automotive / ENERGY": "Medium-sized Romanian automotive component supplier with high logistics complexity, low renewable-energy use and moderate digital maturity.",
    "Cable assembly / DIGITAL": "Medium-sized cable-assembly manufacturer with limited digital maturity, traceability gaps and stable demand variability.",
    "Construction materials / ENERGY": "Large non-metallic construction-material producer, energy-intensive, with limited renewable energy and moderate circularity maturity.",
}


def _init_state():
    st.session_state.setdefault("asca_suggestion", None)


def render_asca_page(
    *,
    assets_dir: str | Path,
    on_configuration: Optional[Callable[[ASCAEvaluation], None]] = None,
    show_title: bool = True,
) -> None:
    """Independent Streamlit page; safe to mount inside SustainSCM without touching the KPI engine."""
    _init_state()
    engine = ASCAEngine(assets_dir)

    if show_title:
        st.title("SustainSCM AI Scenario Agent (ASCA)")
        st.caption(
            "Bounded natural-language configuration → domain gate → validated "
            "metamodel routing. ASCA configures; the metamodel predicts."
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
            "10 PASS · 5 CONDITIONAL · 2 parent-model-only outputs."
        )

    if interpret:
        try:
            st.session_state.asca_suggestion = engine.suggest(description)
            st.session_state.pop("asca_evaluation", None)
        except Exception as exc:
            st.error(str(exc))

    sug = st.session_state.asca_suggestion
    if sug is None:
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suggested archetype", sug.archetype)
    c2.metric("Suggested strategy", sug.strategy)
    c3.metric("Intensity", sug.intensity_label)
    c4.metric("Archetype text match", f"{sug.archetype_similarity:.2f}")
    st.caption("The values below are synthetic configuration values constrained by the documented experimental domain. Review/override before prediction.")

    with st.expander("Review / override ASCA configuration", expanded=True):
        archetype = st.selectbox("Archetype", [f"RO-A{i}" for i in range(1, 9)], index=[f"RO-A{i}" for i in range(1, 9)].index(sug.archetype))
        size_class = st.selectbox("Size class", SIZE_CLASSES, index=SIZE_CLASSES.index(sug.size_class))
        strategy = st.selectbox("Strategy family", STRATEGIES, index=STRATEGIES.index(sug.strategy))
        lam_default = 0.0 if strategy == "BASE" else sug.lambda_intensity
        lam = st.slider("Intervention intensity λ", 0.0, 1.0, float(lam_default), 0.05, disabled=(strategy == "BASE"))
        params: dict[str, float] = {}
        left, right = st.columns(2)
        for i, (feature, (lo, hi)) in enumerate(FORMAL_BOUNDS.items()):
            host = left if i % 2 == 0 else right
            with host:
                params[feature] = st.slider(feature, float(lo), float(hi), float(sug.parameters[feature]), float((hi-lo)/100), format="%.4f")

    if st.button(
        "Validate domain and run eligible metamodels",
        type="primary",
        width="stretch",
    ):
        try:
            ev = engine.evaluate(
                archetype=archetype, size_class=size_class, strategy=strategy,
                lambda_intensity=(0.0 if strategy == "BASE" else lam), parameters=params,
                suggestion=sug,
            )
            st.session_state.asca_evaluation = ev
        except Exception as exc:
            st.error(f"ASCA evaluation failed: {type(exc).__name__}: {exc}")
            st.stop()

    ev = st.session_state.get("asca_evaluation")
    if ev is None:
        st.stop()

    st.subheader("Applicability-domain gate")
    if ev.domain.status == "INSIDE_VALIDATED_DOMAIN":
        st.success("INSIDE VALIDATED DOMAIN — output-specific surrogate routing is enabled.")
    elif ev.domain.status == "NEAR_BOUNDARY":
        st.warning("NEAR BOUNDARY — parent-model route is preferred; surrogate predictions are blocked in this proof of concept.")
    else:
        st.error(f"{ev.domain.status} — surrogate predictions are blocked.")
    if ev.domain.violations:
        st.dataframe(pd.DataFrame(ev.domain.violations), width="stretch")

    st.subheader("Scenario configuration")
    scenario_record = ev.scenario_record()
    scenario_display = pd.DataFrame(
        {
            "parameter": list(scenario_record),
            "value": [str(value) for value in scenario_record.values()],
        }
    )
    st.dataframe(scenario_display, width="stretch", hide_index=True)
    st.caption(f"VSM-C diagnostic priority: {ev.model_row['priority_strategy']} (diagnostic only; it does not prune counterfactual strategies).")

    st.subheader("Metamodel routing and predictions")
    show_cols = ["module", "target", "prediction", "unit", "validation_status", "route", "holdout_nrmse", "holdout_spearman", "message"]
    st.dataframe(ev.predictions[show_cols], width="stretch", hide_index=True)

    eligible = ev.predictions[ev.predictions["prediction"].notna()].copy()
    if not eligible.empty:
        eligible["display_value"] = eligible["prediction"]
        fig = px.bar(
            eligible, x="target", y="display_value", color="module",
            pattern_shape="validation_status",
            hover_data=["unit", "validation_status", "holdout_nrmse", "holdout_spearman"],
            title="ASCA in-domain metamodel screening outputs"
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, width="stretch")

    pass_n = int((ev.predictions.route == "SURROGATE_SCREENING").sum())
    cond_n = int((ev.predictions.route == "SURROGATE_EXPLORATORY").sum())
    parent_n = int((ev.predictions.route == "PARENT_MODEL_REQUIRED").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("PASS predictions", pass_n)
    m2.metric("Conditional predictions", cond_n)
    m3.metric("Parent-model routes", parent_n)

    st.subheader("Reproducibility trace")
    trace = {
        "scenario": ev.scenario_record(),
        "natural_language_rules": sug.trace if sug else [],
        "domain_status": ev.domain.status,
        "environment": ev.environment,
        "output_routes": ev.predictions[["module", "target", "validation_status", "route"]].to_dict(orient="records"),
    }
    st.code(json.dumps(trace, indent=2, ensure_ascii=False), language="json")
    st.download_button("Download ASCA trace (JSON)", json.dumps(trace, indent=2).encode(), file_name=f"{ev.model_row['scenario_id']}_asca_trace.json", mime="application/json")
    st.download_button("Download metamodel screening (CSV)", ev.predictions.to_csv(index=False).encode(), file_name=f"{ev.model_row['scenario_id']}_metamodel_screening.csv", mime="text/csv")

    if on_configuration is not None:
        if st.button("Send validated configuration to SustainSCM callback", disabled=not ev.domain.surrogate_allowed):
            on_configuration(ev)
            st.success("ASCA evaluation passed to the application callback.")
