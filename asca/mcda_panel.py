"""Streamlit presentation for the read-only SustainSCM MCDA56 evidence layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .agent import ASCAEvaluation
from .core import FORMAL_BOUNDS
from .mcda_evidence import (
    MCDAEvidence,
    get_archetype_ranking,
    get_archetype_strategy_evidence,
    get_completion_robustness,
    get_weight_robustness,
    load_mcda_evidence,
)
from .mcda_visuals import build_mean_rank_figure, build_reference_profile_figure
from .robustness_interpreter import (
    COMPLETION_LABELS,
    PROFILE_LABELS,
    build_decision_evidence_payload,
    build_decision_evidence_status,
    build_evidence_scope,
    build_mcda_interpretation,
    build_recommended_next_action,
    build_strategy_comparison_interpretation,
    classify_strategy_evidence,
    summarize_completion_robustness,
    summarize_cross_archetype_robustness,
    summarize_diagnostic_agreement,
    summarize_weight_robustness,
)


@dataclass(frozen=True)
class DecisionEvidenceView:
    """Prepared display/export state derived only from retrieved evidence."""

    evidence: MCDAEvidence
    selected: dict[str, Any]
    leader: dict[str, Any]
    weight: dict[str, Any]
    completion: dict[str, Any]
    scope: dict[str, str]
    classification: str
    status: str
    recommended_action: str
    interpretation: list[str]
    payload: dict[str, Any]
    ranking: pd.DataFrame


def is_reference_anchor_configuration(
    evaluation: ASCAEvaluation, assets_dir: str | Path, *, atol: float = 1e-9
) -> bool:
    """Check the current inputs against the existing structured anchor definition."""

    development = pd.read_csv(Path(assets_dir) / "01_design" / "development_136.csv")
    candidates = development.loc[
        development["stage"].eq("ANCHOR")
        & development["archetype"].eq(evaluation.model_row["archetype"])
        & development["strategy"].eq(evaluation.model_row["strategy"])
    ]
    if candidates.empty:
        return False
    anchor = candidates.iloc[0]
    if str(anchor["size_class"]) != str(evaluation.model_row["size_class"]):
        return False
    fields = ["lambda_intensity", *FORMAL_BOUNDS]
    return all(
        abs(float(anchor[field]) - float(evaluation.model_row[field])) <= atol
        for field in fields
    )


def prepare_decision_evidence(
    evaluation: ASCAEvaluation,
    *,
    assets_dir: str | Path,
    evidence: MCDAEvidence | None = None,
) -> DecisionEvidenceView:
    """Prepare deterministic MCDA display state without altering the ASCA evaluation."""

    data = evidence or load_mcda_evidence()
    archetype = str(evaluation.model_row["archetype"])
    strategy = str(evaluation.model_row["strategy"])
    selected = get_archetype_strategy_evidence(archetype, strategy, data)
    leader_strategy = (
        selected["reference_leader"]["strategy"]
        or selected["reference_leader"]["wsm_strategy"]
    )
    leader = get_archetype_strategy_evidence(archetype, leader_strategy, data)
    weight = get_weight_robustness(archetype, leader_strategy, data)
    completion = get_completion_robustness(archetype, leader_strategy, data)
    scope = build_evidence_scope(
        evaluation.domain.status,
        is_reference_anchor_configuration(evaluation, assets_dir),
    )
    parent_required = int(
        evaluation.predictions["validation_status"].eq("FULL_MODEL_REQUIRED").sum()
    )
    status = build_decision_evidence_status(
        domain_status=evaluation.domain.status,
        parent_model_required_count=parent_required,
        selected_evidence=selected,
    )
    action = build_recommended_next_action(
        domain_status=evaluation.domain.status,
        parent_model_required_count=parent_required,
        selected_evidence=selected,
    )
    interpretation = build_mcda_interpretation(
        selected_evidence=selected,
        leader_evidence=leader,
        weight=weight,
        completion=completion,
        domain_status=evaluation.domain.status,
        recommended_action=action,
    )
    payload = build_decision_evidence_payload(
        scenario_id=str(evaluation.model_row["scenario_id"]),
        domain_status=evaluation.domain.status,
        selected_evidence=selected,
        weight=weight,
        completion=completion,
        evidence_scope=scope,
        decision_status=status,
        recommended_action=action,
    )
    return DecisionEvidenceView(
        evidence=data,
        selected=selected,
        leader=leader,
        weight=weight,
        completion=completion,
        scope=scope,
        classification=classify_strategy_evidence(selected),
        status=status,
        recommended_action=action,
        interpretation=interpretation,
        payload=payload,
        ranking=get_archetype_ranking(archetype, data),
    )


def render_mcda_interpretation(view: DecisionEvidenceView) -> None:
    """Render the bounded interpretation subsection in the main ASCA narrative."""

    st.markdown("#### MCDA and Robustness Interpretation")
    for paragraph in view.interpretation:
        st.write(paragraph)


def _rank_metric(rank: int | None, alternatives: int) -> str:
    return "Benchmark" if rank is None else f"#{rank} / {alternatives}"


def _score_metric(score: float | None) -> str:
    return "Not ranked" if score is None else f"{score:.3f}"


def _render_local_panel(view: DecisionEvidenceView) -> None:
    selected = view.selected
    local = selected["local_mcda"]
    leader = selected["reference_leader"]
    st.markdown("### Reference-Archetype MCDA Evidence")
    a, b, c, d = st.columns(4)
    a.metric("Selected archetype", selected["archetype"])
    b.metric("Strategy being evaluated", selected["strategy"])
    c.metric("Reference WSM leader", leader["wsm_strategy"])
    d.metric("Reference TOPSIS leader", leader["topsis_strategy"])
    wsm_rank, topsis_rank, wsm_score, topsis_score = st.columns(4)
    wsm_rank.metric("Selected strategy WSM rank", _rank_metric(local["wsm_rank"], local["alternatives"]))
    topsis_rank.metric("Selected strategy TOPSIS rank", _rank_metric(local["topsis_rank"], local["alternatives"]))
    wsm_score.metric("Selected strategy WSM score", _score_metric(local["wsm_score"]))
    topsis_score.metric("Selected strategy TOPSIS score", _score_metric(local["topsis_score"]))
    st.caption(
        "This ranking is derived from the 30-KPI SustainSCM evaluation of the "
        "structured reference anchor for the selected archetype. It is "
        "reference-archetype evidence rather than a direct ranking of the user's "
        "real company. BASE is the benchmark and is excluded from the six "
        "competitive strategies."
    )
    st.info(build_strategy_comparison_interpretation(selected))


def _render_dimension_panel(view: DecisionEvidenceView) -> None:
    selected = view.selected
    leader_strategy = (
        selected["reference_leader"]["strategy"]
        or selected["reference_leader"]["wsm_strategy"]
    )
    st.markdown("### Reference Sustainability Profile")
    st.plotly_chart(
        build_reference_profile_figure(
            archetype=selected["archetype"],
            selected_strategy=selected["strategy"],
            leader_strategy=leader_strategy,
            evidence=view.evidence,
        ),
        width="stretch",
        key="asca_mcda_reference_profile",
    )
    st.caption(
        f"The selected intervention's highest absolute reference score is in the "
        f"{selected['strongest_dimension']} dimension. Scores are compatible normalized "
        "indices from the selected archetype anchor; no physical units are mixed."
    )


def _render_cross_archetype_panel(view: DecisionEvidenceView) -> None:
    selected = view.selected
    cross = selected["cross_archetype"]
    n = int(cross["n_archetypes"])
    st.markdown("### Cross-Archetype Strategy Robustness")
    st.caption(f"Stored rank statistics across the structured reference archetypes (n = {n}).")
    if cross["wsm_mean_rank"] is None:
        st.info(summarize_cross_archetype_robustness(selected))
    else:
        wsm_tab, topsis_tab = st.tabs(["WSM", "TOPSIS"])
        for tab, prefix in ((wsm_tab, "wsm"), (topsis_tab, "topsis")):
            with tab:
                metrics = st.columns(5)
                metrics[0].metric("Rank-1 frequency", f"{int(cross[f'{prefix}_rank1_count'])}/{n}")
                metrics[1].metric("Top-2 frequency", f"{int(cross[f'{prefix}_top2_count'])}/{n}")
                metrics[2].metric("Mean rank", f"{float(cross[f'{prefix}_mean_rank']):.3g}")
                metrics[3].metric("Median rank", f"{float(cross[f'{prefix}_median_rank']):.3g}")
                metrics[4].metric("Worst rank", f"#{int(cross[f'{prefix}_worst_rank'])}")
        st.write(summarize_cross_archetype_robustness(selected))

    method = st.radio(
        "Rank method",
        ["WSM", "TOPSIS"],
        horizontal=True,
        key="asca_mcda_rank_method",
    )
    st.plotly_chart(
        build_mean_rank_figure(
            method=method,
            selected_strategy=selected["strategy"],
            evidence=view.evidence,
        ),
        width="stretch",
        key="asca_mcda_mean_rank",
    )
    st.caption("Lower rank indicates stronger performance. BASE is not a competitive alternative.")


def _render_weight_panel(view: DecisionEvidenceView) -> None:
    st.markdown("### Preference-Weight Robustness")
    order = {profile: index for index, profile in enumerate(PROFILE_LABELS)}
    records = sorted(view.weight["profiles"], key=lambda row: order.get(row["profile"], 99))
    frame = pd.DataFrame(
        [
            {
                "Preference profile": PROFILE_LABELS.get(row["profile"], row["profile"]),
                "WSM winner": row["wsm_winner"],
                "WSM leader retained": "Yes" if row["wsm_retained"] else "No",
                "TOPSIS winner": row["topsis_winner"],
                "TOPSIS leader retained": "Yes" if row["topsis_retained"] else "No",
            }
            for row in records
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    st.write(summarize_weight_robustness(view.weight))
    wsm, topsis = st.columns(2)
    wsm_data = view.weight["bounded_random"]["WSM"]
    topsis_data = view.weight["bounded_random"]["TOPSIS"]
    wsm.metric(
        "Bounded random-weight WSM winner frequency",
        f"{float(wsm_data['winner_frequency_pct']):.1f}%",
        help=f"Stored winner: {wsm_data['winner']}",
    )
    topsis.metric(
        "Bounded random-weight TOPSIS winner frequency",
        f"{float(topsis_data['winner_frequency_pct']):.1f}%",
        help=f"Stored winner: {topsis_data['winner']}",
    )
    st.caption(
        "These percentages measure MCDA weight-stability within the tested bounded "
        "preference domain; they are not AI confidence scores."
    )


def _render_completion_panel(view: DecisionEvidenceView) -> None:
    st.markdown("### Completion-Assumption Robustness")
    order = {mode: index for index, mode in enumerate(COMPLETION_LABELS)}
    records = sorted(view.completion["modes"], key=lambda row: order.get(row["mode"], 99))
    frame = pd.DataFrame(
        [
            {
                "Completion test": COMPLETION_LABELS.get(row["mode"], row["mode"]),
                "WSM winner": row["wsm_winner"],
                "WSM margin over second": row["wsm_margin"],
                "TOPSIS winner": row["topsis_winner"],
                "TOPSIS margin over second": row["topsis_margin"],
                "Full-bridge leaders retained": (
                    "Yes" if row["wsm_retained"] and row["topsis_retained"] else "No"
                ),
            }
            for row in records
        ]
    )
    st.dataframe(
        frame.style.format(
            {"WSM margin over second": "{:.3f}", "TOPSIS margin over second": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.write(summarize_completion_robustness(view.completion))
    st.caption(
        "This sensitivity check tests whether the ranking is driven primarily by the "
        "synthetic social/technological completion bridges. It does not convert "
        "synthetic KPI values into empirical observations."
    )


def _render_agreement_panel(view: DecisionEvidenceView) -> None:
    st.markdown("### Diagnostic–Decision Agreement")
    agreement = view.selected["diagnostic_agreement"]
    vsm, wsm, topsis, level = st.columns(4)
    vsm.metric("VSM-C diagnostic priority", agreement["vsm_priority"])
    wsm.metric("WSM reference leader", agreement["wsm_winner"])
    topsis.metric("TOPSIS reference leader", agreement["topsis_winner"])
    level.metric("Agreement", agreement["agreement"])
    st.write(summarize_diagnostic_agreement(agreement))


def render_decision_evidence(view: DecisionEvidenceView) -> None:
    """Render the six scientific evidence panels and their explicit scope."""

    st.markdown("## SustainSCM Decision Evidence")
    st.caption(
        "ASCA retrieves and contextualizes already-computed SustainSCM evidence; "
        "it does not calculate WSM/TOPSIS or create a new scientific result."
    )
    st.markdown("### Evidence scope")
    st.metric("Evidence scope", f"{view.scope['level']} · {view.scope['label']}")
    if view.scope["level"] == "LEVEL C":
        st.warning(view.scope["explanation"])
    else:
        st.info(view.scope["explanation"])

    _render_local_panel(view)
    _render_dimension_panel(view)
    _render_cross_archetype_panel(view)
    _render_weight_panel(view)
    _render_completion_panel(view)
    _render_agreement_panel(view)

    st.markdown("### Decision-evidence status")
    status_message = (
        f"{view.status} · Evidence classification: {view.classification}. This status "
        "describes the available decision evidence; it is not the probability that a "
        "strategy is correct."
    )
    if view.status == "PARENT MODEL REQUIRED":
        st.warning(status_message)
    elif view.status == "STRONG REFERENCE EVIDENCE":
        st.success(status_message)
    else:
        st.info(status_message)

    with st.expander("What does this evidence mean?", expanded=False):
        st.write(
            "The reference MCDA evidence was generated from the 56 structured Romanian "
            "archetype–strategy anchors. Each anchor was completed to the same 30 "
            "SustainSCM KPI architecture and evaluated through the four sustainability "
            "dimensions, composite indices, WSM and TOPSIS. BASE was used as the "
            "within-archetype benchmark and was excluded from competitive MCDA. The "
            "evidence is intended for structured reference comparison, not as a "
            "statistical representation of all Romanian companies."
        )
        st.write(
            "Some social and technological quantities required explicit synthetic "
            "completion bridges because they are not native outputs of MILP, DES or SD. "
            "Sensitivity tests were therefore performed to verify whether the strategy "
            "ranking depends on those bridge assumptions."
        )
        st.markdown("**Evidence provenance**")
        st.write(
            "56 structured anchors · 30 KPIs · 8 reference archetypes · 6 competitive "
            "strategies per archetype · neutral dimension weights · WSM + TOPSIS. "
            "Robustness covers alternative dimension weights, bounded random weights "
            "and completion-assumption sensitivity."
        )


def render_technical_mcda_sources(view: DecisionEvidenceView) -> None:
    """Place source rows in a late technical expander rather than the main narrative."""

    archetype = view.selected["archetype"]
    with st.expander("Technical MCDA evidence / source tables", expanded=False):
        st.caption(
            f"Read-only CSV rows for {archetype}. Values below are loaded from "
            f"{view.evidence.evidence_dir}."
        )
        ranking_tab, robustness_tab, sensitivity_tab = st.tabs(
            ["Local ranking", "Cross-archetype robustness", "Sensitivity"]
        )
        with ranking_tab:
            st.dataframe(
                view.ranking[
                    ["strategy", "WSM", "WSM_rank", "TOPSIS", "TOPSIS_rank", *(
                        "Environmental", "Economic", "Social", "Technological"
                    )]
                ],
                width="stretch",
                hide_index=True,
            )
        with robustness_tab:
            st.dataframe(view.evidence.strategy_robustness, width="stretch", hide_index=True)
        with sensitivity_tab:
            st.markdown("**Deterministic weight profiles**")
            st.dataframe(
                view.evidence.weight_profiles.loc[
                    view.evidence.weight_profiles["archetype"].eq(archetype)
                ],
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Bounded random weights**")
            st.dataframe(
                view.evidence.random_weights.loc[
                    view.evidence.random_weights["archetype"].eq(archetype)
                ],
                width="stretch",
                hide_index=True,
            )
            st.markdown("**Completion sensitivity**")
            st.dataframe(
                view.evidence.completion.loc[
                    view.evidence.completion["archetype"].eq(archetype)
                ],
                width="stretch",
                hide_index=True,
            )
