"""Deterministic interpretation of the read-only SustainSCM MCDA56 evidence.

The interpreter converts structured scientific evidence into deterministic
user-facing descriptions. It does not generate independent sustainability
evidence and does not call an external language model.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


EVIDENCE_CATEGORIES = {
    "ROBUST_BALANCED_REFERENCE",
    "STRONG_SPECIALISED",
    "CONTEXT_DEPENDENT",
    "LIMITED_REFERENCE_SUPPORT",
}

PROFILE_LABELS = {
    "NEUTRAL": "Neutral",
    "ENV_40": "Environmental emphasis",
    "ECON_40": "Economic emphasis",
    "SOCIAL_40": "Social emphasis",
    "TECH_40": "Technological emphasis",
}

COMPLETION_LABELS = {
    "FULL_BRIDGES": "Full bridge effect",
    "HALF_BRIDGE_EFFECT": "Reduced / 50% bridge effect",
    "NO_BRIDGE_CREDIT": "Zero bridge-credit",
}


def classify_strategy_evidence(evidence: Mapping[str, Any]) -> str:
    """Classify support using ranks and sensitivity evidence, never strategy names."""

    local = evidence["local_mcda"]
    cross = evidence["cross_archetype"]
    if not local.get("competitive"):
        return "LIMITED_REFERENCE_SUPPORT"

    n = int(cross.get("n_archetypes") or 0)
    robust_threshold = math.ceil(0.75 * n) if n else 1
    locally_first = local.get("wsm_rank") == 1 and local.get("topsis_rank") == 1
    broadly_first = (
        (cross.get("wsm_rank1_count") or 0) >= robust_threshold
        and (cross.get("topsis_rank1_count") or 0) >= robust_threshold
    )
    sensitivity_stable = bool(
        evidence["robustness"].get("weight_stable")
        and evidence["robustness"].get("completion_stable")
    )
    if locally_first and broadly_first and sensitivity_stable:
        return "ROBUST_BALANCED_REFERENCE"

    ranks = [rank for rank in (local.get("wsm_rank"), local.get("topsis_rank")) if rank]
    best_local_rank = min(ranks) if ranks else 99
    frequent_top_two = max(
        int(cross.get("wsm_top2_count") or 0),
        int(cross.get("topsis_top2_count") or 0),
    ) >= max(1, math.ceil(n / 2))
    if best_local_rank <= 3 and frequent_top_two:
        return "STRONG_SPECIALISED"
    if best_local_rank <= 4 or max(
        int(cross.get("wsm_top2_count") or 0),
        int(cross.get("topsis_top2_count") or 0),
    ) > 0:
        return "CONTEXT_DEPENDENT"
    return "LIMITED_REFERENCE_SUPPORT"


def summarize_cross_archetype_robustness(evidence: Mapping[str, Any]) -> str:
    """Describe local strategy stability while keeping WSM and TOPSIS distinct."""

    strategy = str(evidence["strategy"])
    cross = evidence["cross_archetype"]
    n = int(cross["n_archetypes"])
    if cross.get("wsm_mean_rank") is None:
        return (
            f"{strategy} is the within-archetype benchmark and is not one of the six "
            "competitive strategies in the cross-archetype robustness table."
        )
    return (
        f"Across the {n} structured Romanian reference archetypes, {strategy} was "
        f"within the top two WSM alternatives in {int(cross['wsm_top2_count'])}/{n} "
        f"cases and ranked first in {int(cross['wsm_rank1_count'])}/{n}. Under TOPSIS, "
        f"it was within the top two in {int(cross['topsis_top2_count'])}/{n} cases and "
        f"ranked first in {int(cross['topsis_rank1_count'])}/{n}; its mean ranks were "
        f"{float(cross['wsm_mean_rank']):.3g} (WSM) and "
        f"{float(cross['topsis_mean_rank']):.3g} (TOPSIS)."
    )


def summarize_weight_robustness(weight: Mapping[str, Any]) -> str:
    """Describe deterministic and random-weight evidence within the tested domain."""

    profiles = weight["profiles"]
    neutral = weight["neutral"]
    deterministic = (
        f"The neutral WSM and TOPSIS leaders remain unchanged across all "
        f"{len(profiles)} tested deterministic dimension-weight profiles."
        if weight["deterministic_stable"]
        else "At least one deterministic dimension-weight profile changes a neutral leader."
    )
    wsm = weight["bounded_random"]["WSM"]
    topsis = weight["bounded_random"]["TOPSIS"]
    bounded = (
        f"Within the tested bounded random-weight domain, {wsm['winner']} has a "
        f"{float(wsm['winner_frequency_pct']):.1f}% WSM winner frequency and "
        f"{topsis['winner']} has a {float(topsis['winner_frequency_pct']):.1f}% "
        "TOPSIS winner frequency."
    )
    if neutral["wsm_winner"] != neutral["topsis_winner"]:
        deterministic += (
            f" The neutral methods differ: WSM identifies {neutral['wsm_winner']} and "
            f"TOPSIS identifies {neutral['topsis_winner']}."
        )
    return deterministic + " " + bounded


def summarize_completion_robustness(completion: Mapping[str, Any]) -> str:
    """Describe whether stored winners persist across completion assumptions."""

    modes = completion["modes"]
    if completion["stable"]:
        full = next(mode for mode in modes if mode["mode"] == "FULL_BRIDGES")
        return (
            f"The full-bridge leaders ({full['wsm_winner']} under WSM and "
            f"{full['topsis_winner']} under TOPSIS) are retained under all "
            f"{len(modes)} tested completion assumptions."
        )
    changed = [
        COMPLETION_LABELS.get(mode["mode"], mode["mode"])
        for mode in modes
        if not (mode["wsm_retained"] and mode["topsis_retained"])
    ]
    return "A completion-assumption leader change occurs under: " + ", ".join(changed) + "."


def summarize_diagnostic_agreement(agreement: Mapping[str, Any]) -> str:
    """Explain diagnostic/decision convergence without implying causal validity."""

    level = agreement["agreement"]
    if level == "YES":
        return (
            "The VSM-C diagnostic priority and both MCDA methods converge for this "
            "reference archetype. This is mutually consistent reference evidence; "
            "it does not by itself establish causal validity."
        )
    if level == "PARTIAL":
        return (
            "Two of the three reference signals agree, while one differs. Treat the "
            "result as partial diagnostic-decision convergence and inspect the trade-offs."
        )
    return (
        "The VSM-C diagnostic priority, WSM leader and TOPSIS leader differ. The "
        "reference signals should be interpreted separately rather than collapsed."
    )


def build_evidence_scope(domain_status: str, is_reference_anchor: bool) -> dict[str, str]:
    """Return one of the three explicitly bounded evidence-scope levels."""

    if domain_status != "INSIDE_VALIDATED_DOMAIN":
        return {
            "level": "LEVEL C",
            "label": "Reference evidence only — parent-model confirmation required",
            "explanation": (
                "The selected company configuration lies outside the validated surrogate "
                "domain. The archetype-level MCDA evidence remains useful as reference "
                "evidence, but it does not validate surrogate prediction for the current "
                "configuration."
            ),
        }
    if is_reference_anchor:
        return {
            "level": "LEVEL A",
            "label": "Reference-anchor evidence",
            "explanation": (
                "The current configuration matches the structured reference anchor for "
                "this archetype and strategy."
            ),
        }
    return {
        "level": "LEVEL B",
        "label": "In-domain metamodel screening + reference-archetype MCDA evidence",
        "explanation": (
            "The MCDA ranking comes from the archetype reference anchor; the current "
            "numerical screening comes from the in-domain metamodel configuration."
        ),
    }


def build_decision_evidence_status(
    *,
    domain_status: str,
    parent_model_required_count: int,
    selected_evidence: Mapping[str, Any],
) -> str:
    """Combine transparent evidence components; this is not a probability."""

    if domain_status != "INSIDE_VALIDATED_DOMAIN" or parent_model_required_count > 0:
        return "PARENT MODEL REQUIRED"
    category = classify_strategy_evidence(selected_evidence)
    if category == "ROBUST_BALANCED_REFERENCE":
        return "STRONG REFERENCE EVIDENCE"
    if category in {"STRONG_SPECIALISED", "CONTEXT_DEPENDENT"}:
        return "MODERATE REFERENCE EVIDENCE"
    return "EXPLORATORY"


def build_strategy_comparison_interpretation(evidence: Mapping[str, Any]) -> str:
    """Keep the tested strategy separate from the stored reference leader(s)."""

    selected = str(evidence["strategy"])
    leader = evidence["reference_leader"]
    if not evidence["local_mcda"].get("competitive"):
        return (
            "BASE is retained as the within-archetype benchmark and is not assigned a "
            "competitive MCDA rank. The intervention leaders remain displayed separately."
        )
    if leader["methods_agree"] and selected == leader["strategy"]:
        return (
            f"The {selected} intervention being evaluated is consistent with the balanced "
            "reference leader identified independently by WSM and TOPSIS."
        )
    if leader["methods_agree"]:
        return (
            f"The selected {selected} intervention is retained for counterfactual screening. "
            f"The reference 30-KPI MCDA identifies {leader['strategy']} as the stronger "
            "balanced anchor strategy, so comparison with that configuration may be useful."
        )
    return (
        f"The selected {selected} intervention is retained. The reference methods do not "
        f"share one leader: WSM identifies {leader['wsm_strategy']} and TOPSIS identifies "
        f"{leader['topsis_strategy']}."
    )


def build_recommended_next_action(
    *,
    domain_status: str,
    parent_model_required_count: int,
    selected_evidence: Mapping[str, Any],
) -> str:
    """Recommend a bounded next step while giving domain and blocked routes priority."""

    if domain_status != "INSIDE_VALIDATED_DOMAIN":
        return (
            "Use the reference MCDA only as contextual evidence and execute the "
            "corresponding parent model for the current configuration, or extend the "
            "validated experimental design."
        )
    if parent_model_required_count > 0:
        return (
            "Do not base the final decision on surrogate screening alone. Run the relevant "
            "parent model for each blocked output; the MCDA reference evidence does not fill "
            "a withheld prediction."
        )
    leader = selected_evidence["reference_leader"]
    selected = selected_evidence["strategy"]
    if leader["methods_agree"] and selected == leader["strategy"]:
        return (
            "The selected intervention is consistent with the strongest balanced reference "
            "evidence. Continue screening and confirm high-impact implementation decisions "
            "with company-specific MRV and the relevant parent models."
        )
    if classify_strategy_evidence(selected_evidence) == "STRONG_SPECIALISED":
        return (
            "The selected intervention remains relevant as a targeted strategy. Compare it "
            "with the reference balanced leader before final selection."
        )
    return (
        "Treat the selected intervention as contextual or exploratory reference evidence, "
        "inspect its dimension trade-offs and compare it with the stored MCDA leader before "
        "a company-specific decision."
    )


def build_mcda_interpretation(
    *,
    selected_evidence: Mapping[str, Any],
    leader_evidence: Mapping[str, Any],
    weight: Mapping[str, Any],
    completion: Mapping[str, Any],
    domain_status: str,
    recommended_action: str,
) -> list[str]:
    """Build four concise paragraphs solely from supplied structured evidence."""

    strategy = str(selected_evidence["strategy"])
    archetype = str(selected_evidence["archetype"])
    local = selected_evidence["local_mcda"]
    strongest = selected_evidence["strongest_dimension"]
    if local.get("competitive"):
        paragraph_one = (
            f"Within the {archetype} reference anchor, {strategy} ranks "
            f"#{int(local['wsm_rank'])} of {int(local['alternatives'])} under WSM and "
            f"#{int(local['topsis_rank'])} of {int(local['alternatives'])} under TOPSIS. "
            f"Its highest absolute normalized dimension score is {strongest}."
        )
    else:
        paragraph_one = (
            f"Within the {archetype} reference anchor, BASE remains the benchmark and is "
            f"excluded from the six-strategy competitive ranking. Its highest absolute "
            f"normalized dimension score is {strongest}."
        )

    paragraph_two = summarize_cross_archetype_robustness(selected_evidence)
    if leader_evidence["strategy"] != strategy:
        paragraph_two += " " + summarize_cross_archetype_robustness(leader_evidence)

    paragraph_three = (
        summarize_weight_robustness(weight)
        + " "
        + summarize_completion_robustness(completion)
        + " These tests describe the controlled anchor experiment, not universal company performance."
    )
    paragraph_four = (
        f"For the current configuration, the metamodel domain status is {domain_status}. "
        + recommended_action
    )
    return [paragraph_one, paragraph_two, paragraph_three, paragraph_four]


def build_decision_evidence_payload(
    *,
    scenario_id: str,
    domain_status: str,
    selected_evidence: Mapping[str, Any],
    weight: Mapping[str, Any],
    completion: Mapping[str, Any],
    evidence_scope: Mapping[str, str],
    decision_status: str,
    recommended_action: str,
) -> dict[str, Any]:
    """Create a JSON-ready evidence export separate from the original ASCA trace."""

    return {
        "scenario_id": scenario_id,
        "selected_archetype": selected_evidence["archetype"],
        "tested_strategy": selected_evidence["strategy"],
        "domain_status": domain_status,
        "reference_mcda": {
            "local_selected_strategy": selected_evidence["local_mcda"],
            "reference_leader": selected_evidence["reference_leader"],
            "dimension_profile": selected_evidence["dimension_profile"],
            "strongest_dimension": selected_evidence["strongest_dimension"],
        },
        "cross_archetype_robustness": selected_evidence["cross_archetype"],
        "weight_robustness": weight,
        "completion_robustness": completion,
        "diagnostic_agreement": selected_evidence["diagnostic_agreement"],
        "evidence_classification": classify_strategy_evidence(selected_evidence),
        "evidence_scope": dict(evidence_scope),
        "decision_evidence_status": decision_status,
        "recommended_next_action": recommended_action,
    }
