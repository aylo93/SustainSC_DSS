from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from asca import ASCAEngine
from asca.mcda_evidence import (
    COMPETITIVE_STRATEGIES,
    EXPECTED_ARCHETYPES,
    clear_mcda_evidence_cache,
    get_archetype_leader,
    get_archetype_ranking,
    get_archetype_strategy_evidence,
    get_completion_robustness,
    get_strategy_robustness,
    get_weight_robustness,
    load_mcda_evidence,
)
from asca.mcda_panel import is_reference_anchor_configuration, prepare_decision_evidence
from asca.mcda_visuals import build_mean_rank_figure, build_reference_profile_figure
from asca.robustness_interpreter import (
    build_evidence_scope,
    build_recommended_next_action,
    build_strategy_comparison_interpretation,
    classify_strategy_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "data" / "asca" / "mcda56"
ASSETS = ROOT / "asca_assets"
CABLE_DESCRIPTION = (
    "Medium-sized cable-assembly manufacturer with limited digital maturity, "
    "traceability gaps and stable demand variability."
)


@pytest.fixture(scope="module")
def evidence():
    return load_mcda_evidence(EVIDENCE_DIR)


@pytest.fixture(scope="module")
def cable_evaluation():
    engine = ASCAEngine(ASSETS)
    suggestion = engine.suggest(CABLE_DESCRIPTION)
    return engine.evaluate(
        archetype=suggestion.archetype,
        size_class=suggestion.size_class,
        strategy=suggestion.strategy,
        lambda_intensity=suggestion.lambda_intensity,
        parameters=suggestion.parameters,
        suggestion=suggestion,
    )


def test_a_evidence_loader_validates_complete_real_schema(evidence) -> None:
    assert evidence.archetypes == EXPECTED_ARCHETYPES
    assert set(evidence.strategies) == set(COMPETITIVE_STRATEGIES)
    assert len(evidence.neutral) == 48
    assert len(evidence.dimensions) == 56
    assert not evidence.neutral.duplicated(["archetype", "strategy"]).any()
    assert evidence.summary["n_anchors"] == 56
    assert evidence.summary["n_kpi_observations"] == 1680
    assert evidence.summary["n_nonbase_alternatives"] == 48


def test_b_ro_a1_has_six_alternatives_and_one_leader_per_method(evidence) -> None:
    ranking = get_archetype_ranking("RO-A1", evidence)
    assert set(ranking["strategy"]) == set(COMPETITIVE_STRATEGIES)
    assert ranking["WSM_rank"].eq(1).sum() == 1
    assert ranking["TOPSIS_rank"].eq(1).sum() == 1
    leader = get_archetype_leader("RO-A1", evidence)
    assert leader["wsm_strategy"] == ranking.loc[ranking["WSM_rank"].eq(1), "strategy"].iloc[0]
    assert leader["topsis_strategy"] == ranking.loc[ranking["TOPSIS_rank"].eq(1), "strategy"].iloc[0]


def test_c_ro_a2_digital_remains_selected_and_loads_cross_evidence(evidence) -> None:
    result = get_archetype_strategy_evidence("RO-A2", "DIGITAL", evidence)
    source = evidence.neutral.loc[
        evidence.neutral["archetype"].eq("RO-A2")
        & evidence.neutral["strategy"].eq("DIGITAL")
    ].iloc[0]
    assert result["strategy"] == "DIGITAL"
    assert result["reference_leader"]["wsm_strategy"] != ""
    assert result["local_mcda"]["wsm_rank"] == int(source["WSM_rank"])
    assert result["local_mcda"]["topsis_rank"] == int(source["TOPSIS_rank"])
    assert result["cross_archetype"]["wsm_mean_rank"] == pytest.approx(
        get_strategy_robustness("DIGITAL", evidence)["WSM"]["mean_rank"]
    )


def test_d_selected_leader_wording_reports_balanced_reference_support(evidence) -> None:
    leader = get_archetype_leader("RO-A1", evidence)["wsm_strategy"]
    result = get_archetype_strategy_evidence("RO-A1", leader, evidence)
    text = build_strategy_comparison_interpretation(result)
    assert result["strategy"] == leader
    assert "balanced reference leader" in text
    assert classify_strategy_evidence(result) in {
        "ROBUST_BALANCED_REFERENCE",
        "STRONG_SPECIALISED",
        "CONTEXT_DEPENDENT",
        "LIMITED_REFERENCE_SUPPORT",
    }


def test_e_nonleader_wording_never_overwrites_selected_strategy(evidence) -> None:
    leader = get_archetype_leader("RO-A2", evidence)["wsm_strategy"]
    strategy = next(value for value in COMPETITIVE_STRATEGIES if value != leader)
    result = get_archetype_strategy_evidence("RO-A2", strategy, evidence)
    text = build_strategy_comparison_interpretation(result)
    assert result["strategy"] == strategy
    assert f"selected {strategy}" in text
    assert leader in text
    assert "retained" in text


def test_f_weight_robustness_exactly_follows_source_rows(evidence) -> None:
    archetype = "RO-A3"
    leader = get_archetype_leader(archetype, evidence)["wsm_strategy"]
    result = get_weight_robustness(archetype, leader, evidence)
    source = evidence.weight_profiles.loc[evidence.weight_profiles["archetype"].eq(archetype)]
    for record in result["profiles"]:
        group = source.loc[source["profile"].eq(record["profile"])]
        assert record["wsm_winner"] == group.loc[group["WSM_rank"].eq(1), "strategy"].iloc[0]
        assert record["topsis_winner"] == group.loc[group["TOPSIS_rank"].eq(1), "strategy"].iloc[0]
    random_source = evidence.random_weights.loc[
        evidence.random_weights["archetype"].eq(archetype)
        & evidence.random_weights["strategy"].eq(result["bounded_random"]["WSM"]["winner"])
    ].iloc[0]
    assert result["bounded_random"]["WSM"]["winner_frequency_pct"] == pytest.approx(
        random_source["WSM_win_pct"]
    )


def test_g_completion_robustness_exactly_follows_source_rows(evidence) -> None:
    archetype = "RO-A4"
    result = get_completion_robustness(archetype, evidence=evidence)
    source = evidence.completion.loc[evidence.completion["archetype"].eq(archetype)]
    for record in result["modes"]:
        group = source.loc[source["mode"].eq(record["mode"])]
        wsm = group.sort_values("WSM", ascending=False)
        topsis = group.sort_values("TOPSIS", ascending=False)
        assert record["wsm_winner"] == wsm.iloc[0]["strategy"]
        assert record["wsm_margin"] == pytest.approx(wsm.iloc[0]["WSM"] - wsm.iloc[1]["WSM"])
        assert record["topsis_winner"] == topsis.iloc[0]["strategy"]
        assert record["topsis_margin"] == pytest.approx(
            topsis.iloc[0]["TOPSIS"] - topsis.iloc[1]["TOPSIS"]
        )


def test_h_outside_domain_keeps_mcda_reference_only(cable_evaluation, evidence) -> None:
    engine = ASCAEngine(ASSETS)
    parameters = dict(cable_evaluation.suggestion.parameters)
    parameters["oee"] = 0.90
    outside = engine.evaluate(
        archetype="RO-A2",
        size_class=cable_evaluation.model_row["size_class"],
        strategy="DIGITAL",
        lambda_intensity=cable_evaluation.model_row["lambda_intensity"],
        parameters=parameters,
        suggestion=cable_evaluation.suggestion,
    )
    view = prepare_decision_evidence(outside, assets_dir=ASSETS, evidence=evidence)
    assert outside.domain.status == "OUTSIDE_FINITE_TRAINING_ENVELOPE"
    assert outside.predictions["prediction"].isna().all()
    assert view.selected["strategy"] == "DIGITAL"
    assert view.scope["level"] == "LEVEL C"
    assert "does not validate surrogate prediction" in view.scope["explanation"]
    assert view.status == "PARENT MODEL REQUIRED"


def test_i_full_model_required_values_stay_blank(cable_evaluation, evidence) -> None:
    blocked_before = cable_evaluation.predictions.loc[
        cable_evaluation.predictions["validation_status"].eq("FULL_MODEL_REQUIRED"),
        "prediction",
    ].copy()
    view = prepare_decision_evidence(cable_evaluation, assets_dir=ASSETS, evidence=evidence)
    blocked_after = cable_evaluation.predictions.loc[blocked_before.index, "prediction"]
    assert blocked_before.isna().all()
    assert blocked_after.isna().all()
    assert "does not fill a withheld prediction" in view.recommended_action


def test_j_rank_update_resilience_uses_file_not_strategy_name(evidence, tmp_path) -> None:
    fixture = tmp_path / "mcda56"
    shutil.copytree(EVIDENCE_DIR, fixture)
    neutral_path = fixture / "mcda_neutral_48.csv"
    neutral = pd.read_csv(neutral_path)
    archetype = "RO-A1"
    current = neutral.loc[
        neutral["archetype"].eq(archetype) & neutral["WSM_rank"].eq(1), "strategy"
    ].iloc[0]
    replacement = next(value for value in COMPETITIVE_STRATEGIES if value != current)
    mask_current = neutral["archetype"].eq(archetype) & neutral["strategy"].eq(current)
    mask_replacement = neutral["archetype"].eq(archetype) & neutral["strategy"].eq(replacement)
    old_rank = neutral.loc[mask_replacement, "WSM_rank"].iloc[0]
    neutral.loc[mask_current, "WSM_rank"] = old_rank
    neutral.loc[mask_replacement, "WSM_rank"] = 1
    neutral.to_csv(neutral_path, index=False)
    clear_mcda_evidence_cache()
    updated = load_mcda_evidence(fixture)
    assert get_archetype_leader(archetype, updated)["wsm_strategy"] == replacement
    assert replacement != current


def test_reference_anchor_scope_is_derived_from_anchor_inputs(evidence) -> None:
    engine = ASCAEngine(ASSETS)
    anchor = pd.read_csv(ASSETS / "01_design" / "development_136.csv").query(
        "stage == 'ANCHOR' and archetype == 'RO-A1' and strategy == 'INTEGRATED'"
    ).iloc[0]
    parameters = {name: float(anchor[name]) for name in (
        "demand_load", "demand_cv", "oee", "distance_mult", "resource_mult",
        "renewable_share", "zC", "zD", "zS",
    )}
    evaluation = engine.evaluate(
        archetype="RO-A1",
        size_class=str(anchor["size_class"]),
        strategy=str(anchor["strategy"]),
        lambda_intensity=float(anchor["lambda_intensity"]),
        parameters=parameters,
    )
    assert is_reference_anchor_configuration(evaluation, ASSETS)
    assert build_evidence_scope(evaluation.domain.status, True)["level"] == "LEVEL A"


def test_scientific_wording_has_no_prohibited_claims() -> None:
    paths = [
        ROOT / "asca" / "mcda_evidence.py",
        ROOT / "asca" / "robustness_interpreter.py",
        ROOT / "asca" / "mcda_panel.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    for phrase in (
        "ai recommends",
        "ai proves",
        "ai predicts sustainability",
        "universally optimal",
        "best strategy for romanian companies",
        "confidence 100%",
        "typical romanian company",
    ):
        assert phrase not in text


def test_base_is_benchmark_not_seventh_competitor(evidence) -> None:
    result = get_archetype_strategy_evidence("RO-A1", "BASE", evidence)
    assert result["strategy"] == "BASE"
    assert result["local_mcda"]["competitive"] is False
    assert result["local_mcda"]["wsm_rank"] is None
    assert "benchmark" in build_strategy_comparison_interpretation(result)


def test_decision_figures_use_normalized_scores_and_all_six_ranks(evidence) -> None:
    leader = get_archetype_leader("RO-A2", evidence)["wsm_strategy"]
    profile = build_reference_profile_figure(
        archetype="RO-A2",
        selected_strategy="DIGITAL",
        leader_strategy=leader,
        evidence=evidence,
    )
    assert profile.layout.title.text == "Reference Sustainability Profile"
    assert profile.layout.xaxis.title.text == "Normalized sustainability score (0–100)"
    assert tuple(profile.layout.xaxis.range) == (0, 100)
    assert sum(len(trace.x) for trace in profile.data) == 10

    rank = build_mean_rank_figure(
        method="TOPSIS", selected_strategy="DIGITAL", evidence=evidence
    )
    assert rank.layout.title.text == "Cross-Archetype Mean Strategy Rank"
    assert sum(len(trace.x) for trace in rank.data) == 6
    assert rank.layout.yaxis.autorange == "reversed"
