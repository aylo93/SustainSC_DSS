from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from asca import ASCAEngine
from asca.core import FORMAL_BOUNDS
from asca.interpretation import (
    ALLOWED_ROUTES,
    COMPARISON_EXPORT_COLUMNS,
    build_base_counterfactual,
    build_key_interpretations,
    build_milp_interpretation,
    build_recommended_action,
    build_relative_change_figure,
    compare_with_base,
    comparison_export_frame,
    interpretation_payload,
    strategy_priority_interpretation,
    summarize_configuration,
    summarize_routing,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "asca_assets"
AUTOMOTIVE_ENERGY = (
    "Medium-sized Romanian automotive component supplier with high logistics "
    "complexity, low renewable-energy use and moderate digital maturity."
)


@pytest.fixture(scope="module")
def poc1_bundle():
    engine = ASCAEngine(ASSETS)
    suggestion = engine.suggest(AUTOMOTIVE_ENERGY)
    selected = engine.evaluate(
        archetype=suggestion.archetype,
        size_class=suggestion.size_class,
        strategy=suggestion.strategy,
        lambda_intensity=suggestion.lambda_intensity,
        parameters=suggestion.parameters,
        suggestion=suggestion,
    )
    original_predictions = selected.predictions.copy(deep=True)
    base = build_base_counterfactual(engine, selected)
    comparison = compare_with_base(selected, base)
    return engine, suggestion, selected, original_predictions, base, comparison


def test_a_in_domain_interpretation_comparison_chart_and_exports(poc1_bundle) -> None:
    _, suggestion, selected, original, base, comparison = poc1_bundle
    recorded = pd.read_csv(
        ROOT / "proof_of_concept_results" / "POC1_AUTOMOTIVE_predictions.csv"
    ).sort_values("target").reset_index(drop=True)
    current = selected.predictions.sort_values("target").reset_index(drop=True)

    assert selected.domain.status == "INSIDE_VALIDATED_DOMAIN"
    assert base is not None
    pd.testing.assert_frame_equal(selected.predictions, original)
    assert current["route"].tolist() == recorded["route"].tolist()
    pd.testing.assert_series_equal(
        current["prediction"],
        recorded["prediction"],
        check_names=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(comparison) == 15
    assert set(comparison["route"]) == ALLOWED_ROUTES
    assert build_relative_change_figure(comparison) is not None
    assert list(comparison_export_frame(comparison)) == COMPARISON_EXPORT_COLUMNS
    assert "RO-A1" in summarize_configuration(selected, suggestion)
    payload = interpretation_payload(selected, suggestion, comparison)
    assert payload["scenario_id"] == selected.model_row["scenario_id"]
    assert payload["route_counts"] == {
        "validated_screening": 10,
        "exploratory": 5,
        "parent_model_required": 2,
    }


def test_b_full_model_required_values_remain_withheld(poc1_bundle) -> None:
    _, _, selected, _, _, comparison = poc1_bundle
    blocked = selected.predictions[
        selected.predictions["validation_status"].eq("FULL_MODEL_REQUIRED")
    ]

    assert len(blocked) == 2
    assert blocked["prediction"].isna().all()
    assert not set(blocked["target"]).intersection(comparison["target"])
    milp_text = build_milp_interpretation(selected)
    assert "intentionally withheld" in milp_text
    assert "parent model" in milp_text


def test_c_conditional_outputs_are_visible_but_exploratory(poc1_bundle) -> None:
    _, _, selected, _, _, comparison = poc1_bundle
    conditional = selected.predictions[
        selected.predictions["validation_status"].eq("CONDITIONAL")
    ]
    conditional_comparison = comparison[
        comparison["validation_status"].eq("CONDITIONAL")
    ]

    assert len(conditional) == 5
    assert conditional["prediction"].notna().all()
    assert set(conditional_comparison["interpretation_availability"]) == {
        "Exploratory only"
    }
    assert "exploratory estimate" in build_milp_interpretation(selected)


def test_d_outside_domain_blocks_comparison_without_clipping() -> None:
    engine = ASCAEngine(ASSETS)
    suggestion = engine.suggest(AUTOMOTIVE_ENERGY)
    parameters = dict(suggestion.parameters)
    parameters["oee"] = 0.90
    selected = engine.evaluate(
        archetype=suggestion.archetype,
        size_class=suggestion.size_class,
        strategy=suggestion.strategy,
        lambda_intensity=suggestion.lambda_intensity,
        parameters=parameters,
        suggestion=suggestion,
    )
    base = build_base_counterfactual(engine, selected)

    assert selected.model_row["oee"] == 0.90
    assert selected.domain.status == "OUTSIDE_FINITE_TRAINING_ENVELOPE"
    assert selected.predictions["prediction"].isna().all()
    assert base is None
    assert compare_with_base(selected, base).empty
    assert "Surrogate screening is blocked" in build_recommended_action(selected)


def test_e_tested_strategy_and_diagnostic_priority_remain_distinct(poc1_bundle) -> None:
    _, _, selected, _, _, comparison = poc1_bundle
    note = strategy_priority_interpretation(selected)

    assert selected.model_row["strategy"] == "ENERGY"
    assert selected.model_row["priority_strategy"] == "INTEGRATED"
    assert "Strategy being evaluated: ENERGY" in note
    assert "Diagnostic priority: INTEGRATED" in note
    assert "retained" in note
    assert "optimal" not in " ".join(build_key_interpretations(selected, comparison)).lower()


def test_f_base_companion_preserves_company_inputs_and_uses_normal_router(
    poc1_bundle,
) -> None:
    _, _, selected, original, base, comparison = poc1_bundle
    assert base is not None

    invariant_keys = [
        "archetype",
        "archetype_name",
        "size_class",
        *FORMAL_BOUNDS,
        "ref_output",
        "ref_energy_kwh_fu",
        "ref_material_kg_fu",
        "ref_cost_eur_fu",
        "ref_distance_km",
        "nominal_capacity_fu_y",
        "VSM_PCE",
        "VSM_NVAT_R",
        "VSM_OEE",
        "VSM_WIP_I",
        "VSM_DR",
        "VSM_EI",
        "VSM_GHGI",
        "VSM_WI",
        "VSM_IDR",
        "VSM_SHR",
    ]
    for key in invariant_keys:
        assert base.model_row[key] == pytest.approx(selected.model_row[key]) if isinstance(
            selected.model_row[key], float
        ) else base.model_row[key] == selected.model_row[key]

    assert selected.model_row["strategy"] == "ENERGY"
    assert selected.model_row["lambda_intensity"] == 0.50
    assert base.model_row["strategy"] == "BASE"
    assert base.model_row["lambda_intensity"] == 0.0
    assert base.model_row["scenario_id"].endswith("-BASE-CF")
    assert base.domain.status == "INSIDE_VALIDATED_DOMAIN"
    assert len(base.predictions) == 17
    pd.testing.assert_frame_equal(selected.predictions, original)
    assert not comparison["target"].isin(
        ["milp_total_co2_t", "milp_transport_work_tkm"]
    ).any()
    assert summarize_routing(base.predictions) == summarize_routing(selected.predictions)
