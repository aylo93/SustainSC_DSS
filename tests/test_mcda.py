import numpy as np
import pandas as pd
import pytest

from sustainsc.mcda import (
    DIMENSION_ORDER,
    build_mcda_input,
    calculate_mcda,
    canonical_dimension,
    compute_complete_dimension_indices,
    evaluate_scenario_eligibility,
    expected_kpis_from_metadata,
)


EXPECTED = {
    "environmental": [f"E{i}" for i in range(1, 10)],
    "economic": [f"EC{i}" for i in range(1, 9)],
    "social": [f"S{i}" for i in range(1, 7)],
    "technological": [f"T{i}" for i in range(1, 8)],
}


def metadata():
    return pd.DataFrame(
        [
            {"kpi_code": code, "dimension": dimension}
            for dimension, codes in EXPECTED.items()
            for code in codes
        ]
    )


def result_rows(scenarios=("A", "B")):
    rows = []
    for scenario_number, scenario in enumerate(scenarios):
        for criterion_number, code in enumerate(metadata()["kpi_code"]):
            rows.append(
                {
                    "scenario_code": scenario,
                    "kpi_code": code,
                    "raw_value": float(criterion_number + scenario_number + 1),
                    "normalized_value": float(criterion_number + scenario_number + 10),
                }
            )
    return pd.DataFrame(rows)


def test_canonical_kpi_architecture_and_dimension_aliases():
    expected = expected_kpis_from_metadata(metadata())
    assert tuple(expected) == DIMENSION_ORDER
    assert [len(expected[d]) for d in DIMENSION_ORDER] == [9, 8, 6, 7]
    assert canonical_dimension("ENV") == "environmental"
    assert canonical_dimension("Technology") == "technological"
    with pytest.raises(ValueError, match="Unrecognized"):
        canonical_dimension("planet")


def test_completeness_detects_missing_and_duplicate_kpis():
    rows = result_rows(("A",))
    raw = rows[["scenario_code", "kpi_code", "raw_value"]]
    norm = rows[["scenario_code", "kpi_code", "normalized_value"]]
    ready = evaluate_scenario_eligibility(raw, norm, metadata()).iloc[0]
    assert ready["raw_kpi_count"] == ready["normalized_kpi_count"] == 30
    assert ready["mcda_eligible"]

    incomplete_norm = pd.concat(
        [norm[norm["kpi_code"] != "E1"], norm[norm["kpi_code"] == "S1"]]
    )
    incomplete = evaluate_scenario_eligibility(raw, incomplete_norm, metadata()).iloc[0]
    assert not incomplete["mcda_eligible"]
    assert incomplete["missing_normalized_kpis"] == "E1"
    assert incomplete["duplicate_kpis"] == "S1"


def test_incomplete_dimension_has_no_integrated_dimension_score():
    rows = result_rows(("A",))
    norm = rows[["scenario_code", "kpi_code", "normalized_value"]]
    weights = pd.Series(1.0, index=metadata()["kpi_code"])
    complete, incomplete = compute_complete_dimension_indices(norm, metadata(), weights)
    assert len(complete) == 4
    assert incomplete.empty

    complete, incomplete = compute_complete_dimension_indices(
        norm[norm["kpi_code"] != "E9"], metadata(), weights
    )
    assert set(complete["dimension"]) == {"economic", "social", "technological"}
    assert incomplete.iloc[0]["dimension"] == "environmental"
    assert incomplete.iloc[0]["missing_kpis"] == "E9"


def test_wsm_and_topsis_share_population_and_are_finite():
    rows = result_rows()
    raw = rows[["scenario_code", "kpi_code", "raw_value"]]
    norm = rows[["scenario_code", "kpi_code", "normalized_value"]]
    eligibility = evaluate_scenario_eligibility(raw, norm, metadata())
    weights = pd.Series(1 / 30, index=metadata()["kpi_code"])
    mcda_input = build_mcda_input(norm, weights, eligibility)
    result = calculate_mcda(mcda_input, eligibility)
    assert set(result.wsm["scenario_code"]) == set(result.topsis["scenario_code"]) == {"A", "B"}
    assert np.isfinite(result.wsm["WSM_score"]).all()
    assert np.isfinite(result.topsis["TOPSIS_score"]).all()
    assert result.topsis["TOPSIS_score"].between(0, 100).all()


def test_weight_alignment_and_validation_are_label_based():
    rows = result_rows()
    raw = rows[["scenario_code", "kpi_code", "raw_value"]]
    norm = rows[["scenario_code", "kpi_code", "normalized_value"]]
    eligibility = evaluate_scenario_eligibility(raw, norm, metadata())
    reversed_codes = list(reversed(metadata()["kpi_code"].tolist()))
    weights = pd.Series(1 / 30, index=reversed_codes)
    mcda_input = build_mcda_input(norm, weights, eligibility)
    assert list(mcda_input.matrix.columns) == reversed_codes
    assert np.isclose(mcda_input.weights.sum(), 1)

    bad_weights = weights.copy()
    bad_weights.iloc[0] = np.nan
    with pytest.raises(ValueError, match="weights"):
        build_mcda_input(norm, bad_weights, eligibility)


def test_zero_variance_criteria_are_explicitly_removed_and_ties_preserved():
    rows = result_rows(("A", "B", "C"))
    rows.loc[rows["kpi_code"] == "T7", "normalized_value"] = 50.0
    raw = rows[["scenario_code", "kpi_code", "raw_value"]]
    norm = rows[["scenario_code", "kpi_code", "normalized_value"]]
    eligibility = evaluate_scenario_eligibility(raw, norm, metadata())
    weights = pd.Series(1 / 30, index=metadata()["kpi_code"])
    result = calculate_mcda(build_mcda_input(norm, weights, eligibility), eligibility)
    assert "T7" in result.diagnostics["zero_variance_criteria"]
    assert len(result.topsis) == 3

    tied = norm.copy()
    tied.loc[tied["scenario_code"] == "B", "normalized_value"] = tied.loc[
        tied["scenario_code"] == "A", "normalized_value"
    ].to_numpy()
    tied_eligibility = evaluate_scenario_eligibility(raw, tied, metadata())
    tied_result = calculate_mcda(
        build_mcda_input(tied, weights, tied_eligibility), tied_eligibility
    )
    scores = tied_result.topsis.set_index("scenario_code")["TOPSIS_score"]
    assert scores["A"] == pytest.approx(scores["B"])
