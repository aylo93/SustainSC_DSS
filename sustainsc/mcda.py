"""Validated sustainability dimensions and KPI-level MCDA calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

DIMENSION_ORDER = ("environmental", "economic", "social", "technological")
MCDA_CRITERION_LEVEL = "kpi"

_DIMENSION_ALIASES = {
    "environmental": "environmental",
    "env": "environmental",
    "economic": "economic",
    "eco": "economic",
    "social": "social",
    "soc": "social",
    "technological": "technological",
    "technology": "technological",
    "tech": "technological",
}


def canonical_dimension(value: object) -> str:
    """Return one of the four production dimension labels.

    Aliases are intentionally explicit. Unknown labels fail at the metadata
    boundary instead of being guessed or propagated into calculations.
    """

    key = str(value).strip().lower()
    try:
        return _DIMENSION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unrecognized sustainability dimension: {value!r}") from exc


def expected_kpis_from_metadata(kpi_metadata: pd.DataFrame) -> dict[str, frozenset[str]]:
    required = {"kpi_code", "dimension"}
    missing_columns = required - set(kpi_metadata.columns)
    if missing_columns:
        raise ValueError(f"KPI metadata missing columns: {sorted(missing_columns)}")

    metadata = kpi_metadata.loc[:, ["kpi_code", "dimension"]].copy()
    metadata["kpi_code"] = metadata["kpi_code"].astype(str).str.strip()
    metadata["dimension"] = metadata["dimension"].map(canonical_dimension)
    if metadata["kpi_code"].duplicated().any():
        duplicates = sorted(metadata.loc[metadata["kpi_code"].duplicated(False), "kpi_code"].unique())
        raise ValueError(f"Duplicate KPI metadata codes: {duplicates}")
    return {
        dimension: frozenset(metadata.loc[metadata["dimension"] == dimension, "kpi_code"])
        for dimension in DIMENSION_ORDER
    }


def _finite_codes(frame: pd.DataFrame, value_column: str) -> set[str]:
    if frame.empty:
        return set()
    values = pd.to_numeric(frame[value_column], errors="coerce")
    return set(frame.loc[np.isfinite(values), "kpi_code"].astype(str).str.strip())


def _joined(values: Iterable[str]) -> str:
    return ", ".join(sorted(set(values)))


def evaluate_scenario_eligibility(
    raw_results: pd.DataFrame,
    normalized_results: pd.DataFrame,
    kpi_metadata: pd.DataFrame,
    scenario_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Produce an auditable completeness result for every scenario."""

    expected_by_dimension = expected_kpis_from_metadata(kpi_metadata)
    expected = frozenset().union(*expected_by_dimension.values())
    scenarios = set(str(v).strip() for v in (scenario_codes or ()))
    scenarios.update(raw_results.get("scenario_code", pd.Series(dtype=str)).dropna().astype(str).str.strip())
    scenarios.update(
        normalized_results.get("scenario_code", pd.Series(dtype=str)).dropna().astype(str).str.strip()
    )

    rows: list[dict[str, object]] = []
    for scenario in sorted(scenarios):
        raw = raw_results[raw_results["scenario_code"].astype(str).str.strip() == scenario]
        norm = normalized_results[
            normalized_results["scenario_code"].astype(str).str.strip() == scenario
        ]
        raw_codes = _finite_codes(raw, "raw_value")
        norm_codes = _finite_codes(norm, "normalized_value")
        raw_duplicates = sorted(
            raw.loc[raw.duplicated(["kpi_code"], keep=False), "kpi_code"].astype(str).unique()
        )
        norm_duplicates = sorted(
            norm.loc[norm.duplicated(["kpi_code"], keep=False), "kpi_code"].astype(str).unique()
        )
        missing_raw = sorted(expected - raw_codes)
        missing_norm = sorted(expected - norm_codes)
        missing_dimensions = [
            dimension
            for dimension, codes in expected_by_dimension.items()
            if not codes.issubset(norm_codes)
        ]
        unknown_raw = sorted(raw_codes - expected)
        unknown_norm = sorted(norm_codes - expected)
        raw_complete = not missing_raw and not raw_duplicates and not unknown_raw
        normalized_complete = not missing_norm and not norm_duplicates and not unknown_norm
        dimensions_complete = not missing_dimensions
        eligible = raw_complete and normalized_complete and dimensions_complete

        reasons = []
        if missing_raw:
            reasons.append("missing raw: " + _joined(missing_raw))
        if missing_norm:
            reasons.append("missing normalized: " + _joined(missing_norm))
        if raw_duplicates:
            reasons.append("duplicate raw: " + _joined(raw_duplicates))
        if norm_duplicates:
            reasons.append("duplicate normalized: " + _joined(norm_duplicates))
        if unknown_raw or unknown_norm:
            reasons.append("unrecognized KPI: " + _joined(unknown_raw + unknown_norm))
        if missing_dimensions:
            reasons.append("incomplete dimensions: " + _joined(missing_dimensions))

        counts = {
            dimension: len(norm_codes & codes)
            for dimension, codes in expected_by_dimension.items()
        }
        rows.append(
            {
                "scenario_code": scenario,
                "raw_kpi_count": len(raw_codes & expected),
                "normalized_kpi_count": len(norm_codes & expected),
                **{f"{dimension}_count": counts[dimension] for dimension in DIMENSION_ORDER},
                "raw_complete": raw_complete,
                "normalized_complete": normalized_complete,
                "four_dimensions_complete": dimensions_complete,
                "mcda_eligible": eligible,
                "wsm_eligible": eligible,
                "topsis_eligible": eligible,
                "missing_raw_kpis": _joined(missing_raw),
                "missing_normalized_kpis": _joined(missing_norm),
                "duplicate_kpis": _joined(raw_duplicates + norm_duplicates),
                "missing_dimensions": _joined(missing_dimensions),
                "null_raw_values": int(pd.to_numeric(raw.get("raw_value"), errors="coerce").isna().sum()),
                "null_normalized_values": int(
                    pd.to_numeric(norm.get("normalized_value"), errors="coerce").isna().sum()
                ),
                "status": "Ready" if eligible else "Incomplete",
                "reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def compute_complete_dimension_indices(
    normalized_results: pd.DataFrame,
    kpi_metadata: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate a dimension only when its complete normalized KPI set exists."""

    expected = expected_kpis_from_metadata(kpi_metadata)
    weight_series = pd.to_numeric(weights, errors="coerce")
    rows = []
    incomplete = []
    for scenario, scenario_frame in normalized_results.groupby("scenario_code", sort=True):
        for dimension in DIMENSION_ORDER:
            required = expected[dimension]
            subset = scenario_frame[scenario_frame["kpi_code"].isin(required)].copy()
            finite = pd.to_numeric(subset["normalized_value"], errors="coerce")
            present = set(subset.loc[np.isfinite(finite), "kpi_code"])
            duplicates = set(
                subset.loc[subset.duplicated(["kpi_code"], keep=False), "kpi_code"]
            )
            missing = required - present
            if missing or duplicates:
                incomplete.append(
                    {
                        "scenario_code": scenario,
                        "dimension": dimension,
                        "missing_kpis": _joined(missing),
                        "duplicate_kpis": _joined(duplicates),
                        "status": "INCOMPLETE",
                    }
                )
                continue
            values = subset.set_index("kpi_code")["normalized_value"].astype(float).reindex(sorted(required))
            local_weights = weight_series.reindex(values.index)
            if local_weights.isna().any() or not np.isfinite(local_weights).all() or local_weights.sum() <= 0:
                incomplete.append(
                    {
                        "scenario_code": scenario,
                        "dimension": dimension,
                        "missing_kpis": "",
                        "duplicate_kpis": "",
                        "status": "INVALID_WEIGHTS",
                    }
                )
                continue
            rows.append(
                {
                    "scenario_code": scenario,
                    "dimension": dimension,
                    "dimension_index": float(np.average(values, weights=local_weights)),
                    "kpis_used": len(required),
                    "dimension_status": "COMPLETE",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(incomplete)


@dataclass(frozen=True)
class MCDAInput:
    matrix: pd.DataFrame
    weights: pd.Series
    eligible_scenarios: tuple[str, ...]
    excluded_scenarios: pd.DataFrame
    criterion_level: str = MCDA_CRITERION_LEVEL
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MCDAResult:
    wsm: pd.DataFrame
    topsis: pd.DataFrame
    eligibility: pd.DataFrame
    diagnostics: dict[str, object]


def build_mcda_input(
    normalized_results: pd.DataFrame,
    weights: pd.Series,
    eligibility: pd.DataFrame,
    selected_scenarios: Iterable[str] | None = None,
    reference_scenario_code: str | None = None,
) -> MCDAInput:
    selected = tuple(dict.fromkeys(str(v).strip() for v in (selected_scenarios or ())))
    eligible_table = eligibility[eligibility["mcda_eligible"]].copy()
    reference = str(reference_scenario_code or "").strip()
    if reference:
        eligible_table = eligible_table[eligible_table["scenario_code"] != reference]
    if selected:
        eligible_table = eligible_table[eligible_table["scenario_code"].isin(selected)]
        excluded = eligibility[
            eligibility["scenario_code"].isin(selected) & ~eligibility["mcda_eligible"]
        ].copy()
    else:
        excluded = eligibility[~eligibility["mcda_eligible"]].copy()
    if reference:
        reference_rows = eligibility[eligibility["scenario_code"] == reference]
        excluded = pd.concat([excluded, reference_rows], ignore_index=True).drop_duplicates("scenario_code")
    eligible = tuple(eligible_table["scenario_code"])

    work = normalized_results[normalized_results["scenario_code"].isin(eligible)].copy()
    if work.duplicated(["scenario_code", "kpi_code"]).any():
        raise ValueError("MCDA input contains duplicate scenario/KPI pairs")
    matrix = work.pivot(index="scenario_code", columns="kpi_code", values="normalized_value")
    matrix = matrix.reindex(index=list(eligible), columns=list(weights.index)).apply(
        pd.to_numeric, errors="coerce"
    )
    aligned_weights = pd.to_numeric(weights.reindex(matrix.columns), errors="coerce")
    errors = []
    if not matrix.index.is_unique:
        errors.append("scenario index is not unique")
    if not matrix.columns.is_unique:
        errors.append("criterion columns are not unique")
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        errors.append("matrix contains NaN or infinite values")
    if aligned_weights.isna().any() or not np.isfinite(aligned_weights).all():
        errors.append("weights are missing or non-finite")
    if (aligned_weights < 0).any():
        errors.append("weights contain negative values")
    if not np.isclose(float(aligned_weights.sum()), 1.0):
        errors.append(f"weights sum to {aligned_weights.sum():.12g}, not 1")
    if errors:
        raise ValueError("; ".join(errors))

    diagnostics = {
        "matrix_shape": matrix.shape,
        "scenario_count": matrix.shape[0],
        "criterion_count": matrix.shape[1],
        "nan_count": int(matrix.isna().sum().sum()),
        "inf_count": int(np.isinf(matrix.to_numpy(dtype=float)).sum()),
        "weight_sum": float(aligned_weights.sum()),
        "excluded_scenarios": excluded["scenario_code"].tolist(),
    }
    LOGGER.info("Validated MCDA input: %s", diagnostics)
    return MCDAInput(matrix, aligned_weights, eligible, excluded, diagnostics=diagnostics)


def calculate_mcda(mcda_input: MCDAInput, eligibility: pd.DataFrame) -> MCDAResult:
    matrix = mcda_input.matrix
    weights = mcda_input.weights
    if matrix.empty:
        return MCDAResult(pd.DataFrame(), pd.DataFrame(), eligibility, mcda_input.diagnostics)

    wsm_values = matrix.to_numpy(dtype=float) @ weights.to_numpy(dtype=float)
    wsm = pd.DataFrame({"scenario_code": matrix.index, "WSM_score": wsm_values})

    zero_variance = matrix.columns[matrix.nunique(dropna=False) <= 1].tolist()
    topsis_matrix = matrix.drop(columns=zero_variance)
    if topsis_matrix.empty:
        topsis = pd.DataFrame(columns=["scenario_code", "TOPSIS_score"])
    else:
        topsis_weights = weights.reindex(topsis_matrix.columns)
        topsis_weights = topsis_weights / topsis_weights.sum()
        norms = np.sqrt((topsis_matrix**2).sum(axis=0))
        zero_norm = norms.index[np.isclose(norms, 0.0)].tolist()
        if zero_norm:
            topsis_matrix = topsis_matrix.drop(columns=zero_norm)
            topsis_weights = topsis_weights.drop(index=zero_norm)
            topsis_weights = topsis_weights / topsis_weights.sum()
            norms = np.sqrt((topsis_matrix**2).sum(axis=0))
        weighted = topsis_matrix.divide(norms, axis=1).multiply(topsis_weights, axis=1)
        ideal_best = weighted.max(axis=0)
        ideal_worst = weighted.min(axis=0)
        d_plus = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
        denominator = d_plus + d_minus
        if np.isclose(denominator, 0.0).any():
            bad = denominator.index[np.isclose(denominator, 0.0)].tolist()
            raise ValueError(f"TOPSIS closeness denominator is zero for scenarios: {bad}")
        closeness = d_minus / denominator
        topsis = pd.DataFrame(
            {"scenario_code": closeness.index, "TOPSIS_score": closeness.to_numpy() * 100.0}
        )

    diagnostics = {
        **mcda_input.diagnostics,
        "zero_variance_criteria": zero_variance,
        "topsis_criterion_count": topsis_matrix.shape[1],
    }
    LOGGER.info("MCDA results: %s", diagnostics)
    return MCDAResult(wsm, topsis, eligibility, diagnostics)
