"""Read-only access to the completed SustainSCM MCDA56 evidence package.

These functions retrieve already-computed scientific evidence. They do not
calculate or alter the SustainSCM MCDA results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .core import STRATEGIES


DEFAULT_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "asca" / "mcda56"
)
EXPECTED_ARCHETYPES = tuple(f"RO-A{i}" for i in range(1, 9))
COMPETITIVE_STRATEGIES = tuple(strategy for strategy in STRATEGIES if strategy != "BASE")
DIMENSIONS = ("Environmental", "Economic", "Social", "Technological")

FILE_NAMES = {
    "neutral": "mcda_neutral_48.csv",
    "dimensions": "dimension_indices_56.csv",
    "strategy_robustness": "strategy_robustness_summary.csv",
    "winners": "archetype_winners.csv",
    "weight_profiles": "mcda_all_weight_profiles.csv",
    "random_weights": "bounded_random_weight_winner_frequency.csv",
    "completion": "completion_sensitivity.csv",
    "summary": "summary.json",
}

REQUIRED_COLUMNS = {
    "neutral": {
        "archetype", "archetype_name", "strategy", "profile", *DIMENSIONS,
        "GEOM", "ARITH", "WSM", "WSM_rank", "TOPSIS", "TOPSIS_rank",
        "priority_strategy",
    },
    "dimensions": {
        "archetype", "archetype_name", "strategy", "config_id", *DIMENSIONS,
        "SUSTAIN_INDEX_ARITH", "SUSTAIN_INDEX_GEOM",
    },
    "strategy_robustness": {
        "strategy", "method", "rank1_count", "top2_count", "mean_rank",
        "median_rank", "worst_rank", "best_rank", "mean_score",
    },
    "winners": {
        "archetype", "archetype_name", "VSM_priority", "WSM_winner",
        "WSM_score", "TOPSIS_winner", "TOPSIS_score", "WSM_TOPSIS_agree",
        "VSM_WSM_agree", "VSM_TOPSIS_agree",
    },
    "weight_profiles": {
        "archetype", "archetype_name", "strategy", "profile", *DIMENSIONS,
        "GEOM", "ARITH", "WSM", "WSM_rank", "TOPSIS", "TOPSIS_rank",
        "priority_strategy",
    },
    "random_weights": {"archetype", "strategy", "WSM_win_pct", "TOPSIS_win_pct"},
    "completion": {
        "mode", "archetype", "strategy", *DIMENSIONS, "WSM", "WSM_rank",
        "TOPSIS", "TOPSIS_rank",
    },
}

NUMERIC_COLUMNS = {
    "neutral": [*DIMENSIONS, "GEOM", "ARITH", "WSM", "WSM_rank", "TOPSIS", "TOPSIS_rank"],
    "dimensions": [*DIMENSIONS, "SUSTAIN_INDEX_ARITH", "SUSTAIN_INDEX_GEOM"],
    "strategy_robustness": [
        "rank1_count", "top2_count", "mean_rank", "median_rank", "worst_rank",
        "best_rank", "mean_score",
    ],
    "winners": ["WSM_score", "TOPSIS_score"],
    "weight_profiles": [*DIMENSIONS, "GEOM", "ARITH", "WSM", "WSM_rank", "TOPSIS", "TOPSIS_rank"],
    "random_weights": ["WSM_win_pct", "TOPSIS_win_pct"],
    "completion": [*DIMENSIONS, "WSM", "WSM_rank", "TOPSIS", "TOPSIS_rank"],
}


class MCDAEvidenceError(RuntimeError):
    """Raised when the read-only evidence package is missing or inconsistent."""


@dataclass(frozen=True)
class MCDAEvidence:
    """Validated in-memory views of the source CSVs; no MCDA is executed here."""

    evidence_dir: Path
    neutral: pd.DataFrame
    dimensions: pd.DataFrame
    strategy_robustness: pd.DataFrame
    winners: pd.DataFrame
    weight_profiles: pd.DataFrame
    random_weights: pd.DataFrame
    completion: pd.DataFrame
    summary: Mapping[str, Any]

    @property
    def archetypes(self) -> tuple[str, ...]:
        return tuple(sorted(self.neutral["archetype"].astype(str).unique()))

    @property
    def strategies(self) -> tuple[str, ...]:
        return tuple(sorted(self.neutral["strategy"].astype(str).unique()))


def _require_columns(name: str, frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS[name] - set(frame.columns))
    if missing:
        raise MCDAEvidenceError(
            f"{FILE_NAMES[name]} is missing required columns: {', '.join(missing)}"
        )


def _assert_one_rank_one(frame: pd.DataFrame, group_columns: list[str], label: str) -> None:
    for method in ("WSM", "TOPSIS"):
        counts = frame.loc[frame[f"{method}_rank"].eq(1)].groupby(group_columns).size()
        expected = frame.groupby(group_columns).ngroups
        if len(counts) != expected or not counts.eq(1).all():
            raise MCDAEvidenceError(
                f"{label} must contain exactly one {method} rank-1 row per "
                f"{'/'.join(group_columns)}."
            )


def validate_mcda_evidence_schema(tables: Mapping[str, pd.DataFrame]) -> None:
    """Validate the real package schemas and scientific record cardinalities."""

    for name in REQUIRED_COLUMNS:
        if name not in tables:
            raise MCDAEvidenceError(f"Evidence table not supplied for validation: {name}")
        _require_columns(name, tables[name])

    neutral = tables["neutral"]
    archetypes = set(neutral["archetype"].astype(str))
    strategies = set(neutral["strategy"].astype(str))
    if archetypes != set(EXPECTED_ARCHETYPES):
        raise MCDAEvidenceError(
            f"Neutral MCDA archetypes are {sorted(archetypes)}; expected "
            f"{list(EXPECTED_ARCHETYPES)}."
        )
    if strategies != set(COMPETITIVE_STRATEGIES):
        raise MCDAEvidenceError(
            f"Neutral MCDA strategies are {sorted(strategies)}; expected the six "
            f"non-BASE alternatives {sorted(COMPETITIVE_STRATEGIES)}."
        )
    if neutral.duplicated(["archetype", "strategy"]).any():
        raise MCDAEvidenceError("Duplicate archetype-strategy rows found in neutral MCDA evidence.")
    if len(neutral) != len(EXPECTED_ARCHETYPES) * len(COMPETITIVE_STRATEGIES):
        raise MCDAEvidenceError("Neutral MCDA evidence does not contain 8 x 6 alternatives.")
    if set(neutral["profile"].astype(str)) != {"NEUTRAL"}:
        raise MCDAEvidenceError("The neutral MCDA file contains a non-neutral profile.")
    _assert_one_rank_one(neutral, ["archetype"], "Neutral MCDA evidence")

    dimensions = tables["dimensions"]
    if dimensions.duplicated(["archetype", "strategy"]).any():
        raise MCDAEvidenceError("Duplicate archetype-strategy dimension profiles found.")
    expected_dimension_strategies = set(STRATEGIES)
    for archetype, group in dimensions.groupby("archetype"):
        if set(group["strategy"].astype(str)) != expected_dimension_strategies:
            raise MCDAEvidenceError(f"Incomplete dimension profiles for {archetype}.")
    if len(dimensions) != 56:
        raise MCDAEvidenceError("Dimension evidence must contain the 56 structured anchors.")

    robustness = tables["strategy_robustness"]
    if robustness.duplicated(["strategy", "method"]).any():
        raise MCDAEvidenceError("Duplicate strategy-method robustness rows found.")
    if set(robustness["strategy"].astype(str)) != set(COMPETITIVE_STRATEGIES):
        raise MCDAEvidenceError("Strategy robustness does not cover all six alternatives.")
    if set(robustness["method"].astype(str)) != {"WSM", "TOPSIS"}:
        raise MCDAEvidenceError("Strategy robustness must contain WSM and TOPSIS.")

    winners = tables["winners"]
    if winners.duplicated(["archetype"]).any() or set(winners["archetype"].astype(str)) != archetypes:
        raise MCDAEvidenceError("Archetype-winner evidence must contain one row per archetype.")

    profiles = tables["weight_profiles"]
    if profiles.duplicated(["profile", "archetype", "strategy"]).any():
        raise MCDAEvidenceError("Duplicate deterministic weight-profile rows found.")
    _assert_one_rank_one(profiles, ["profile", "archetype"], "Weight-profile evidence")

    random_weights = tables["random_weights"]
    if random_weights.duplicated(["archetype", "strategy"]).any():
        raise MCDAEvidenceError("Duplicate bounded random-weight rows found.")

    completion = tables["completion"]
    if completion.duplicated(["mode", "archetype", "strategy"]).any():
        raise MCDAEvidenceError("Duplicate completion-sensitivity rows found.")
    _assert_one_rank_one(completion, ["mode", "archetype"], "Completion evidence")


@lru_cache(maxsize=8)
def _load_mcda_evidence_cached(directory: str) -> MCDAEvidence:
    evidence_dir = Path(directory)
    missing = [
        evidence_dir / file_name
        for file_name in FILE_NAMES.values()
        if not (evidence_dir / file_name).is_file()
    ]
    if missing:
        raise MCDAEvidenceError(
            "Reference MCDA evidence file(s) missing: "
            + ", ".join(str(path) for path in missing)
        )

    tables: dict[str, pd.DataFrame] = {}
    for name, file_name in FILE_NAMES.items():
        if name == "summary":
            continue
        try:
            frame = pd.read_csv(evidence_dir / file_name)
            for column in NUMERIC_COLUMNS[name]:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            tables[name] = frame
        except Exception as exc:
            raise MCDAEvidenceError(f"Could not read {file_name}: {exc}") from exc

    validate_mcda_evidence_schema(tables)
    try:
        summary = json.loads((evidence_dir / FILE_NAMES["summary"]).read_text(encoding="utf-8"))
    except Exception as exc:
        raise MCDAEvidenceError(f"Could not read {FILE_NAMES['summary']}: {exc}") from exc

    expected_summary = {
        "n_anchors": 56,
        "n_kpi_observations": 1680,
        "n_nonbase_alternatives": 48,
    }
    for key, expected in expected_summary.items():
        if int(summary.get(key, -1)) != expected:
            raise MCDAEvidenceError(
                f"Summary field {key}={summary.get(key)!r}; expected {expected}."
            )

    return MCDAEvidence(evidence_dir=evidence_dir, summary=summary, **tables)


def load_mcda_evidence(evidence_dir: str | Path | None = None) -> MCDAEvidence:
    """Load and cache the existing CSV evidence without recalculating any score."""

    directory = Path(evidence_dir or DEFAULT_EVIDENCE_DIR).resolve()
    return _load_mcda_evidence_cached(str(directory))


def clear_mcda_evidence_cache() -> None:
    """Clear only the read cache, primarily for controlled data-update tests."""

    _load_mcda_evidence_cached.cache_clear()


def get_archetype_ranking(
    archetype: str, evidence: MCDAEvidence | None = None
) -> pd.DataFrame:
    """Return the six source ranking rows for one reference archetype."""

    data = evidence or load_mcda_evidence()
    ranking = data.neutral.loc[data.neutral["archetype"].eq(archetype)].copy()
    if ranking.empty:
        raise KeyError(f"No neutral MCDA evidence for archetype {archetype}.")
    return ranking.sort_values(["WSM_rank", "TOPSIS_rank", "strategy"]).reset_index(drop=True)


def get_archetype_leader(
    archetype: str, evidence: MCDAEvidence | None = None
) -> dict[str, Any]:
    """Retrieve WSM and TOPSIS leaders from their stored rank fields."""

    ranking = get_archetype_ranking(archetype, evidence)
    wsm = ranking.loc[ranking["WSM_rank"].eq(1)].iloc[0]
    topsis = ranking.loc[ranking["TOPSIS_rank"].eq(1)].iloc[0]
    common = str(wsm["strategy"]) if wsm["strategy"] == topsis["strategy"] else None
    return {
        "strategy": common,
        "wsm_strategy": str(wsm["strategy"]),
        "wsm_score": float(wsm["WSM"]),
        "wsm_rank": int(wsm["WSM_rank"]),
        "topsis_strategy": str(topsis["strategy"]),
        "topsis_score": float(topsis["TOPSIS"]),
        "topsis_rank": int(topsis["TOPSIS_rank"]),
        "methods_agree": common is not None,
    }


def get_strategy_robustness(
    strategy: str, evidence: MCDAEvidence | None = None
) -> dict[str, Any] | None:
    """Retrieve stored cross-archetype rank statistics for one strategy."""

    data = evidence or load_mcda_evidence()
    rows = data.strategy_robustness.loc[
        data.strategy_robustness["strategy"].eq(strategy)
    ]
    if rows.empty:
        return None
    result: dict[str, Any] = {"strategy": strategy, "n_archetypes": len(data.archetypes)}
    for row in rows.itertuples(index=False):
        result[str(row.method)] = {
            "rank1_count": int(row.rank1_count),
            "top2_count": int(row.top2_count),
            "mean_rank": float(row.mean_rank),
            "median_rank": float(row.median_rank),
            "worst_rank": int(row.worst_rank),
            "best_rank": int(row.best_rank),
            "mean_score": float(row.mean_score),
        }
    return result


def get_strategy_dimension_profile(
    archetype: str, strategy: str, evidence: MCDAEvidence | None = None
) -> dict[str, float]:
    """Retrieve one stored dimension profile, including the global geometric index."""

    data = evidence or load_mcda_evidence()
    rows = data.dimensions.loc[
        data.dimensions["archetype"].eq(archetype)
        & data.dimensions["strategy"].eq(strategy)
    ]
    if rows.empty:
        raise KeyError(f"No dimension profile for {archetype}/{strategy}.")
    row = rows.iloc[0]
    return {
        **{dimension: float(row[dimension]) for dimension in DIMENSIONS},
        "SUSTAIN_INDEX_GEOM": float(row["SUSTAIN_INDEX_GEOM"]),
    }


def get_archetype_dimension_profiles(
    archetype: str, evidence: MCDAEvidence | None = None
) -> pd.DataFrame:
    """Return all seven stored dimension profiles, including the BASE benchmark."""

    data = evidence or load_mcda_evidence()
    rows = data.dimensions.loc[data.dimensions["archetype"].eq(archetype)].copy()
    if rows.empty:
        raise KeyError(f"No dimension profiles for archetype {archetype}.")
    return rows.reset_index(drop=True)


def _ranked_winner(group: pd.DataFrame, method: str) -> str:
    return str(group.loc[group[f"{method}_rank"].eq(1), "strategy"].iloc[0])


def get_weight_robustness(
    archetype: str,
    strategy: str | None = None,
    evidence: MCDAEvidence | None = None,
) -> dict[str, Any]:
    """Retrieve deterministic and bounded-random preference sensitivity evidence."""

    data = evidence or load_mcda_evidence()
    rows = data.weight_profiles.loc[data.weight_profiles["archetype"].eq(archetype)].copy()
    random_rows = data.random_weights.loc[data.random_weights["archetype"].eq(archetype)].copy()
    if rows.empty or random_rows.empty:
        raise KeyError(f"No weight-robustness evidence for archetype {archetype}.")

    records: list[dict[str, Any]] = []
    for profile, group in rows.groupby("profile", sort=False):
        records.append(
            {
                "profile": str(profile),
                "wsm_winner": _ranked_winner(group, "WSM"),
                "topsis_winner": _ranked_winner(group, "TOPSIS"),
            }
        )
    neutral = next(record for record in records if record["profile"] == "NEUTRAL")
    for record in records:
        record["wsm_retained"] = record["wsm_winner"] == neutral["wsm_winner"]
        record["topsis_retained"] = record["topsis_winner"] == neutral["topsis_winner"]

    bounded: dict[str, Any] = {}
    for method in ("WSM", "TOPSIS"):
        column = f"{method}_win_pct"
        winner_row = random_rows.sort_values([column, "strategy"], ascending=[False, True]).iloc[0]
        neutral_strategy = neutral[f"{method.lower()}_winner"]
        neutral_row = random_rows.loc[random_rows["strategy"].eq(neutral_strategy)].iloc[0]
        bounded[method] = {
            "winner": str(winner_row["strategy"]),
            "winner_frequency_pct": float(winner_row[column]),
            "neutral_winner": neutral_strategy,
            "neutral_winner_frequency_pct": float(neutral_row[column]),
            "retained": str(winner_row["strategy"]) == neutral_strategy,
        }

    strategy_stable = None
    if strategy is not None:
        strategy_rows = rows.loc[rows["strategy"].eq(strategy)]
        random_strategy = random_rows.loc[random_rows["strategy"].eq(strategy)]
        strategy_stable = bool(
            len(strategy_rows) == len(records)
            and strategy_rows["WSM_rank"].eq(1).all()
            and strategy_rows["TOPSIS_rank"].eq(1).all()
            and not random_strategy.empty
            and float(random_strategy.iloc[0]["WSM_win_pct"]) > 50.0
            and float(random_strategy.iloc[0]["TOPSIS_win_pct"]) > 50.0
        )

    return {
        "archetype": archetype,
        "strategy": strategy,
        "profiles": records,
        "neutral": neutral,
        "deterministic_stable": all(
            record["wsm_retained"] and record["topsis_retained"] for record in records
        ),
        "bounded_random": bounded,
        "random_frequencies": random_rows.to_dict(orient="records"),
        "strategy_stable": strategy_stable,
    }


def get_completion_robustness(
    archetype: str,
    strategy: str | None = None,
    evidence: MCDAEvidence | None = None,
) -> dict[str, Any]:
    """Retrieve stored bridge-completion sensitivity winners and score margins."""

    data = evidence or load_mcda_evidence()
    rows = data.completion.loc[data.completion["archetype"].eq(archetype)].copy()
    if rows.empty:
        raise KeyError(f"No completion-sensitivity evidence for archetype {archetype}.")
    records: list[dict[str, Any]] = []
    for mode, group in rows.groupby("mode", sort=False):
        wsm_sorted = group.sort_values(["WSM", "strategy"], ascending=[False, True])
        topsis_sorted = group.sort_values(["TOPSIS", "strategy"], ascending=[False, True])
        records.append(
            {
                "mode": str(mode),
                "wsm_winner": str(wsm_sorted.iloc[0]["strategy"]),
                "wsm_margin": float(wsm_sorted.iloc[0]["WSM"] - wsm_sorted.iloc[1]["WSM"]),
                "topsis_winner": str(topsis_sorted.iloc[0]["strategy"]),
                "topsis_margin": float(
                    topsis_sorted.iloc[0]["TOPSIS"] - topsis_sorted.iloc[1]["TOPSIS"]
                ),
            }
        )
    full = next(record for record in records if record["mode"] == "FULL_BRIDGES")
    for record in records:
        record["wsm_retained"] = record["wsm_winner"] == full["wsm_winner"]
        record["topsis_retained"] = record["topsis_winner"] == full["topsis_winner"]
    strategy_stable = None
    if strategy is not None:
        strategy_stable = all(
            record["wsm_winner"] == strategy and record["topsis_winner"] == strategy
            for record in records
        )
    return {
        "archetype": archetype,
        "strategy": strategy,
        "modes": records,
        "stable": all(
            record["wsm_retained"] and record["topsis_retained"] for record in records
        ),
        "strategy_stable": strategy_stable,
    }


def get_diagnostic_mcda_agreement(
    archetype: str, evidence: MCDAEvidence | None = None
) -> dict[str, Any]:
    """Retrieve stored VSM-C, WSM and TOPSIS leaders and derive agreement level."""

    data = evidence or load_mcda_evidence()
    rows = data.winners.loc[data.winners["archetype"].eq(archetype)]
    if rows.empty:
        raise KeyError(f"No diagnostic-agreement evidence for archetype {archetype}.")
    row = rows.iloc[0]
    choices = [str(row["VSM_priority"]), str(row["WSM_winner"]), str(row["TOPSIS_winner"])]
    unique = len(set(choices))
    agreement = "YES" if unique == 1 else "PARTIAL" if unique == 2 else "NO"
    return {
        "vsm_priority": choices[0],
        "wsm_winner": choices[1],
        "topsis_winner": choices[2],
        "agreement": agreement,
    }


def get_archetype_strategy_evidence(
    archetype: str,
    strategy: str,
    evidence: MCDAEvidence | None = None,
) -> dict[str, Any]:
    """Compose local and cross-archetype evidence without replacing the strategy."""

    data = evidence or load_mcda_evidence()
    if strategy not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {strategy}")
    ranking = get_archetype_ranking(archetype, data)
    selected = ranking.loc[ranking["strategy"].eq(strategy)]
    local = {
        "wsm_score": None,
        "wsm_rank": None,
        "topsis_score": None,
        "topsis_rank": None,
        "alternatives": len(COMPETITIVE_STRATEGIES),
        "competitive": strategy != "BASE",
    }
    if not selected.empty:
        row = selected.iloc[0]
        local.update(
            wsm_score=float(row["WSM"]),
            wsm_rank=int(row["WSM_rank"]),
            topsis_score=float(row["TOPSIS"]),
            topsis_rank=int(row["TOPSIS_rank"]),
        )

    cross = get_strategy_robustness(strategy, data)
    cross_flat: dict[str, Any] = {"n_archetypes": len(data.archetypes)}
    for method in ("WSM", "TOPSIS"):
        stats = cross.get(method, {}) if cross else {}
        for key in (
            "mean_rank", "median_rank", "top2_count", "rank1_count", "worst_rank",
            "best_rank", "mean_score",
        ):
            cross_flat[f"{method.lower()}_{key}"] = stats.get(key)

    weight = get_weight_robustness(archetype, strategy, data)
    completion = get_completion_robustness(archetype, strategy, data)
    dimension_profile = get_strategy_dimension_profile(archetype, strategy, data)
    strongest_dimension = max(DIMENSIONS, key=dimension_profile.get)

    return {
        "archetype": archetype,
        "strategy": strategy,
        "local_mcda": local,
        "reference_leader": get_archetype_leader(archetype, data),
        "cross_archetype": cross_flat,
        "dimension_profile": dimension_profile,
        "strongest_dimension": strongest_dimension,
        "robustness": {
            "weight_stable": bool(weight["strategy_stable"]),
            "completion_stable": bool(completion["strategy_stable"]),
        },
        "diagnostic_agreement": get_diagnostic_mcda_agreement(archetype, data),
    }
