"""Shared numerical-comparison policy for normalization and decision support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NumericalComparisonConfig:
    score_tolerance: float = 1e-5
    value_tolerance: float = 1e-9
    version: str = "1"


NUMERICAL_COMPARISON = NumericalComparisonConfig()


def snap_to_threshold(
    score: float,
    thresholds: Iterable[float],
    tolerance: float = NUMERICAL_COMPARISON.score_tolerance,
) -> float:
    value = float(score)
    for threshold in thresholds:
        boundary = float(threshold)
        if abs(value - boundary) <= tolerance:
            return boundary
    return value


def comparison_effect(delta: float | None) -> str:
    if delta is None:
        return "Missing"
    if abs(float(delta)) <= NUMERICAL_COMPARISON.score_tolerance:
        return "Same"
    return "Improved" if float(delta) > 0 else "Worse"
