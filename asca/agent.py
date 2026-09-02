from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core import ASCASuggestion, FORMAL_BOUNDS, build_model_row, suggest_from_description
from .models import DomainCheck, MetamodelRouter


@dataclass
class ASCAEvaluation:
    suggestion: ASCASuggestion | None
    model_row: dict[str, Any]
    domain: DomainCheck
    predictions: pd.DataFrame
    environment: dict[str, Any]

    def scenario_record(self) -> dict[str, Any]:
        keys = ["scenario_id", "archetype", "archetype_name", "size_class", "strategy", "lambda_intensity", *FORMAL_BOUNDS, "priority_strategy"]
        return {k: self.model_row.get(k) for k in keys}


class ASCAEngine:
    """Thin configuration/orchestration layer. It never fabricates parent-model outputs."""

    def __init__(self, assets_dir: str | Path):
        self.assets_dir = Path(assets_dir)
        self.router = MetamodelRouter(self.assets_dir)

    def suggest(self, description: str) -> ASCASuggestion:
        if not description or not description.strip():
            raise ValueError("A company/scenario description is required.")
        return suggest_from_description(description, self.assets_dir)

    def evaluate(
        self,
        *,
        archetype: str,
        size_class: str,
        strategy: str,
        lambda_intensity: float,
        parameters: dict[str, float],
        suggestion: ASCASuggestion | None = None,
    ) -> ASCAEvaluation:
        row = build_model_row(
            archetype=archetype,
            size_class=size_class,
            strategy=strategy,
            lambda_intensity=lambda_intensity,
            parameters=parameters,
            assets_dir=self.assets_dir,
        )
        domain = self.router.check_domain(row)
        predictions = self.router.predict(row, domain)
        return ASCAEvaluation(suggestion, row, domain, predictions, self.router.environment_status())
