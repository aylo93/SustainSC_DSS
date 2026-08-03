"""Canonical KPI formula identifiers and deprecated compatibility aliases."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FORMULA_ALIASES = {
    "energy_fuel_cost_share": "energy_cost_share",
    "oee_total": "oee",
    "community_complaints_count": "community_incidents_total",
    "dpp_covered_volume": "dpp_valid_volume",
    "decisions_supported_by_analytics": "analytics_supported_decisions",
    "average_lead_time_days": "average_lead_time",
    "lead_time_days": "average_lead_time",
}


def resolve_formula_id(formula_id: str) -> str:
    """Resolve a known semantic alias without masking unknown formula IDs."""

    normalized = (formula_id or "").strip()
    canonical = FORMULA_ALIASES.get(normalized, normalized)
    if canonical != normalized:
        logger.warning("Deprecated formula ID %s resolved to %s", normalized, canonical)
    return canonical
