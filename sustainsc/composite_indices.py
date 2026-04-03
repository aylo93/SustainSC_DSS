from __future__ import annotations

import pandas as pd
import numpy as np

from sustainsc.config import engine, SessionLocal
from sustainsc.models import KPI, KPIResult, Scenario


COMPOSITE_KPIS = [
    ("ENV_INDEX", "Environmental Index", "environmental"),
    ("ECO_INDEX", "Economic Index", "economic"),
    ("SOC_INDEX", "Social Index", "social"),
    ("TECH_INDEX", "Technological Index", "technological"),
    ("SUSTAIN_INDEX", "Global Sustainability Index", "sustainability"),
]


def ensure_composite_kpis(session):
    for code, name, dimension in COMPOSITE_KPIS:
        existing = session.query(KPI).filter_by(code=code).first()
        if existing:
            existing.name = name
            existing.dimension = dimension
            existing.decision_level = "strategic"
            existing.flow = "informational"
            existing.unit = "score_0_100"
            existing.is_benefit = True
            existing.formula_id = code.lower()
            existing.protocol_notes = "Composite index from normalized KPI scores"
        else:
            session.add(
                KPI(
                    code=code,
                    name=name,
                    description=name,
                    dimension=dimension,
                    decision_level="strategic",
                    flow="informational",
                    unit="score_0_100",
                    is_benefit=True,
                    formula_id=code.lower(),
                    protocol_notes="Composite index from normalized KPI scores",
                )
            )
    session.flush()


def _normalize_dimension_weights(dimension_weights: dict) -> dict:
    cleaned = {k: max(float(v), 0.0) for k, v in dimension_weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        n = len(cleaned)
        return {k: 1.0 / n for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def _corrected_geometric_sustain_index(dim_scores: dict, dimension_weights: dict) -> float | None:
    dims = ["environmental", "economic", "social", "technological"]

    vals = []
    ws = []
    for d in dims:
        v = dim_scores.get(d, None)
        w = dimension_weights.get(d, 0.0)
        if v is None or pd.isna(v) or w <= 0:
            continue
        vals.append(float(v))
        ws.append(float(w))

    if not vals or sum(ws) <= 0:
        return None

    vals = np.array(vals, dtype=float)
    ws = np.array(ws, dtype=float)
    ws = ws / ws.sum()

    # Correct weighted geometric mean on 0..100 scale
    return float(100.0 * np.prod((np.maximum(vals, 1e-6) / 100.0) ** ws))


def run_composite_indices(context_id: str = "aggregates_ton", dimension_weights: dict | None = None):
    if dimension_weights is None:
        dimension_weights = {
            "environmental": 0.25,
            "economic": 0.25,
            "social": 0.25,
            "technological": 0.25,
        }

    dimension_weights = _normalize_dimension_weights(dimension_weights)

    nr = pd.read_sql(
        """
        SELECT
            n.kpi_id,
            n.scenario_id,
            n.period_end,
            n.normalized_value,
            k.code AS kpi_code,
            s.code AS scenario_code
        FROM sc_kpi_normalized_result n
        JOIN sc_kpi k ON k.id = n.kpi_id
        JOIN sc_scenario s ON s.id = n.scenario_id
        """,
        engine,
    )

    rules = pd.read_csv("data/kpi_normalization_rules.csv")
    rules.columns = [c.strip().lower() for c in rules.columns]
    rules["context_id"] = rules["context_id"].astype(str).str.strip()
    rules["kpi_code"] = rules["kpi_code"].astype(str).str.strip()
    rules["dimension"] = rules["dimension"].astype(str).str.strip()
    rules["weight"] = pd.to_numeric(rules["weight"], errors="coerce")

    rules = rules[rules["context_id"] == context_id][["kpi_code", "dimension", "weight"]].copy()

    session = SessionLocal()
    try:
        ensure_composite_kpis(session)

        comp_map = {
            k.code: k.id
            for k in session.query(KPI).filter(
                KPI.code.in_(["ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"])
            ).all()
        }
        sc_map = {s.code: s.id for s in session.query(Scenario).all()}

        # Clear previous composite results only
        for code in ["ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"]:
            kpi_id = comp_map.get(code)
            if kpi_id:
                session.query(KPIResult).filter_by(kpi_id=kpi_id).delete(synchronize_session=False)

        session.flush()

        # If there are no normalized results or no rules, keep composite KPIs in catalog and stop
        if nr.empty or rules.empty:
            session.commit()
            return

        df = nr.merge(rules, on="kpi_code", how="inner")
        if df.empty:
            session.commit()
            return

        dim_to_comp = {
            "environmental": "ENV_INDEX",
            "economic": "ECO_INDEX",
            "social": "SOC_INDEX",
            "technological": "TECH_INDEX",
        }

        for scenario_code, dfg in df.groupby("scenario_code"):
            sc_id = sc_map.get(scenario_code)
            if sc_id is None:
                continue

            period_end = pd.to_datetime(dfg["period_end"], errors="coerce").max()
            dim_scores = {}

            for dim, sub in dfg.groupby("dimension"):
                sub = sub.dropna(subset=["normalized_value", "weight"]).copy()
                if sub.empty:
                    continue

                w = sub["weight"].astype(float).to_numpy()
                x = sub["normalized_value"].astype(float).to_numpy()

                if w.sum() <= 0:
                    continue

                score = float(np.average(x, weights=w))
                dim_scores[dim] = score

                comp_code = dim_to_comp.get(dim)
                comp_kpi_id = comp_map.get(comp_code)
                if comp_code and comp_kpi_id:
                    session.add(
                        KPIResult(
                            kpi_id=comp_kpi_id,
                            scenario_id=sc_id,
                            product_id=None,
                            facility_id=None,
                            period_start=None,
                            period_end=period_end,
                            value=score,
                        )
                    )

            sustain_value = _corrected_geometric_sustain_index(dim_scores, dimension_weights)
            if sustain_value is not None and "SUSTAIN_INDEX" in comp_map:
                session.add(
                    KPIResult(
                        kpi_id=comp_map["SUSTAIN_INDEX"],
                        scenario_id=sc_id,
                        product_id=None,
                        facility_id=None,
                        period_start=None,
                        period_end=period_end,
                        value=float(sustain_value),
                    )
                )

        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    run_composite_indices()