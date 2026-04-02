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
            session.add(KPI(
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
            ))
    session.flush()


def run_composite_indices(context_id="aggregates_ton", dimension_weights=None):
    if dimension_weights is None:
        dimension_weights = {
            "environmental": 0.25,
            "economic": 0.25,
            "social": 0.25,
            "technological": 0.25,
        }

    nr = pd.read_sql(
        """
        SELECT
            n.kpi_id,
            n.scenario_id,
            n.period_end,
            n.normalized_value,
            k.code as kpi_code,
            s.code as scenario_code
        FROM sc_kpi_normalized_result n
        JOIN sc_kpi k ON k.id = n.kpi_id
        JOIN sc_scenario s ON s.id = n.scenario_id
        """,
        engine,
    )

    rules = pd.read_csv("data/kpi_normalization_rules.csv")
    rules = rules[rules["context_id"] == context_id][["kpi_code", "dimension", "weight"]]

    with SessionLocal() as session:
        try:
            ensure_composite_kpis(session)

            if nr.empty or rules.empty:
                session.commit()
                return

            df = nr.merge(rules, on="kpi_code", how="inner")
            if df.empty:
                return

            comp_map = {k.code: k.id for k in session.query(KPI).filter(KPI.code.in_(
                ["ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"]
            )).all()}
            sc_map = {s.code: s.id for s in session.query(Scenario).all()}

            # limpia resultados compuestos previos
            for code in ["ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"]:
                kpi_id = comp_map.get(code)
                if kpi_id:
                    session.query(KPIResult).filter_by(kpi_id=kpi_id).delete()

            session.flush()

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
                    sub = sub.dropna(subset=["normalized_value", "weight"])
                    if sub.empty:
                        continue

                    w = sub["weight"].astype(float)
                    x = sub["normalized_value"].astype(float)
                    score = float((w * x).sum() / w.sum())
                    dim_scores[dim] = score

                    code = dim_to_comp.get(dim)
                    if code and code in comp_map:
                        session.add(KPIResult(
                            kpi_id=comp_map[code],
                            scenario_id=sc_id,
                            product_id=None,
                            facility_id=None,
                            period_start=None,
                            period_end=period_end,
                            value=score,
                        ))

                if dim_scores:
                    valid = {k: v for k, v in dim_scores.items() if not pd.isna(v)}
                    if valid:
                        wdim = {k: dimension_weights.get(k, 0.0) for k in valid.keys()}
                        totw = sum(wdim.values())
                        if totw > 0:
                            wdim = {k: v / totw for k, v in wdim.items()}

                            # media geométrica ponderada correcta en escala 0..100
                            g = 1.0
                            for dim, val in valid.items():
                                g *= (max(float(val), 1e-6) / 100.0) ** wdim[dim]
                            g *= 100.0

                            session.add(KPIResult(
                                kpi_id=comp_map["SUSTAIN_INDEX"],
                                scenario_id=sc_id,
                                product_id=None,
                                facility_id=None,
                                period_start=None,
                                period_end=period_end,
                                value=float(g),
                            ))

            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error in composite indices calculation: {e}")
            raise


if __name__ == "__main__":
    run_composite_indices()