from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import traceback
from sqlalchemy import text

# -----------------------------------------------------------------------------
# DB URL
# -----------------------------------------------------------------------------

if os.path.exists("/mount/src"):
    os.environ["SUSTAINSC_DB_URL"] = "sqlite:////tmp/sustainsc.db"


def _default_db_url() -> str:
    if os.getenv("SUSTAINSC_DB_URL"):
        return os.environ["SUSTAINSC_DB_URL"]

    if Path("/mount/src").exists() or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true":
        return "sqlite:////tmp/sustainsc.db"

    db_path = Path(tempfile.gettempdir()) / "sustainsc.db"
    return f"sqlite:///{db_path.as_posix()}"


os.environ.setdefault("SUSTAINSC_DB_URL", _default_db_url())

# -----------------------------------------------------------------------------
# sustainsc imports
# -----------------------------------------------------------------------------

from sustainsc.config import engine, SessionLocal, Base
from sustainsc.dpp_service import (
    build_dpp_passport,
    dpp_passport_to_json,
    summarize_dpp_mrv,
)
from sustainsc.kpi_engine import run_full_pipeline
from sustainsc.models import Measurement, Scenario, ProductBatch, KPIResult, KPINormalizedResult
from sustainsc.dashboard_workflow import assess_analysis_readiness, has_restrictive_filters
from sustainsc.ui import (
    apply_design_system,
    render_data_status_panel,
    render_downloadable_table,
    render_empty_state,
    render_filter_summary,
    render_page_header,
    render_section_header,
    render_workflow_progress,
)
from sustainsc.ui.chart_theme import DIMENSION_COLOR_MAP
from scenario_completion_page import render_scenario_completion_page


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

COMPOSITE_CODES = {"ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"}


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------

def ensure_schema():
    """
    Ensure all tables exist before any SELECT COUNT(*) calls.
    """
    Base.metadata.create_all(bind=engine)


def _safe_count(table_name: str) -> int:
    with engine.connect() as con:
        return int(con.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)


@st.cache_resource(show_spinner=False)
def bootstrap_everything():
    try:
        ensure_schema()

        from load_example_data import load_cost_factors, load_emission_factors, load_kpis

        kpi_count = _safe_count("sc_kpi")
        emission_factor_count = _safe_count("sc_emission_factor")
        cost_factor_count = _safe_count("sc_cost_factor")

        # Reference metadata is required to process uploaded measurements, but
        # scenarios and operational data must always come from the user.
        if kpi_count == 0 or emission_factor_count == 0 or cost_factor_count == 0:
            session = SessionLocal()
            try:
                if emission_factor_count == 0:
                    load_emission_factors(session)
                if cost_factor_count == 0:
                    load_cost_factors(session)
                if kpi_count == 0:
                    load_kpis(session)
            finally:
                session.close()

        return True, None

    except Exception as e:
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def color_semaforo(val):
    if val == "Green":
        return "background-color: #d4edda; color: black"
    if val == "Amber":
        return "background-color: #fff3cd; color: black"
    if val == "Red":
        return "background-color: #f8d7da; color: black"
    if val == "Need BASE":
        return "background-color: #d1ecf1; color: black"
    if val == "Missing":
        return "background-color: #e2e3e5; color: black"
    return ""


def _default_base_index(options):
    if not options:
        return 0
    for i, s in enumerate(options):
        if "BASE" in str(s).upper():
            return i
    return 0


def _apply_common_filters(df, dim_sel, level_sel, flow_sel):
    out = df.copy()
    if dim_sel != "All" and "dimension" in out.columns:
        out = out[out["dimension"] == dim_sel]
    if level_sel != "All" and "decision_level" in out.columns:
        out = out[out["decision_level"] == level_sel]
    if flow_sel != "All" and "flow" in out.columns:
        out = out[out["flow"] == flow_sel]
    return out


def _normalized_delta(ref_score, other_score):
    try:
        if pd.isna(ref_score) or pd.isna(other_score):
            return None
        return float(other_score) - float(ref_score)
    except Exception:
        return None


def _effect_from_normalized_delta(delta_pts, tol=0.5):
    if delta_pts is None or pd.isna(delta_pts):
        return "Missing"
    if float(delta_pts) > tol:
        return "Improved"
    if float(delta_pts) < -tol:
        return "Worse"
    return "Same"


def normalize_dim_weights(raw_weights: dict) -> dict:
    cleaned = {k: max(float(v), 0.0) for k, v in raw_weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        n = len(cleaned)
        return {k: 1.0 / n for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def corrected_sustain_index(dim_scores: dict, dim_weights: dict, method: str = "geometric"):
    dims = ["environmental", "economic", "social", "technological"]

    vals = []
    ws = []
    for d in dims:
        v = dim_scores.get(d, None)
        w = dim_weights.get(d, 0.0)
        if v is None or pd.isna(v) or w <= 0:
            continue
        vals.append(float(v))
        ws.append(float(w))

    if not vals or sum(ws) <= 0:
        return None

    ws = np.array(ws, dtype=float)
    ws = ws / ws.sum()
    vals = np.array(vals, dtype=float)

    if method == "arithmetic":
        return float(np.sum(ws * vals))

    return float(100.0 * np.prod((np.maximum(vals, 1e-6) / 100.0) ** ws))


def _semaforo_badge(val: str) -> str:
    mapping = {
        "Green": "🟢 Green",
        "Amber": "🟠 Amber",
        "Red": "🔴 Red",
        "Need BASE": "🔵 Need BASE",
        "Missing": "⚪ Missing",
    }
    return mapping.get(val, str(val) if val is not None else "")


def render_dpp_passport(passport: dict):
    import pandas as pd
    import streamlit as st

    identity = passport.get("product_identity", {}) or {}
    events = passport.get("traceability_events", []) or []
    validation = passport.get("validation", {}) or {}
    raw_scope = passport.get("sustainability_claims", {}) or {}
    normalized_scope = passport.get("decision_support_summary", {}) or {}
    raw_kpis = raw_scope.get("results", passport.get("raw_kpis", [])) or []
    norm_kpis = normalized_scope.get("results", passport.get("normalized_kpis", [])) or []

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Passport summary", "Traceability events", "KPI summary", "Raw JSON"]
    )

    with tab1:
        st.markdown("### Prototype validation")
        v1, v2 = st.columns(2)
        v1.metric("Status", "Valid" if validation.get("is_valid") else "Invalid")
        v2.metric("Completeness", f"{validation.get('completeness_score', 0):.1f}%")
        for error in validation.get("errors", []):
            st.error(error)
        for warning in validation.get("warnings", []):
            st.warning(warning)
        st.caption("Prototype completeness validation; this is not legal compliance validation.")

        st.markdown("### Product identity")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Product code", identity.get("product_code", "—"))
        c2.metric("Batch code", identity.get("batch_code", "—"))
        c3.metric("Scenario", identity.get("scenario_code", "—"))
        c4.metric("Status", identity.get("status", "—"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Origin facility", identity.get("origin_facility", "—"))
        c6.metric("Quantity", f"{identity.get('quantity', '—')} {identity.get('unit', '')}".strip())
        c7.metric("Production date", identity.get("production_date", "—"))
        c8.metric("Passport type", passport.get("passport_type", "—"))

        st.markdown("*Product name*")
        st.write(identity.get("product_name", "—"))

        st.markdown("*Notes*")
        st.write(identity.get("notes", "—") or "—")

    with tab2:
        st.markdown("### Traceability event history")
        if events:
            events_df = pd.DataFrame(events)
            wanted = [
                "timestamp", "event_type", "facility", "process",
                "transport_leg", "quantity", "unit", "source_system", "comment"
            ]
            wanted = [c for c in wanted if c in events_df.columns]
            render_downloadable_table(
                events_df[wanted],
                filename=f"{identity.get('batch_code', 'batch')}_traceability_events.csv",
                key="download_dpp_traceability_events",
            )
        else:
            st.info("No traceability events found for this batch.")

    with tab3:
        st.markdown("### Normalized KPI")
        st.caption(
            f"Scope: {normalized_scope.get('scope', 'scenario')}. "
            "These values support scenario decisions and are not batch physical properties."
        )
        if norm_kpis:
            norm_df = pd.DataFrame(norm_kpis)
            if "semaforo" in norm_df.columns:
                norm_df["status"] = norm_df["semaforo"].apply(_semaforo_badge)

            wanted = [
                "kpi_code", "kpi_name", "raw_value",
                "normalized_value", "status", "period_end"
            ]
            wanted = [c for c in wanted if c in norm_df.columns]
            render_downloadable_table(
                norm_df[wanted],
                filename=f"{identity.get('batch_code', 'batch')}_normalized_kpis.csv",
                key="download_dpp_normalized_kpis",
            )
        else:
            st.info("No normalized KPI found for this passport.")

        st.markdown("### Raw KPI")
        st.caption(
            f"Scope: {raw_scope.get('scope', 'product_scenario')}. "
            "No scenario total is allocated arbitrarily to this batch."
        )
        if raw_kpis:
            raw_df = pd.DataFrame(raw_kpis)
            wanted = ["kpi_code", "kpi_name", "value", "period_end"]
            wanted = [c for c in wanted if c in raw_df.columns]
            render_downloadable_table(
                raw_df[wanted],
                filename=f"{identity.get('batch_code', 'batch')}_raw_kpis.csv",
                key="download_dpp_raw_kpis",
            )
        else:
            st.info("No raw KPI linked to this batch/product-scenario combination yet.")

    with tab4:
        st.markdown("### Raw passport JSON")
        st.json(passport)


def render_dpp_section() -> None:
    """Render DPP and traceability after the integrated dashboard analyses."""

    render_section_header(
        "Digital Product Passport and traceability",
        "DPP-ready batch-level prototype with identity, validation, event history "
        "and clearly scoped sustainability claims.",
    )
    session = SessionLocal()
    try:
        batches = (
            session.query(ProductBatch)
            .order_by(ProductBatch.batch_code)
            .all()
        )
        batch_options = [batch.batch_code for batch in batches]
        scenario_by_batch = {batch.batch_code: batch.scenario_id for batch in batches}
    finally:
        session.close()

    if not batch_options:
        st.info(
            "No product batches or traceability events were imported with the active dataset."
        )
        return

    batch_code = st.selectbox("Batch code", batch_options, key="dpp_batch_code")
    st.session_state["selected_batch"] = batch_code
    include_raw = st.checkbox(
        "Include product-scenario raw KPI results", value=True, key="dpp_include_raw"
    )
    include_normalized = st.checkbox(
        "Include scenario decision-support results",
        value=True,
        key="dpp_include_normalized",
    )

    session = SessionLocal()
    try:
        passport = build_dpp_passport(
            session,
            batch_code,
            include_raw_kpis=include_raw,
            include_normalized_kpis=include_normalized,
        )
        scenario_id = scenario_by_batch.get(batch_code)
        dpp_summary = (
            summarize_dpp_mrv(session, scenario_id) if scenario_id is not None else None
        )
    finally:
        session.close()

    render_dpp_passport(passport)
    st.download_button(
        "Download DPP JSON",
        dpp_passport_to_json(passport).encode("utf-8"),
        file_name=f"{batch_code}_dpp.json",
        mime="application/json",
    )

    if dpp_summary is not None:
        st.markdown("### Scenario-level DPP MRV summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total batches", int(dpp_summary["dpp_batches_total"]))
        s2.metric("Valid batches", int(dpp_summary["dpp_batches_valid"]))
        s3.metric("DPP volume", f"{dpp_summary['dpp_volume']:.2f}")
        s4.metric("Valid DPP volume", f"{dpp_summary['dpp_valid_volume']:.2f}")
        s5.metric(
            "Average completeness",
            f"{dpp_summary['dpp_completeness_average']:.1f}%",
        )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_kpi_catalog():
    q = """
    SELECT
        id AS kpi_id,
        code AS kpi_code,
        name AS kpi_name,
        dimension,
        decision_level,
        flow,
        unit
    FROM sc_kpi
    WHERE code NOT IN ('ENV_INDEX','ECO_INDEX','SOC_INDEX','TECH_INDEX','SUSTAIN_INDEX')
    ORDER BY code
    """
    df = pd.read_sql(q, engine)
    if not df.empty:
        df["dimension"] = df["dimension"].fillna("unknown")
        df["decision_level"] = df["decision_level"].fillna("unknown")
        df["flow"] = df["flow"].fillna("unknown")
    return df


@st.cache_data(ttl=30)
def load_raw_kpi_results():
    q = """
    SELECT
        s.code AS scenario_code,
        k.code AS kpi_code,
        r.value AS raw_value,
        r.period_end
    FROM sc_kpi_result r
    JOIN sc_kpi k ON k.id = r.kpi_id
    JOIN sc_scenario s ON s.id = r.scenario_id
    WHERE k.code NOT IN ('ENV_INDEX','ECO_INDEX','SOC_INDEX','TECH_INDEX','SUSTAIN_INDEX')
    """
    df = pd.read_sql(q, engine)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df["scenario_code"] = df["scenario_code"].fillna("NONE")
    return df


@st.cache_data(ttl=30)
def load_normalized_results():
    q = """
    SELECT
        n.scenario_id,
        s.code AS scenario_code,
        k.id AS kpi_id,
        k.code AS kpi_code,
        k.name AS kpi_name,
        k.dimension,
        k.decision_level,
        k.flow,
        k.unit,
        n.raw_value,
        n.normalized_value,
        n.semaforo,
        n.lower_ref,
        n.upper_ref,
        n.baseline_value,
        n.normalization_method,
        n.notes,
        n.period_end
    FROM sc_kpi_normalized_result n
    JOIN sc_kpi k ON k.id = n.kpi_id
    JOIN sc_scenario s ON s.id = n.scenario_id
    WHERE k.code NOT IN ('ENV_INDEX','ECO_INDEX','SOC_INDEX','TECH_INDEX','SUSTAIN_INDEX')
    """
    df = pd.read_sql(q, engine)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df["scenario_code"] = df["scenario_code"].fillna("NONE")
        df["dimension"] = df["dimension"].fillna("unknown")
        df["decision_level"] = df["decision_level"].fillna("unknown")
        df["flow"] = df["flow"].fillna("unknown")
    return df


@st.cache_data(ttl=30)
def load_normalization_rules():
    path = Path(__file__).parent / "data" / "kpi_normalization_rules.csv"
    if not path.exists():
        return pd.DataFrame()

    rules = pd.read_csv(path)
    rules.columns = [c.strip().lower() for c in rules.columns]
    rules["kpi_code"] = rules["kpi_code"].astype(str).str.strip()
    rules["dimension"] = rules["dimension"].astype(str).str.strip()
    rules["weight"] = pd.to_numeric(rules["weight"], errors="coerce")
    return rules


def latest_per_kpi_scenario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df2 = df.dropna(subset=["scenario_code", "kpi_code"]).sort_values("period_end")
    return df2.groupby(["scenario_code", "kpi_code"], as_index=False).tail(1)


def build_raw_plus_normalized_table(
    catalog_df: pd.DataFrame,
    raw_latest: pd.DataFrame,
    norm_latest: pd.DataFrame,
    scenario_code: str,
    dim_sel: str,
    level_sel: str,
    flow_sel: str,
):
    base_catalog = _apply_common_filters(catalog_df, dim_sel, level_sel, flow_sel).copy()

    raw_s = raw_latest[raw_latest["scenario_code"] == scenario_code][["kpi_code", "raw_value"]].copy()
    norm_s = norm_latest[norm_latest["scenario_code"] == scenario_code][[
        "kpi_code", "normalized_value", "semaforo", "baseline_value",
        "lower_ref", "upper_ref", "normalization_method"
    ]].copy()

    out = base_catalog.merge(raw_s, on="kpi_code", how="left").merge(norm_s, on="kpi_code", how="left")
    return out.sort_values(["dimension", "kpi_code"])


# -----------------------------------------------------------------------------
# Composite indices, MCDA, sensitivity
# -----------------------------------------------------------------------------

def compute_dimension_indices(norm_latest: pd.DataFrame, rules_df: pd.DataFrame, dim_weights: dict):
    if norm_latest.empty or rules_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    weights = (
        rules_df[["kpi_code", "dimension", "weight"]]
        .dropna(subset=["kpi_code", "dimension", "weight"])
        .drop_duplicates()
        .rename(columns={"dimension": "rule_dimension", "weight": "local_weight"})
    )

    merged = norm_latest.merge(weights, on="kpi_code", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for (scenario_code, rule_dimension), g in merged.groupby(["scenario_code", "rule_dimension"]):
        gg = g.dropna(subset=["normalized_value", "local_weight"]).copy()
        if gg.empty:
            continue

        x = gg["normalized_value"].astype(float).to_numpy()
        w = gg["local_weight"].astype(float).to_numpy()
        if w.sum() <= 0:
            continue

        score = float(np.average(x, weights=w))
        rows.append({
            "scenario_code": scenario_code,
            "dimension": rule_dimension,
            "dimension_index": score,
            "kpis_used": len(gg),
        })

    dim_long = pd.DataFrame(rows)
    if dim_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    dim_wide = dim_long.pivot(index="scenario_code", columns="dimension", values="dimension_index").reset_index()

    for col in ["environmental", "economic", "social", "technological"]:
        if col not in dim_wide.columns:
            dim_wide[col] = np.nan

    dim_wide["SUSTAIN_INDEX_GEOM"] = dim_wide.apply(
        lambda r: corrected_sustain_index(
            {
                "environmental": r.get("environmental"),
                "economic": r.get("economic"),
                "social": r.get("social"),
                "technological": r.get("technological"),
            },
            dim_weights,
            method="geometric",
        ),
        axis=1,
    )

    dim_wide["SUSTAIN_INDEX_ARITH"] = dim_wide.apply(
        lambda r: corrected_sustain_index(
            {
                "environmental": r.get("environmental"),
                "economic": r.get("economic"),
                "social": r.get("social"),
                "technological": r.get("technological"),
            },
            dim_weights,
            method="arithmetic",
        ),
        axis=1,
    )

    return dim_long, dim_wide


def build_normalized_comparison(norm_latest, reference_scenario, selected_scenarios, dim_sel, level_sel, flow_sel, tol):
    df = _apply_common_filters(norm_latest, dim_sel, level_sel, flow_sel)
    df = df[df["scenario_code"].isin([reference_scenario] + selected_scenarios)].copy()

    index_cols = ["kpi_code", "kpi_name", "dimension", "decision_level", "flow", "unit"]

    ref = (
        df[df["scenario_code"] == reference_scenario][index_cols + ["normalized_value", "semaforo"]]
        .rename(columns={
            "normalized_value": "reference_score",
            "semaforo": "reference_semaforo",
        })
        .copy()
    )

    detailed_frames = []
    for sc in selected_scenarios:
        comp = (
            df[df["scenario_code"] == sc][index_cols + ["normalized_value", "semaforo"]]
            .rename(columns={
                "normalized_value": "scenario_score",
                "semaforo": "scenario_semaforo",
            })
            .copy()
        )

        merged = ref.merge(comp, on=index_cols, how="outer")
        merged["reference_scenario"] = reference_scenario
        merged["scenario"] = sc
        merged["delta_pts"] = merged.apply(
            lambda r: _normalized_delta(r.get("reference_score"), r.get("scenario_score")),
            axis=1,
        )
        merged["effect"] = merged["delta_pts"].apply(lambda x: _effect_from_normalized_delta(x, tol=tol))
        detailed_frames.append(merged)

    if not detailed_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    detailed = pd.concat(detailed_frames, ignore_index=True)

    summary = (
        detailed.groupby("scenario", group_keys=False)
        .apply(lambda g: pd.Series({
            "Improved": int((g["effect"] == "Improved").sum()),
            "Worse": int((g["effect"] == "Worse").sum()),
            "Same": int((g["effect"] == "Same").sum()),
            "Missing": int((g["effect"] == "Missing").sum()),
            "Mean Δ (pts)": float(g["delta_pts"].dropna().mean()) if g["delta_pts"].notna().any() else np.nan,
            "Median Δ (pts)": float(g["delta_pts"].dropna().median()) if g["delta_pts"].notna().any() else np.nan,
            "Net score": int((g["effect"] == "Improved").sum()) - int((g["effect"] == "Worse").sum()),
        }), include_groups=False)
        .reset_index()
        .sort_values(["Net score", "Mean Δ (pts)"], ascending=False)
    )

    by_dim = (
        detailed.groupby(["scenario", "dimension"], group_keys=False)
        .apply(lambda g: pd.Series({
            "Improved": int((g["effect"] == "Improved").sum()),
            "Worse": int((g["effect"] == "Worse").sum()),
            "Same": int((g["effect"] == "Same").sum()),
            "Mean Δ (pts)": float(g["delta_pts"].dropna().mean()) if g["delta_pts"].notna().any() else np.nan,
        }), include_groups=False)
        .reset_index()
    )

    return detailed, summary, by_dim


def build_global_kpi_weights(rules_df: pd.DataFrame, dim_weights: dict):
    if rules_df.empty:
        return pd.DataFrame()

    out = rules_df[["kpi_code", "dimension", "weight"]].dropna().copy()
    out["weight"] = out["weight"].astype(float)
    out["local_weight_norm"] = out["weight"] / out.groupby("dimension")["weight"].transform("sum")
    out["dimension_weight"] = out["dimension"].map(dim_weights).fillna(0.0)
    out["global_weight"] = out["local_weight_norm"] * out["dimension_weight"]
    return out[["kpi_code", "dimension", "local_weight_norm", "dimension_weight", "global_weight"]]


def compute_wsm_scores(norm_latest: pd.DataFrame, global_weights: pd.DataFrame, scenario_list: list[str]):
    if norm_latest.empty or global_weights.empty or not scenario_list:
        return pd.DataFrame()

    merged = (
        norm_latest[norm_latest["scenario_code"].isin(scenario_list)]
        .merge(global_weights[["kpi_code", "global_weight"]], on="kpi_code", how="inner")
        .dropna(subset=["normalized_value", "global_weight"])
        .copy()
    )
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for sc, g in merged.groupby("scenario_code"):
        w = g["global_weight"].astype(float).to_numpy()
        x = g["normalized_value"].astype(float).to_numpy()
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        score = float(np.sum(w * x))
        rows.append({
            "scenario_code": sc,
            "WSM_score": score,
            "kpis_used_wsm": len(g),
        })

    return pd.DataFrame(rows)


def compute_topsis_scores(norm_latest: pd.DataFrame, global_weights: pd.DataFrame, scenario_list: list[str]):
    if norm_latest.empty or global_weights.empty or len(scenario_list) < 2:
        return pd.DataFrame()

    work = (
        norm_latest[norm_latest["scenario_code"].isin(scenario_list)]
        .merge(global_weights[["kpi_code", "global_weight"]], on="kpi_code", how="inner")
        .copy()
    )
    if work.empty:
        return pd.DataFrame()

    matrix = work.pivot_table(index="scenario_code", columns="kpi_code", values="normalized_value", aggfunc="first")
    matrix = matrix.reindex([s for s in scenario_list if s in matrix.index])

    complete_cols = [c for c in matrix.columns if matrix[c].notna().all()]
    if not complete_cols:
        return pd.DataFrame()

    X = matrix[complete_cols].astype(float).copy()
    w = (
        global_weights.drop_duplicates(subset=["kpi_code"])
        .set_index("kpi_code")
        .loc[complete_cols, "global_weight"]
        .astype(float)
    )
    if w.sum() <= 0:
        return pd.DataFrame()
    w = w / w.sum()

    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom
    V = R * w

    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)

    d_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    den = d_pos + d_neg

    closeness = np.where(den > 0, (d_neg / den) * 100.0, np.nan)

    return pd.DataFrame({
        "scenario_code": X.index.tolist(),
        "TOPSIS_score": closeness,
        "kpis_used_topsis": len(complete_cols),
    })


def build_one_way_sensitivity(selected_dim_row: pd.Series):
    if selected_dim_row is None or selected_dim_row.empty:
        return pd.DataFrame()

    dims = ["environmental", "economic", "social", "technological"]
    dim_scores = {d: selected_dim_row.get(d, np.nan) for d in dims}

    steps = np.round(np.arange(0.10, 0.71, 0.05), 2)
    rows = []

    for focus in dims:
        others = [d for d in dims if d != focus]
        for a in steps:
            weights = {focus: float(a)}
            rem = (1.0 - float(a)) / len(others)
            for od in others:
                weights[od] = rem

            rows.append({
                "focus_dimension": focus,
                "focus_weight": float(a),
                "SUSTAIN_INDEX_GEOM": corrected_sustain_index(dim_scores, weights, method="geometric"),
                "SUSTAIN_INDEX_ARITH": corrected_sustain_index(dim_scores, weights, method="arithmetic"),
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Measurements import
# -----------------------------------------------------------------------------

def normalize_measurements_upload(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    required = ["scenario_code", "variable_name", "value", "timestamp"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "unit" not in out.columns:
        out["unit"] = ""

    if "source_system" not in out.columns:
        out["source_system"] = "uploaded_measurements_csv"

    if "comment" not in out.columns:
        out["comment"] = ""

    out["scenario_code"] = out["scenario_code"].astype(str).str.strip()
    out["variable_name"] = out["variable_name"].astype(str).str.strip()
    out["unit"] = out["unit"].fillna("").astype(str).str.strip()
    out["source_system"] = out["source_system"].fillna("uploaded_measurements_csv").astype(str).str.strip()
    out["comment"] = out["comment"].fillna("").astype(str).str.strip()

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    bad_value = out["value"].isna().sum()
    bad_ts = out["timestamp"].isna().sum()

    if bad_value > 0:
        raise ValueError(f"'value' contains {bad_value} invalid numeric rows.")
    if bad_ts > 0:
        raise ValueError(f"'timestamp' contains {bad_ts} invalid datetime rows.")

    out = out[out["scenario_code"] != ""].copy()
    out = out[out["variable_name"] != ""].copy()

    return out


def write_measurements_to_db(df: pd.DataFrame, replace_uploaded_scenarios: bool = True):
    session = SessionLocal()
    try:
        sc_map = {s.code: s.id for s in session.query(Scenario).all()}

        uploaded_codes = sorted(df["scenario_code"].dropna().astype(str).str.strip().unique().tolist())
        for scode in uploaded_codes:
            if scode not in sc_map:
                sc = Scenario(
                    code=scode,
                    name=scode,
                    description="auto-created from uploaded measurements",
                    notes="created by dashboard import",
                )
                session.add(sc)
                session.flush()
                sc_map[scode] = sc.id

        if replace_uploaded_scenarios and uploaded_codes:
            ids_to_clear = [sc_map[c] for c in uploaded_codes if c in sc_map]
            if ids_to_clear:
                # IMPORTANT:
                # If measurements are replaced, raw and normalized KPI results for those
                # scenarios must also be removed. Otherwise the dashboard may display
                # stale normalized rows such as "Need BASE" even after importing a
                # corrected CSV.
                session.query(Measurement).filter(Measurement.scenario_id.in_(ids_to_clear)).delete(
                    synchronize_session=False
                )
                session.query(KPIResult).filter(KPIResult.scenario_id.in_(ids_to_clear)).delete(
                    synchronize_session=False
                )
                session.query(KPINormalizedResult).filter(
                    KPINormalizedResult.scenario_id.in_(ids_to_clear)
                ).delete(synchronize_session=False)

                # Relative-vs-BASE normalization depends on the reference scenario.
                # If BASE changes, all normalized rows must be rebuilt.
                if any(str(c).upper() == "BASE" for c in uploaded_codes):
                    session.query(KPINormalizedResult).delete(synchronize_session=False)

                session.flush()

        written = 0
        for _, row in df.iterrows():
            session.add(
                Measurement(
                    scenario_id=sc_map[row["scenario_code"]],
                    variable_name=str(row["variable_name"]).strip(),
                    value=float(row["value"]),
                    unit=str(row["unit"]).strip(),
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    source_system=str(row["source_system"]).strip(),
                    comment=str(row["comment"]).strip(),
                    product_id=None,
                    facility_id=None,
                    process_id=None,
                    transport_leg_id=None,
                )
            )
            written += 1

        session.commit()
        return written, uploaded_codes
    finally:
        session.close()


# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="SustainSCM DSS - KPI Dashboard", layout="wide")
apply_design_system()

boot_ok, boot_msg = bootstrap_everything()
if not boot_ok:
    st.error(f"❌ Failed to bootstrap database: {boot_msg}")
    st.stop()

def _load_uploaded_csv(uploaded_file, loader) -> int:
    if uploaded_file is None:
        return 0
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp.write(uploaded_file.getbuffer())
        path = Path(temp.name)
    try:
        return int(loader(path))
    finally:
        path.unlink(missing_ok=True)


def import_completed_mrv(result, *, batches_file=None, events_file=None):
    """Persist a validated completion result and refresh every KPI output."""
    from load_product_batches import load_product_batches_file
    from load_traceability_events import load_traceability_events_file

    completed = normalize_measurements_upload(result.software_upload)
    written, imported_codes = write_measurements_to_db(
        completed,
        replace_uploaded_scenarios=True,
    )
    batches_written = _load_uploaded_csv(batches_file, load_product_batches_file)
    events_written = _load_uploaded_csv(events_file, load_traceability_events_file)
    run_full_pipeline(debug_missing=False)
    load_kpi_catalog.clear()
    load_raw_kpi_results.clear()
    load_normalized_results.clear()
    load_normalization_rules.clear()
    run_ids = sorted(
        {
            scenario_result.run_id
            for scenario_result in result.scenario_results.values()
            if scenario_result.run_id
        }
    )
    st.session_state["import_completed"] = True
    st.session_state["active_scenario"] = imported_codes[0] if imported_codes else None
    st.session_state["last_import_run_id"] = ",".join(run_ids) or str(uuid.uuid4())
    st.session_state["last_import_timestamp"] = datetime.utcnow().isoformat()
    st.session_state["selected_batch"] = None
    st.session_state["show_import_page"] = False
    st.success(
        f"Imported {written} measurements, {batches_written} batches and "
        f"{events_written} traceability events."
    )
    st.rerun()


if _safe_count("sc_measurement") == 0 or st.session_state.get("show_import_page", False):
    render_page_header(
        "SustainSCM DSS",
        "Data-driven decision support for sustainable supply-chain management, "
        "causal MRV completion, scenario evaluation and traceability.",
        metadata="Data Import · No active analytical dataset" if _safe_count("sc_measurement") == 0 else "Data Import",
    )
    render_workflow_progress(
        {
            "Import": "ready",
            "Validate": "pending",
            "Complete MRV": "pending",
            "Calculate KPIs": "pending",
            "Explore DPP": "pending",
        }
    )
    render_empty_state(
        "Load an MRV scenario workbook",
        "Upload the scientific input template to validate evidence, complete "
        "missing variables and calculate decision-support indicators.",
    )
    render_section_header(
        "Guided data import",
        "Step 1 — optional batch context. Step 2 — MRV workbook validation. "
        "Step 3 — review and commit.",
    )
    batches_upload = st.file_uploader(
        "Product batches CSV (optional)",
        type=["csv"],
        key="initial_product_batches_csv",
        help="Load this before traceability events so batch references can be resolved.",
    )
    events_upload = st.file_uploader(
        "Traceability events CSV (optional)",
        type=["csv"],
        key="initial_traceability_events_csv",
    )
    render_scenario_completion_page(
        config_dir=Path(__file__).resolve().parent / "config",
        on_commit=lambda result: import_completed_mrv(
            result,
            batches_file=batches_upload,
            events_file=events_upload,
        ),
    )
    st.stop()

with engine.connect() as connection:
    data_status = {
        "scenarios": int(connection.execute(text("SELECT COUNT(*) FROM sc_scenario")).scalar() or 0),
        "measurements": int(connection.execute(text("SELECT COUNT(*) FROM sc_measurement")).scalar() or 0),
        "batches": int(connection.execute(text("SELECT COUNT(*) FROM sc_product_batch")).scalar() or 0),
        "events": int(connection.execute(text("SELECT COUNT(*) FROM sc_traceability_event")).scalar() or 0),
        "last_measurement": connection.execute(text("SELECT MAX(timestamp) FROM sc_measurement")).scalar(),
    }

render_page_header(
    "KPI Dashboard",
    "Integrated scenario, normalization, sustainability-index and decision-ranking analysis.",
    metadata=(
        "Reference: BASE · Last update: "
        + str(st.session_state.get("last_import_timestamp") or data_status["last_measurement"] or "unknown")
    ),
)
render_workflow_progress(
    {
        "Import": "complete",
        "Validate": "complete",
        "Complete MRV": "complete",
        "Calculate KPIs": "complete",
        "Explore DPP": "ready" if data_status["batches"] else "pending",
    }
)
render_section_header(
    "Active data context",
    "Committed database records used by every analytical view on this page.",
)
render_data_status_panel(
    {
        "Scenarios": data_status["scenarios"],
        "Measurements": data_status["measurements"],
        "Batches": data_status["batches"],
        "Traceability events": data_status["events"],
    }
)
if st.button("Go to Data Import", key="go_to_data_import"):
    st.session_state["show_import_page"] = True
    st.rerun()

st.sidebar.header("Controls")


# -----------------------------------------------------------------------------
# Load all data
# -----------------------------------------------------------------------------

catalog_df = load_kpi_catalog()
raw_df = load_raw_kpi_results()
norm_df = load_normalized_results()
rules_df = load_normalization_rules()

if catalog_df.empty:
    st.warning("⚠️ KPI catalog is empty.")
    st.stop()

if norm_df.empty:
    st.warning(
        "No normalized KPI results are available. Return to Data Import and "
        "load a valid MRV workbook before opening the dashboard."
    )
    st.stop()

raw_latest = latest_per_kpi_scenario(raw_df)
norm_latest = latest_per_kpi_scenario(norm_df)
full_dashboard_df = norm_latest.copy(deep=True)

# Sidebar filters from KPI catalog
dimensions = ["All"] + sorted(catalog_df["dimension"].dropna().unique().tolist())
decision_levels = ["All"] + sorted(catalog_df["decision_level"].dropna().unique().tolist())
flows = ["All"] + sorted(catalog_df["flow"].dropna().unique().tolist())
scenario_options = sorted(norm_latest["scenario_code"].dropna().unique().tolist())

def reset_table_filters() -> None:
    st.session_state["filter_dimension"] = "All"
    st.session_state["filter_level"] = "All"
    st.session_state["filter_flow"] = "All"


st.sidebar.caption("Filters affect the detailed KPI table only.")
st.sidebar.button("Reset table filters", on_click=reset_table_filters, width="stretch")
sel_dim = st.sidebar.selectbox("Dimension", dimensions, index=0, key="filter_dimension")
sel_level = st.sidebar.selectbox("Decision level", decision_levels, index=0, key="filter_level")
sel_flow = st.sidebar.selectbox("Flow", flows, index=0, key="filter_flow")
sel_scenario = st.sidebar.selectbox(
    "Scenario (main view)",
    scenario_options,
    index=_default_base_index(scenario_options),
    key="filter_scenario",
)
all_dimensions = dimensions[1:]
all_levels = decision_levels[1:]
all_flows = flows[1:]
restrictive_filters = has_restrictive_filters(
    all_dimensions if sel_dim == "All" else [sel_dim],
    all_dimensions,
    all_levels if sel_level == "All" else [sel_level],
    all_levels,
    all_flows if sel_flow == "All" else [sel_flow],
    all_flows,
)

# -----------------------------------------------------------------------------
# Section 1: Raw KPI values + normalized interpretation
# -----------------------------------------------------------------------------

render_section_header(
    "Detailed KPI evidence",
    f"Raw values, normalized interpretation and status for scenario {sel_scenario}.",
)
render_filter_summary(
    {"Dimension": sel_dim, "Decision level": sel_level, "Flow": sel_flow}
)
st.caption(
    "This table displays the raw KPI values for technical interpretation and, alongside them, "
    "the normalized score and traffic-light classification. Comparative analyses below use normalized scores."
)

filtered_table_df = build_raw_plus_normalized_table(
    catalog_df=catalog_df,
    raw_latest=raw_latest,
    norm_latest=norm_latest,
    scenario_code=sel_scenario,
    dim_sel=sel_dim,
    level_sel=sel_level,
    flow_sel=sel_flow,
)

show_cols = [
    "kpi_code", "kpi_name", "dimension", "decision_level", "flow", "unit",
    "raw_value", "normalized_value", "semaforo", "baseline_value",
    "lower_ref", "upper_ref", "normalization_method"
]
show_cols = [c for c in show_cols if c in filtered_table_df.columns]

styled_main = filtered_table_df[show_cols].style.map(color_semaforo, subset=["semaforo"])
render_downloadable_table(
    filtered_table_df[show_cols],
    filename=f"{sel_scenario}_detailed_kpis.csv",
    key="download_detailed_kpis",
    display_data=styled_main,
)
st.caption(f"Rows shown: {len(filtered_table_df)} KPI base items.")

if restrictive_filters:
    st.info(
        "The active sidebar filters are applied to the detailed KPI table only. "
        "Integrated indices, sensitivity analysis, and scenario-ranking views "
        "require the complete set of scenarios and sustainability dimensions. "
        "Clear the filters to display those analyses."
    )
    render_dpp_section()
    st.stop()

# -----------------------------------------------------------------------------
# Section 2: Normalized scenario comparison vs reference
# -----------------------------------------------------------------------------

st.markdown('<div id="scenario-compare"></div>', unsafe_allow_html=True)
render_section_header(
    "Normalized scenario comparison",
    "Deviation from the reference after KPI directionality has been encoded.",
)
st.caption("All scenario deviations are computed using normalized KPI scores, so directionality is already encoded.")

base_like = [s for s in scenario_options if "BASE" in s.upper()]
ref_default = base_like[0] if base_like else scenario_options[0]
analysis_readiness = assess_analysis_readiness(
    full_dashboard_df,
    all_scenarios=scenario_options,
    reference_scenario=ref_default,
)
if not analysis_readiness.ready:
    st.warning(
        "Integrated analyses are unavailable because the imported dataset is "
        f"incomplete. {analysis_readiness.message}"
    )
    render_dpp_section()
    st.stop()

reference_scenario = st.selectbox(
    "Reference scenario",
    scenario_options,
    index=scenario_options.index(ref_default),
    key="reference_scenario_norm"
)

default_compare = [s for s in scenario_options if s != reference_scenario][:4]
compare_scenarios = st.multiselect(
    "Scenarios to compare against the reference",
    options=[s for s in scenario_options if s != reference_scenario],
    default=default_compare,
    key="compare_scenarios_norm"
)

same_tolerance = st.slider(
    "Tolerance for 'Same' (normalized points)",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1,
)

detailed_cmp, summary_cmp, by_dim_cmp = build_normalized_comparison(
    norm_latest=full_dashboard_df,
    reference_scenario=reference_scenario,
    selected_scenarios=compare_scenarios,
    dim_sel="All",
    level_sel="All",
    flow_sel="All",
    tol=same_tolerance,
)

if detailed_cmp.empty:
    st.info("No normalized comparison data available for the selected filters.")
else:
    st.markdown("### Summary: improved / worse / same")
    render_downloadable_table(
        summary_cmp,
        filename="normalized_comparison_summary.csv",
        key="download_comparison_summary_table",
    )

    st.markdown("### Summary by dimension")
    by_dim_show = by_dim_cmp.sort_values(["scenario", "dimension"])
    render_downloadable_table(
        by_dim_show,
        filename="normalized_comparison_by_dimension.csv",
        key="download_comparison_dimension_table",
    )

    st.markdown("### Detailed KPI effects (normalized)")
    det_show = detailed_cmp[
        [
            "scenario", "kpi_code", "kpi_name", "dimension",
            "reference_score", "scenario_score", "delta_pts",
            "reference_semaforo", "scenario_semaforo", "effect"
        ]
    ].sort_values(["scenario", "dimension", "kpi_code"])
    render_downloadable_table(
        det_show,
        filename="normalized_comparison_detail.csv",
        key="download_comparison_detail_table",
    )

    if compare_scenarios:
        st.markdown("### Top improvers / worsenings")
        focus_scenario = st.selectbox(
            "Scenario for top movers",
            options=compare_scenarios,
            key="focus_scenario_top_movers"
        )
        focus_df = detailed_cmp[detailed_cmp["scenario"] == focus_scenario].copy()

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Top improvements**")
            top_imp = focus_df.sort_values("delta_pts", ascending=False).head(10)
            top_imp_show = top_imp[["kpi_code", "kpi_name", "dimension", "delta_pts", "effect"]]
            render_downloadable_table(
                top_imp_show,
                filename=f"{focus_scenario}_top_improvements.csv",
                key="download_top_improvements",
            )

        with col_b:
            st.write("**Top worsenings**")
            top_wrs = focus_df.sort_values("delta_pts", ascending=True).head(10)
            top_wrs_show = top_wrs[["kpi_code", "kpi_name", "dimension", "delta_pts", "effect"]]
            render_downloadable_table(
                top_wrs_show,
                filename=f"{focus_scenario}_top_worsenings.csv",
                key="download_top_worsenings",
            )

    st.markdown("### Traffic-light distribution by scenario")
    selected_for_traffic = [reference_scenario] + compare_scenarios
    traffic_base = full_dashboard_df.copy()
    traffic_base = traffic_base[traffic_base["scenario_code"].isin(selected_for_traffic)].copy()

    if traffic_base.empty:
        st.info("No traffic-light data available for selected scenarios.")
    else:
        traffic_df = (
            traffic_base.groupby(["scenario_code", "semaforo"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for col in ["Green", "Amber", "Red", "Need BASE", "Missing"]:
            if col not in traffic_df.columns:
                traffic_df[col] = 0

        traffic_df = traffic_df[
            ["scenario_code", "Green", "Amber", "Red", "Need BASE", "Missing"]
        ].sort_values("scenario_code")

        render_downloadable_table(
            traffic_df,
            filename="traffic_light_distribution.csv",
            key="download_traffic_distribution",
        )

    st.download_button(
        "📥 Download normalized comparison summary (CSV)",
        summary_cmp.to_csv(index=False).encode("utf-8"),
        file_name="normalized_comparison_summary.csv",
        mime="text/csv",
    )

    st.download_button(
        "📥 Download normalized comparison detail (CSV)",
        det_show.to_csv(index=False).encode("utf-8"),
        file_name="normalized_comparison_detail.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# Section 3: Composite indices, sensitivity and MCDA
# -----------------------------------------------------------------------------

render_section_header(
    "Integrated sustainability and decision analysis",
    "Composite indices, weight sensitivity, WSM and TOPSIS rankings.",
)
st.caption(
    "Dimension indices are weighted averages of normalized KPI scores within each dimension. "
    "The corrected SUSTAIN_INDEX is the weighted geometric mean of the four dimension indices."
)

st.markdown("### Dimension weights for global sustainability analysis")
wcol1, wcol2, wcol3, wcol4 = st.columns(4)

w_env_raw = wcol1.slider("Environmental", 0.0, 100.0, 25.0, 1.0)
w_eco_raw = wcol2.slider("Economic", 0.0, 100.0, 25.0, 1.0)
w_soc_raw = wcol3.slider("Social", 0.0, 100.0, 25.0, 1.0)
w_tech_raw = wcol4.slider("Technological", 0.0, 100.0, 25.0, 1.0)

dim_weights = normalize_dim_weights({
    "environmental": w_env_raw,
    "economic": w_eco_raw,
    "social": w_soc_raw,
    "technological": w_tech_raw,
})

st.write(
    pd.DataFrame({
        "dimension": list(dim_weights.keys()),
        "normalized_weight": list(dim_weights.values()),
    })
)

dim_long_df, dim_wide_df = compute_dimension_indices(full_dashboard_df, rules_df, dim_weights)

if dim_wide_df.empty:
    st.info("No composite/dimension indices could be computed from normalized KPI results.")
else:
    st.markdown("### Composite index cards")
    selected_dim_row = dim_wide_df[dim_wide_df["scenario_code"] == sel_scenario]
    if selected_dim_row.empty:
        selected_dim_row = dim_wide_df.iloc[[0]]

    r = selected_dim_row.iloc[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ENV_INDEX", f"{r.get('environmental', np.nan):.1f}" if pd.notna(r.get("environmental")) else "—")
    c2.metric("ECO_INDEX", f"{r.get('economic', np.nan):.1f}" if pd.notna(r.get("economic")) else "—")
    c3.metric("SOC_INDEX", f"{r.get('social', np.nan):.1f}" if pd.notna(r.get("social")) else "—")
    c4.metric("TECH_INDEX", f"{r.get('technological', np.nan):.1f}" if pd.notna(r.get("technological")) else "—")
    c5.metric("SUSTAIN_INDEX", f"{r.get('SUSTAIN_INDEX_GEOM', np.nan):.1f}" if pd.notna(r.get("SUSTAIN_INDEX_GEOM")) else "—")
    c6.metric("Arithmetic alt.", f"{r.get('SUSTAIN_INDEX_ARITH', np.nan):.1f}" if pd.notna(r.get("SUSTAIN_INDEX_ARITH")) else "—")

    st.markdown("### Dimension indices by scenario")
    dim_show = dim_wide_df[
        [
            "scenario_code", "environmental", "economic", "social", "technological",
            "SUSTAIN_INDEX_GEOM", "SUSTAIN_INDEX_ARITH"
        ]
    ].sort_values("SUSTAIN_INDEX_GEOM", ascending=False)
    render_downloadable_table(
        dim_show,
        filename="dimension_indices_by_scenario.csv",
        key="download_dimension_indices",
    )

    profile_long = dim_show.melt(
        id_vars="scenario_code",
        value_vars=["environmental", "economic", "social", "technological"],
        var_name="dimension",
        value_name="score",
    )
    profile_fig = px.bar(
        profile_long,
        x="scenario_code",
        y="score",
        color="dimension",
        barmode="group",
        color_discrete_map=DIMENSION_COLOR_MAP,
        title="Dimension profile by scenario",
        labels={"scenario_code": "Scenario", "score": "Normalized score", "dimension": "Dimension"},
        template="sustainscm",
    )
    profile_fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra>%{fullData.name}</extra>"
    )
    profile_fig.update_yaxes(range=[0, 100])
    st.plotly_chart(profile_fig, width="stretch", config={"displaylogo": False})

    st.markdown("### Corrected Sustain Index ranking")
    ranking = dim_show.sort_values("SUSTAIN_INDEX_GEOM", ascending=True)
    ranking_fig = px.bar(
        ranking,
        x="SUSTAIN_INDEX_GEOM",
        y="scenario_code",
        orientation="h",
        title="Geometric sustainability index ranking",
        labels={"SUSTAIN_INDEX_GEOM": "Index score", "scenario_code": "Scenario"},
        template="sustainscm",
        color_discrete_sequence=["#087F78"],
    )
    ranking_fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Geometric index: %{x:.2f}<extra></extra>"
    )
    ranking_fig.update_xaxes(range=[0, 100])
    st.plotly_chart(ranking_fig, width="stretch", config={"displaylogo": False})

    st.markdown("### Sensitivity analysis for selected scenario")
    sens_df = build_one_way_sensitivity(r)
    if not sens_df.empty:
        geom_fig = px.line(
            sens_df,
            x="focus_weight",
            y="SUSTAIN_INDEX_GEOM",
            color="focus_dimension",
            color_discrete_map=DIMENSION_COLOR_MAP,
            markers=True,
            title="One-way sensitivity — geometric index",
            labels={
                "focus_weight": "Focused dimension weight",
                "SUSTAIN_INDEX_GEOM": "Geometric index",
                "focus_dimension": "Dimension",
            },
            template="sustainscm",
        )
        geom_fig.update_traces(
            hovertemplate="Weight: %{x:.2f}<br>Index: %{y:.2f}<extra>%{fullData.name}</extra>"
        )
        st.plotly_chart(geom_fig, width="stretch", config={"displaylogo": False})

        arith_fig = px.line(
            sens_df,
            x="focus_weight",
            y="SUSTAIN_INDEX_ARITH",
            color="focus_dimension",
            color_discrete_map=DIMENSION_COLOR_MAP,
            markers=True,
            title="One-way sensitivity — arithmetic alternative",
            labels={
                "focus_weight": "Focused dimension weight",
                "SUSTAIN_INDEX_ARITH": "Arithmetic index",
                "focus_dimension": "Dimension",
            },
            template="sustainscm",
        )
        arith_fig.update_traces(
            hovertemplate="Weight: %{x:.2f}<br>Index: %{y:.2f}<extra>%{fullData.name}</extra>"
        )
        st.plotly_chart(arith_fig, width="stretch", config={"displaylogo": False})

        with st.expander("Show sensitivity table"):
            render_downloadable_table(
                sens_df,
                filename=f"{sel_scenario}_sensitivity_analysis.csv",
                key="download_sensitivity_table",
            )

    st.markdown("### MCDA (normalized KPI scores)")
    st.caption(
        "WSM uses the normalized KPI scores and the rule weights. "
        "TOPSIS is computed on the subset of KPI that are complete across the selected scenarios."
    )

    default_mcda = [reference_scenario] + compare_scenarios if compare_scenarios else [reference_scenario]
    mcda_scenarios = st.multiselect(
        "Scenarios for MCDA ranking",
        options=scenario_options,
        default=[s for s in default_mcda if s in scenario_options],
        key="mcda_scenarios"
    )

    global_weights = build_global_kpi_weights(rules_df, dim_weights)
    wsm_df = compute_wsm_scores(full_dashboard_df, global_weights, mcda_scenarios)
    topsis_df = compute_topsis_scores(full_dashboard_df, global_weights, mcda_scenarios)

    mcda_df = pd.merge(wsm_df, topsis_df, on="scenario_code", how="outer")
    if not mcda_df.empty:
        mcda_df["Rank_WSM"] = mcda_df["WSM_score"].rank(ascending=False, method="dense")
        if "TOPSIS_score" in mcda_df.columns:
            mcda_df["Rank_TOPSIS"] = mcda_df["TOPSIS_score"].rank(ascending=False, method="dense")
        mcda_df = mcda_df.sort_values(["Rank_WSM", "scenario_code"])

        render_downloadable_table(
            mcda_df,
            filename="mcda_scenario_ranking.csv",
            key="download_mcda_ranking",
        )

        if "WSM_score" in mcda_df.columns:
            st.write("**WSM ranking**")
            wsm_plot = mcda_df.sort_values("WSM_score", ascending=True)
            wsm_fig = px.bar(
                wsm_plot,
                x="WSM_score",
                y="scenario_code",
                orientation="h",
                title="Weighted-sum ranking",
                labels={"WSM_score": "WSM score", "scenario_code": "Scenario"},
                template="sustainscm",
                color_discrete_sequence=["#2F6B9A"],
            )
            wsm_fig.update_traces(hovertemplate="<b>%{y}</b><br>WSM: %{x:.3f}<extra></extra>")
            st.plotly_chart(wsm_fig, width="stretch", config={"displaylogo": False})

        if "TOPSIS_score" in mcda_df.columns:
            st.write("**TOPSIS ranking**")
            topsis_plot = mcda_df.sort_values("TOPSIS_score", ascending=True)
            topsis_fig = px.bar(
                topsis_plot,
                x="TOPSIS_score",
                y="scenario_code",
                orientation="h",
                title="TOPSIS closeness ranking",
                labels={"TOPSIS_score": "TOPSIS closeness", "scenario_code": "Scenario"},
                template="sustainscm",
                color_discrete_sequence=["#6657A6"],
            )
            topsis_fig.update_traces(
                hovertemplate="<b>%{y}</b><br>TOPSIS: %{x:.3f}<extra></extra>"
            )
            st.plotly_chart(topsis_fig, width="stretch", config={"displaylogo": False})
    else:
        st.info("MCDA ranking could not be computed with the current scenario selection.")

render_dpp_section()
