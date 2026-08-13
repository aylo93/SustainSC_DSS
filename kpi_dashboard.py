from __future__ import annotations

import os
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

    # Local CLI recalculation and the dashboard must use the same database.
    # Deployed/headless environments retain the isolated /tmp database above.
    db_path = Path(__file__).resolve().parent / "sustainsc.db"
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
from sustainsc.dpp_import import (
    DPPImportValidationError,
    import_dpp_workbook,
    read_dpp_workbook,
)
from sustainsc.kpi_engine import run_full_pipeline
from sustainsc.models import (
    Measurement, Scenario, ProductBatch, KPIResult, KPINormalizedResult,
    ImportRun, ImportRunScenario, EmissionFactor,
)
from sustainsc.dataset_scope import (
    activate_import_run,
    assert_scenario_integrity,
    ensure_dataset_schema,
    utc_now_naive,
)
from sustainsc.mrv_validation import (
    canonicalize_common_mrv_units,
    select_common_mrv,
    validate_completed_mrv,
)
from sustainsc.dashboard_workflow import (
    assess_analysis_readiness,
    format_traffic_light_status,
    format_reference_value,
    has_restrictive_filters,
)
from sustainsc.mcda import (
    DIMENSION_ORDER,
    build_mcda_input,
    calculate_mcda,
    canonical_dimension,
    compute_complete_dimension_indices,
    evaluate_scenario_eligibility,
)
from sustainsc.composite_indices import corrected_sustain_index
from sustainsc.numerical import NUMERICAL_COMPARISON, comparison_effect
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
from sustainsc.ui.chart_theme import (
    DIMENSION_COLOR_MAP,
    build_horizontal_ranking_chart,
)
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
    ensure_dataset_schema()


def _safe_count(table_name: str) -> int:
    with engine.connect() as con:
        return int(con.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)


def _has_active_import_run() -> bool:
    with engine.connect() as con:
        return bool(
            con.execute(
                text("SELECT 1 FROM sc_import_run WHERE is_active = 1 LIMIT 1")
            ).scalar()
        )


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


def _effect_from_normalized_delta(delta_pts, tol=None):
    if delta_pts is None or pd.isna(delta_pts):
        return "Missing"
    tolerance = NUMERICAL_COMPARISON.score_tolerance
    if tol is not None:
        tolerance = min(float(tol), tolerance)
    if abs(float(delta_pts)) <= tolerance:
        return "Same"
    return comparison_effect(float(delta_pts))


def normalize_dim_weights(raw_weights: dict) -> dict:
    cleaned = {k: max(float(v), 0.0) for k, v in raw_weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        n = len(cleaned)
        return {k: 1.0 / n for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


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
        active_run_id = st.session_state.get("active_import_run_id")
        active_scenario_ids = [
            row[0]
            for row in session.query(ImportRunScenario.scenario_id)
            .filter(ImportRunScenario.import_run_id == active_run_id)
            .all()
        ]
        batches = (
            session.query(ProductBatch)
            .filter(
                ProductBatch.scenario_id.in_(active_scenario_ids),
                ProductBatch.import_run_id == active_run_id,
            )
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
            import_run_id=st.session_state.get("active_import_run_id"),
        )
        scenario_id = scenario_by_batch.get(batch_code)
        dpp_summary = (
            summarize_dpp_mrv(
                session, scenario_id, st.session_state.get("active_import_run_id")
            ) if scenario_id is not None else None
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
        df["dimension"] = df["dimension"].map(canonical_dimension)
        df["decision_level"] = df["decision_level"].fillna("unknown")
        df["flow"] = df["flow"].fillna("unknown")
    return df


@st.cache_data(ttl=30)
def load_active_context(import_run_id: int) -> dict:
    counts = pd.read_sql(
        text(
            "SELECT COUNT(*) AS measurement_count, "
            "COUNT(DISTINCT scenario_id) AS scenario_count "
            "FROM sc_measurement WHERE import_run_id = :import_run_id"
        ),
        engine,
        params={"import_run_id": import_run_id},
    ).iloc[0]
    dictionary = pd.read_csv(Path(__file__).parent / "config" / "mrv_dictionary.csv")
    common_count = int(
        dictionary["common_upload_variable"]
        .astype(str).str.strip().str.lower()
        .isin({"yes", "true", "1"})
        .sum()
    )
    scenario_count = int(counts["scenario_count"])
    measurement_count = int(counts["measurement_count"])
    expected = scenario_count * common_count
    dpp_counts = pd.read_sql(
        text(
            "SELECT "
            "(SELECT COUNT(*) FROM sc_product_batch WHERE import_run_id=:import_run_id) AS batches, "
            "(SELECT COUNT(*) FROM sc_traceability_event WHERE import_run_id=:import_run_id) AS events"
        ),
        engine,
        params={"import_run_id": import_run_id},
    ).iloc[0]
    dpp_ready = 0
    scoped_session = SessionLocal()
    try:
        for (batch_code,) in (
            scoped_session.query(ProductBatch.batch_code)
            .filter(ProductBatch.import_run_id == import_run_id).all()
        ):
            passport = build_dpp_passport(
                scoped_session, batch_code,
                include_raw_kpis=False, include_normalized_kpis=False,
                import_run_id=import_run_id,
            )
            dpp_ready += int(passport["validation"]["is_valid"])
    finally:
        scoped_session.close()
    return {
        "scenario_count": scenario_count,
        "measurement_count": measurement_count,
        "common_variable_count": common_count,
        "expected_measurement_count": expected,
        "integrity": "PASS" if measurement_count == expected else "FAIL",
        "batch_count": int(dpp_counts["batches"]),
        "event_count": int(dpp_counts["events"]),
        "dpp_ready_count": dpp_ready,
    }


@st.cache_data(ttl=30)
def load_raw_kpi_results(import_run_id: int):
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
      AND r.import_run_id = :import_run_id
    """
    df = pd.read_sql(q, engine, params={"import_run_id": import_run_id})
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df["scenario_code"] = df["scenario_code"].fillna("NONE")
    return df


@st.cache_data(ttl=30)
def load_normalized_results(import_run_id: int):
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
      AND n.import_run_id = :import_run_id
    """
    df = pd.read_sql(q, engine, params={"import_run_id": import_run_id})
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df["scenario_code"] = df["scenario_code"].fillna("NONE")
        df["dimension"] = df["dimension"].map(canonical_dimension)
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
    rules["dimension"] = rules["dimension"].map(canonical_dimension)
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

    metadata = rules_df[["kpi_code", "dimension"]].drop_duplicates()
    weights = rules_df.drop_duplicates("kpi_code").set_index("kpi_code")["weight"]
    dim_long, _ = compute_complete_dimension_indices(norm_latest, metadata, weights)
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
        same = merged["effect"] == "Same"
        merged.loc[same, "scenario_semaforo"] = merged.loc[same, "reference_semaforo"]
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

def normalize_measurements_upload(
    df: pd.DataFrame, *, dictionary: pd.DataFrame | None = None
) -> pd.DataFrame:
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

    dictionary_path = Path(__file__).resolve().parent / "config" / "mrv_dictionary.csv"
    if dictionary is None:
        out = select_common_mrv(out, dictionary_path=dictionary_path)
        out = canonicalize_common_mrv_units(out, dictionary_path=dictionary_path)
        validate_completed_mrv(out, dictionary_path=dictionary_path)
    else:
        required_units = dict(zip(
            dictionary.loc[
                dictionary["common_upload_variable"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"}),
                "variable_name",
            ],
            dictionary.loc[
                dictionary["common_upload_variable"].astype(str).str.strip().str.lower().isin({"yes", "true", "1"}),
                "canonical_unit",
            ],
        ))
        out = out[out["variable_name"].isin(required_units)].copy()
        out["unit"] = out["variable_name"].map(required_units)
        validate_completed_mrv(out, dictionary=dictionary)
    return out


def write_measurements_to_db(
    df: pd.DataFrame,
    replace_uploaded_scenarios: bool = True,
    *,
    dataset_name: str = "Imported dataset",
    source_filename: str | None = None,
    case_id: str | None = None,
    dataset_id: str | None = None,
    schema_version: str | None = None,
    reference_scenario_code: str | None = None,
    factor_set_id: str | None = None,
):
    session = SessionLocal()
    try:
        import_run = ImportRun(
            dataset_name=dataset_name,
            case_id=case_id,
            dataset_id=dataset_id,
            schema_version=schema_version,
            factor_set_id=factor_set_id,
            source_filename=source_filename,
            import_timestamp=utc_now_naive(),
            status="importing",
            reference_scenario_code=reference_scenario_code or (
                "BASE" if "BASE" in set(df["scenario_code"].astype(str)) else None
            ),
            scenario_count=0,
            measurement_count=0,
            is_active=False,
        )
        session.add(import_run)
        session.flush()
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

        # Historical data are preserved. Dataset membership, not scenario names,
        # determines what belongs to the newly active import.
        for scode in uploaded_codes:
            session.add(
                ImportRunScenario(import_run_id=import_run.id, scenario_id=sc_map[scode])
            )
        session.flush()

        written = 0
        for _, row in df.iterrows():
            session.add(
                Measurement(
                    scenario_id=sc_map[row["scenario_code"]],
                    import_run_id=import_run.id,
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

        import_run.scenario_count = len(uploaded_codes)
        import_run.measurement_count = written
        assert_scenario_integrity(session, import_run.id, uploaded_codes)
        activate_import_run(session, import_run)
        session.commit()
        return written, uploaded_codes, import_run.id
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

def import_completed_mrv(result, *, dpp_workbook_bytes: bytes | None = None):
    """Persist a validated completion result and refresh every KPI output."""
    if not result.can_commit:
        raise ValueError("MRV commit blocked: resolve metadata, structural, and production QA blockers first.")
    parsed = result.parsed_workbook
    metadata = parsed.metadata
    completed = normalize_measurements_upload(
        result.software_upload, dictionary=parsed.variable_dictionary
    )
    written, imported_codes, import_run_id = write_measurements_to_db(
        completed,
        replace_uploaded_scenarios=True,
        dataset_name=str(metadata.get("dataset_name")),
        source_filename=getattr(result, "source_filename", None),
        case_id=str(metadata.get("case_id")),
        dataset_id=str(metadata.get("dataset_id")),
        schema_version=str(metadata.get("template_schema_version")),
        reference_scenario_code=str(metadata.get("default_reference_scenario")),
        factor_set_id=str(metadata.get("default_emission_factor_set_id") or "") or None,
    )
    factor_session = SessionLocal()
    try:
        for row in parsed.factor_register.itertuples():
            if str(getattr(row, "approval_status", "")).strip().lower() != "approved":
                continue
            existing = factor_session.query(EmissionFactor).filter_by(code=str(row.factor_code)).first()
            values = {
                "name": str(row.factor_code),
                "activity_type": str(getattr(row, "scope", "") or row.factor_code),
                "unit": str(row.unit), "value": float(row.value),
                "valid_from": pd.to_datetime(getattr(row, "valid_from", None), errors="coerce"),
                "valid_to": pd.to_datetime(getattr(row, "valid_to", None), errors="coerce"),
                "source": str(getattr(row, "source", "")),
                "analytical_role": str(getattr(row, "analytical_role", "")),
                "factor_set_id": str(row.factor_set_id),
            }
            values["valid_from"] = None if pd.isna(values["valid_from"]) else values["valid_from"].to_pydatetime()
            values["valid_to"] = None if pd.isna(values["valid_to"]) else values["valid_to"].to_pydatetime()
            if existing is None:
                factor_session.add(EmissionFactor(code=str(row.factor_code), **values))
            else:
                for key, value in values.items():
                    setattr(existing, key, value)
        factor_session.commit()
    finally:
        factor_session.close()
    dpp_outcome = None
    if dpp_workbook_bytes is not None:
        dpp_session = SessionLocal()
        try:
            try:
                dpp_outcome = import_dpp_workbook(
                    dpp_session,
                    dpp_workbook_bytes,
                    active_import_run_id=import_run_id,
                )
            except DPPImportValidationError as exc:
                st.session_state["show_import_page"] = True
                st.error(
                    "The MRV dataset was created, but DPP validation prevented the "
                    "batch/event transaction. Correct the issues below and reimport "
                    "the DPP workbook into this active dataset."
                )
                render_downloadable_table(
                    pd.DataFrame({"Validation issue": exc.result.errors}),
                    filename="dpp_commit_validation_issues.csv",
                    key="download_dpp_commit_validation_issues",
                )
                return
        finally:
            dpp_session.close()
    run_full_pipeline(debug_missing=False, import_run_id=import_run_id)
    load_kpi_catalog.clear()
    load_active_context.clear()
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
    st.session_state["active_import_run_id"] = import_run_id
    st.session_state["last_import_run_id"] = import_run_id
    st.session_state["last_import_timestamp"] = datetime.utcnow().isoformat()
    st.session_state["selected_batch"] = None
    st.session_state["show_import_page"] = False
    if dpp_outcome is None:
        st.success(f"Imported {written} measurements.")
    else:
        st.session_state["dpp_import_summary"] = dpp_outcome.summaries
        st.session_state["dpp_import_message"] = (
            "DPP and traceability import completed. "
            f"Batches: {dpp_outcome.product_batches.rows_read} read, "
            f"{dpp_outcome.product_batches.created} created, "
            f"{dpp_outcome.product_batches.updated} updated, "
            f"{dpp_outcome.product_batches.rejected} rejected. "
            f"Events: {dpp_outcome.traceability_events.rows_read} read, "
            f"{dpp_outcome.traceability_events.created} created, "
            f"{dpp_outcome.traceability_events.updated} updated, "
            f"{dpp_outcome.traceability_events.rejected} rejected."
        )
    st.rerun()


if not _has_active_import_run() or st.session_state.get("show_import_page", False):
    render_page_header(
        "SustainSCM DSS",
        "Data-driven decision support for sustainable supply-chain management, "
        "causal MRV completion, scenario evaluation and traceability.",
        metadata="Data Import · No active analytical dataset" if not _has_active_import_run() else "Data Import",
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
        "Step 1 — optional DPP and traceability workbook. "
        "Step 2 — MRV scenario workbook validation. "
        "Step 3 — review and commit.",
    )
    render_section_header(
        "DPP & Traceability Data",
        "Upload one Excel workbook containing product batches and their event histories. "
        "Accepted format: XLSX. Required sheets: 01_PRODUCT_BATCHES, 02_TRACEABILITY_EVENTS.",
    )
    dpp_workbook = st.file_uploader(
        "DPP & Traceability workbook",
        type=["xlsx"],
        key="dpp_traceability_workbook_upload",
        help=("Upload one SustainSCM workbook containing the "
              "01_PRODUCT_BATCHES and 02_TRACEABILITY_EVENTS sheets."),
    )
    workbook_bytes = None
    if dpp_workbook is not None:
        try:
            workbook_bytes = dpp_workbook.getvalue()
            batch_preview, event_preview = read_dpp_workbook(workbook_bytes)
            referenced_scenarios = sorted(
                batch_preview.get("scenario_code", pd.Series(dtype=str))
                .dropna().astype(str).str.strip().unique().tolist()
            )
            referenced_facilities = sorted(set(
                batch_preview.get("origin_facility_code", pd.Series(dtype=str))
                .dropna().astype(str).str.strip().tolist()
                + event_preview.get("facility_code", pd.Series(dtype=str))
                .dropna().astype(str).str.strip().tolist()
            ))
            st.write(f"File name: {dpp_workbook.name}")
            c1, c2 = st.columns(2)
            c1.metric("Product batches detected", len(batch_preview))
            c2.metric("Traceability events detected", len(event_preview))
            st.caption(
                "Referenced scenarios: " + (", ".join(referenced_scenarios) or "none")
                + " · Referenced facilities: " + (", ".join(referenced_facilities) or "none")
            )
            batch_columns = [
                "batch_code", "product_code", "scenario_code", "origin_facility_code",
                "production_date", "quantity", "unit", "status",
            ]
            event_columns = [
                "event_code", "batch_code", "event_type", "timestamp", "facility_code",
                "process_code", "transport_leg_code", "quantity", "unit",
            ]
            tab_batches, tab_events, tab_issues = st.tabs(
                ["Product Batches", "Traceability Events", "Validation Issues"]
            )
            with tab_batches:
                batch_display = batch_preview[
                    [c for c in batch_columns if c in batch_preview]
                ]
                render_downloadable_table(
                    batch_display, filename="dpp_batch_preview.csv",
                    key="download_dpp_batch_preview",
                )
            with tab_events:
                event_display = event_preview[
                    [c for c in event_columns if c in event_preview]
                ]
                render_downloadable_table(
                    event_display, filename="traceability_event_preview.csv",
                    key="download_traceability_event_preview",
                )
            with tab_issues:
                if not _has_active_import_run():
                    st.warning("Import MRV data first so active scenario membership can be validated.")
                else:
                    st.info("Select Validate and Import DPP Data to run full relational validation.")
            if not _has_active_import_run():
                st.info(
                    "The validated DPP workbook will be committed with the MRV "
                    "workbook during Review and Commit."
                )
            elif st.button(
                "Validate and Import DPP Data",
                type="primary",
                key="import_dpp_workbook",
            ):
                dpp_session = SessionLocal()
                try:
                    outcome = import_dpp_workbook(dpp_session, workbook_bytes)
                finally:
                    dpp_session.close()
                run_full_pipeline(
                    debug_missing=False,
                    import_run_id=st.session_state.get("active_import_run_id"),
                )
                load_active_context.clear()
                load_raw_kpi_results.clear()
                load_normalized_results.clear()
                st.session_state["dpp_import_summary"] = outcome.summaries
                st.session_state["dpp_import_message"] = (
                    f"Imported {outcome.product_batches.created} new and "
                    f"{outcome.product_batches.updated} updated batches; "
                    f"{outcome.traceability_events.created} new and "
                    f"{outcome.traceability_events.updated} updated events."
                )
                st.session_state["show_import_page"] = False
                st.rerun()
        except DPPImportValidationError as exc:
            st.error("Validation errors prevented import.")
            issues = pd.DataFrame({"Validation issues": exc.result.errors})
            render_downloadable_table(
                issues, filename="dpp_validation_issues.csv",
                key="download_dpp_validation_issues",
            )
        except (ValueError, OSError) as exc:
            st.error(str(exc))

    render_scenario_completion_page(
        config_dir=Path(__file__).resolve().parent / "config",
        on_commit=lambda result: import_completed_mrv(
            result, dpp_workbook_bytes=workbook_bytes
        ),
    )
    st.stop()

dpp_import_message = st.session_state.pop("dpp_import_message", None)
if dpp_import_message:
    st.success(dpp_import_message)

with engine.connect() as connection:
    active_run_row = connection.execute(text(
        "SELECT id, dataset_name, source_filename, import_timestamp, scenario_count, "
        "measurement_count, reference_scenario_code, factor_set_id, last_kpi_calculation "
        "FROM sc_import_run WHERE is_active = 1 "
        "ORDER BY import_timestamp DESC, id DESC LIMIT 1"
    )).mappings().first()
    if active_run_row is None:
        st.warning("No active imported dataset. Go to Data Import to create one.")
        st.stop()
    active_import_run_id = int(active_run_row["id"])
    st.session_state["active_import_run_id"] = active_import_run_id
    active_context = load_active_context(active_import_run_id)
    data_status = {
        "scenarios": active_context["scenario_count"],
        "measurements": active_context["measurement_count"],
        "batches": active_context["batch_count"],
        "events": active_context["event_count"],
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
        "Active dataset": active_run_row["dataset_name"],
        "Import/run ID": active_import_run_id,
        "Source file": active_run_row["source_filename"] or "upload",
        "Import timestamp": active_run_row["import_timestamp"],
        "Scenario count": data_status["scenarios"],
        "Measurement count": data_status["measurements"],
        "Product batches": data_status["batches"],
        "Traceability events": data_status["events"],
        "DPP-ready batches": active_context["dpp_ready_count"],
        "Common variables/scenario": active_context["common_variable_count"],
        "Dataset integrity": active_context["integrity"],
        "Reference scenario": active_run_row["reference_scenario_code"] or "not set",
        "Factor-set ID": active_run_row["factor_set_id"] or "default",
        "Last KPI calculation": active_run_row["last_kpi_calculation"] or "pending",
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
raw_df = load_raw_kpi_results(active_import_run_id)
norm_df = load_normalized_results(active_import_run_id)
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
mcda_eligibility = evaluate_scenario_eligibility(
    raw_latest,
    norm_latest,
    catalog_df[["kpi_code", "dimension"]],
)

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

display_table_df = filtered_table_df[show_cols].copy()
display_table_df["baseline_value"] = display_table_df.apply(
    lambda row: format_reference_value(
        row.get("baseline_value"),
        row.get("normalization_method"),
    ),
    axis=1,
)
display_table_df["semaforo"] = display_table_df["semaforo"].map(
    format_traffic_light_status
)
render_downloadable_table(
    filtered_table_df[show_cols],
    filename=f"{sel_scenario}_detailed_kpis.csv",
    key="download_detailed_kpis",
    display_data=display_table_df,
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

    complete_profiles = dim_show.dropna(
        subset=["environmental", "economic", "social", "technological"]
    ).copy()
    incomplete_profiles = dim_show[
        ~dim_show["scenario_code"].isin(complete_profiles["scenario_code"])
    ]
    if not incomplete_profiles.empty:
        st.warning(
            f"{len(incomplete_profiles)} scenarios excluded because their "
            "four-dimensional profile is incomplete."
        )
    profile_long = complete_profiles.melt(
        id_vars="scenario_code",
        value_vars=list(DIMENSION_ORDER),
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
    ranking_fig = build_horizontal_ranking_chart(
        dim_show,
        scenario_col="scenario_code",
        score_col="SUSTAIN_INDEX_GEOM",
        title="Geometric sustainability index ranking",
        x_title="Index score",
        color="#087F78",
        decimals=2,
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
        "WSM and TOPSIS use the same validated, complete 30-KPI scenario matrix. "
        "Normalized scores are already benefit-oriented."
    )

    mcda_candidates = [s for s in scenario_options if s != reference_scenario]
    if not mcda_candidates:
        st.info(
            "MCDA ranking is not available for a single scenario. "
            "Import at least one additional scenario to perform this comparative analysis."
        )
        render_dpp_section()
        st.stop()

    default_mcda = mcda_candidates
    mcda_scenarios = st.multiselect(
        "Scenarios for MCDA ranking",
        options=mcda_candidates,
        default=[s for s in default_mcda if s in scenario_options],
        key="mcda_scenarios"
    )

    global_weights = build_global_kpi_weights(rules_df, dim_weights)
    weight_series = (
        global_weights.drop_duplicates("kpi_code")
        .set_index("kpi_code")["global_weight"]
        .reindex(catalog_df["kpi_code"])
    )
    mcda_input = build_mcda_input(
        full_dashboard_df,
        weight_series,
        mcda_eligibility,
        mcda_scenarios,
        reference_scenario_code=reference_scenario,
    )
    mcda_result = calculate_mcda(mcda_input, mcda_eligibility)
    mcda_df = pd.merge(
        mcda_result.wsm,
        mcda_result.topsis,
        on="scenario_code",
        how="inner",
        validate="one_to_one",
    )
    excluded_count = len(mcda_input.excluded_scenarios)
    st.caption(f"Excluded from MCDA: {excluded_count} scenario(s).")
    with st.expander("MCDA data completeness"):
        diagnostic_columns = [
            "scenario_code", "raw_kpi_count", "normalized_kpi_count",
            "environmental_count", "economic_count", "social_count",
            "technological_count", "wsm_eligible", "topsis_eligible",
            "status", "reason",
        ]
        render_downloadable_table(
            mcda_eligibility[diagnostic_columns],
            filename="mcda_data_completeness.csv",
            key="download_mcda_completeness",
        )
        removed = mcda_result.diagnostics.get("zero_variance_criteria", [])
        if removed:
            st.caption(
                "TOPSIS removed non-discriminating zero-variance criteria and "
                f"renormalized their weights: {', '.join(removed)}"
            )
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
            wsm_fig = build_horizontal_ranking_chart(
                mcda_df,
                scenario_col="scenario_code",
                score_col="WSM_score",
                title="Weighted-sum ranking",
                x_title="WSM score",
                color="#2F6B9A",
                decimals=3,
            )
            st.plotly_chart(wsm_fig, width="stretch", config={"displaylogo": False})

        if "TOPSIS_score" in mcda_df.columns:
            st.write("**TOPSIS ranking**")
            topsis_fig = build_horizontal_ranking_chart(
                mcda_df,
                scenario_col="scenario_code",
                score_col="TOPSIS_score",
                title="TOPSIS closeness ranking",
                x_title="TOPSIS closeness",
                color="#6657A6",
                decimals=3,
            )
            st.plotly_chart(topsis_fig, width="stretch", config={"displaylogo": False})
    else:
        st.info("MCDA ranking could not be computed with the current scenario selection.")

render_dpp_section()
