# ...existing code...
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# En Streamlit Cloud existe /mount/src
if os.path.exists("/mount/src"):
    os.environ["SUSTAINSC_DB_URL"] = "sqlite:////tmp/sustainsc.db"

import pandas as pd
import streamlit as st
from sqlalchemy import text

# ============================================================================
# 0) Set DB URL BEFORE importing sustainsc modules
# ============================================================================

def _default_db_url() -> str:
    """Get appropriate DB URL for environment (local vs Cloud)"""
    if os.getenv("SUSTAINSC_DB_URL"):
        return os.environ["SUSTAINSC_DB_URL"]

    # Streamlit Community Cloud
    if Path("/mount/src").exists() or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true":
        return "sqlite:////tmp/sustainsc.db"

    # Local development (Windows/Mac/Linux)
    db_path = Path(tempfile.gettempdir()) / "sustainsc.db"
    return f"sqlite:///{db_path.as_posix()}"

os.environ.setdefault("SUSTAINSC_DB_URL", _default_db_url())

# ============================================================================
# 1) Bootstrap everything (imports inside to ensure DB_URL is set)
# ============================================================================

@st.cache_resource
def bootstrap_everything():
    """
    Bootstrap pipeline:
    1) Create schema (incluye sc_emission_factor, sc_kpi, etc.)
    2) Load example data (BASE/S1/S2 + measurements) → upsert idempotente
    3) Create MILP demo scenarios → upsert idempotente
    4) Recompute KPIs
    """
    from sustainsc.config import engine
    from sustainsc.models import Base
    from load_example_data import main as load_example_data_main
    from sustainsc.milp_interface import register_demo_milp_scenarios
    from sustainsc.kpi_engine import run_engine

    try:
        # 1) Asegura esquema completo
        with st.spinner("Creating database schema..."):
            Base.metadata.create_all(bind=engine)

        # 2) Carga CSVs (loader es upsert → idempotente)
        with st.spinner("Loading example data (BASE/S1/S2 + measurements)..."):
            load_example_data_main()

        # 3) Crea escenarios MILP (si existen, actualiza sin duplicar)
        with st.spinner("Registering demo MILP scenarios..."):
            register_demo_milp_scenarios()

        # 4) Recalcula KPIs
        with st.spinner("Computing KPI results..."):
            run_engine()

        return True

    except Exception as e:
        st.error(f"Bootstrap failed: {e}")
        return False

# ============================================================================
# 2) Data utilities & helpers
# ============================================================================

def _table_exists(con, table_name: str) -> bool:
    """Check if table exists in SQLite"""
    r = con.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
        {"t": table_name},
    ).fetchone()
    return r is not None

def _count_rows(con, table_name: str) -> int:
    """Count rows in table safely"""
    if not _table_exists(con, table_name):
        return 0
    try:
        return int(con.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)
    except Exception:
        return 0

def _pct_delta(base, other):
    """Calculate percentage change"""
    try:
        if base is None or pd.isna(base) or float(base) == 0:
            return None
        return (float(other) - float(base)) / float(base) * 100.0
    except Exception:
        return None

def _effect_label(delta, is_benefit):
    """Label effect direction"""
    if delta is None or pd.isna(delta):
        return "Missing"
    is_benefit = int(is_benefit) if not pd.isna(is_benefit) else 0
    if float(delta) == 0:
        return "Same"
    if is_benefit == 1:
        return "Improved" if float(delta) > 0 else "Worse"
    else:
        return "Improved" if float(delta) < 0 else "Worse"

def _get_table_columns(table_name: str, engine) -> list[str]:
    """Get column names from table"""
    with engine.connect() as con:
        if not _table_exists(con, table_name):
            return []
        rows = con.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        return [r[1] for r in rows]

@st.cache_data(ttl=60)
def load_kpi_data(engine):
    """Load KPI metadata + results + scenarios (cached 60s)"""
    kpi_cols = _get_table_columns("sc_kpi", engine)
    has_flow = "flow" in kpi_cols

    kpi_sql = (
        "SELECT id as kpi_id, code as kpi_code, name as kpi_name, "
        "dimension, decision_level, "
        + ("flow, " if has_flow else "'n/a' as flow, ")
        + "unit, is_benefit "
        "FROM sc_kpi ORDER BY code"
    )

    res_sql = (
        "SELECT id as result_id, kpi_id, scenario_id, period_end, value "
        "FROM sc_kpi_result ORDER BY period_end"
    )

    sc_sql = (
        "SELECT id as scenario_id, code as scenario_code, name as scenario_name "
        "FROM sc_scenario ORDER BY id"
    )

    with engine.connect() as con:
        kpi_df = pd.read_sql_query(text(kpi_sql), con)
        res_df = pd.read_sql_query(text(res_sql), con)
        sc_df = pd.read_sql_query(text(sc_sql), con)

    df = res_df.merge(kpi_df, on="kpi_id", how="left").merge(sc_df, on="scenario_id", how="left")
    df["scenario_code"] = df["scenario_code"].fillna("NONE")
    df["scenario_name"] = df["scenario_name"].fillna("NoScenario")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df, kpi_df, sc_df

def latest_per_kpi_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Get latest result per (KPI, scenario) pair"""
    df2 = df.dropna(subset=["kpi_code"]).sort_values("period_end")
    return df2.groupby(["scenario_code", "kpi_code"], as_index=False).tail(1)

@st.cache_data(ttl=60)
def load_vsm_measurements(engine):
    """Load VSM-C measurements (cached 60s)"""
    q = """
    SELECT m.variable_name, m.value, m.unit, m.timestamp, s.code as scenario_code
    FROM sc_measurement m
    LEFT JOIN sc_scenario s ON s.id = m.scenario_id
    WHERE m.variable_name LIKE 'vsm_%'
    """
    with engine.connect() as con:
        dfv = pd.read_sql(q, con)
    dfv["timestamp"] = pd.to_datetime(dfv["timestamp"], errors="coerce")
    dfv["scenario_code"] = dfv["scenario_code"].fillna("NONE")
    return dfv

# ============================================================================
# 3) Streamlit App UI
# ============================================================================

st.set_page_config(page_title="SustainSCM DSS - KPI Dashboard", layout="wide")
st.title("SustainSCM DSS – KPI Dashboard")

# Bootstrap
if not bootstrap_everything():
    st.error("Failed to bootstrap database")
    st.stop()

# Import engine after bootstrap (DB_URL guaranteed to be set)
from sustainsc.config import engine
from sustainsc.vsmc import main as vsmc_main

VSM_CSV = Path(__file__).parent / "data" / "vsm_steps.csv"

# Sidebar: Rebuild button
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Rebuild demo (full)"):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Rebuild failed: {e}")

# Load KPI data
df_all, kpi_meta, sc_meta = load_kpi_data(engine)

if df_all.empty:
    st.warning("⚠️ No KPI results found. Try rebuilding or check logs.")
    st.stop()

latest = latest_per_kpi_scenario(df_all)

# Sidebar: Filters
st.sidebar.header("Filters")
dimensions = ["All"] + sorted(latest["dimension"].dropna().unique().tolist())
decision_levels = ["All"] + sorted(latest["decision_level"].dropna().unique().tolist())
flows = ["All"] + sorted(latest["flow"].dropna().unique().tolist())
scenarios = sorted(latest["scenario_code"].dropna().unique().tolist())

sel_dim = st.sidebar.selectbox("Dimension", dimensions, index=0)
sel_level = st.sidebar.selectbox("Decision level", decision_levels, index=0)
sel_flow = st.sidebar.selectbox("Flow", flows, index=0)
sel_scenario = st.sidebar.selectbox("Scenario (main view)", scenarios, index=0)

# Apply filters
view = latest.copy()
if sel_dim != "All":
    view = view[view["dimension"] == sel_dim]
if sel_level != "All":
    view = view[view["decision_level"] == sel_level]
if sel_flow != "All":
    view = view[view["flow"] == sel_flow]
view = view[view["scenario_code"] == sel_scenario]

# ============================================================================
# Section 1: Latest KPI Values
# ============================================================================

st.subheader(f"Latest KPI values – Scenario: {sel_scenario}")
st.caption("Most recent KPIResult per KPI for selected scenario.")

display_cols = ["kpi_code", "kpi_name", "dimension", "decision_level", "flow", "value", "unit"]
st.dataframe(view[display_cols].sort_values("kpi_code"), use_container_width=True)

# ============================================================================
# Section 2: Time Series Trend
# ============================================================================

st.markdown("### Trend time series (per KPI)")
kpi_list = sorted(view["kpi_code"].unique().tolist())

if not kpi_list:
    st.info("No KPIs available under current filters.")
else:
    sel_kpi = st.selectbox("Select KPI for trend", kpi_list, index=0)
    ts = df_all[
        (df_all["scenario_code"] == sel_scenario) &
        (df_all["kpi_code"] == sel_kpi)
    ].copy()
    ts = ts.sort_values("period_end").dropna(subset=["period_end"])

    if ts.empty:
        st.info("No time series data for selected KPI.")
    else:
        ts_plot = ts[["period_end", "value"]].rename(columns={"period_end": "timestamp"})
        st.line_chart(ts_plot.set_index("timestamp"))

# ============================================================================
# Section 3: VSM-C Diagnostics
# ============================================================================

st.markdown("## VSM-C Diagnostics")
st.caption("Diagnóstico VSM-C (lead time, VA ratio, hotspots) y escenario Kaizen auto-generado.")

# Try to run VSM-C if CSV exists
try:
    if VSM_CSV.exists():
        with st.spinner("Running VSM-C analysis..."):
            vsmc_main(kaizen=True, base_code="BASE", new_code="VSMC_KAIZEN_01")
except Exception as e:
    st.warning(f"VSM-C skipped: {e}")

vsm_df = load_vsm_measurements(engine)

if vsm_df.empty:
    st.info("No VSM-C data found yet.")
else:
    vsm_scenarios = sorted(vsm_df["scenario_code"].unique().tolist())
    default_idx = vsm_scenarios.index(sel_scenario) if sel_scenario in vsm_scenarios else 0
    sel_vsm = st.selectbox("Scenario for VSM-C view", vsm_scenarios, index=default_idx)

    vv = vsm_df[vsm_df["scenario_code"] == sel_vsm].copy()
    if vv["timestamp"].notna().any():
        latest_ts = vv["timestamp"].max()
        vv = vv[vv["timestamp"] == latest_ts]

    def _get(var):
        r = vv[vv["variable_name"] == var]
        return None if r.empty else float(r.iloc[0]["value"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lead time (min)", f"{_get('vsm_total_lead_time_min'):.1f}" if _get("vsm_total_lead_time_min") is not None else "—")
    col2.metric("Cycle time (min)", f"{_get('vsm_total_cycle_time_min'):.1f}" if _get("vsm_total_cycle_time_min") is not None else "—")
    col3.metric("Wait time (min)", f"{_get('vsm_total_wait_time_min'):.1f}" if _get("vsm_total_wait_time_min") is not None else "—")
    col4.metric("VA ratio (%)", f"{_get('vsm_va_ratio_pct'):.1f}" if _get("vsm_va_ratio_pct") is not None else "—")

    col5, col6, col7 = st.columns(3)
    col5.metric("Total VSM emissions (tCO2e)", f"{_get('vsm_total_emissions_tco2e'):.3f}" if _get("vsm_total_emissions_tco2e") is not None else "—")
    col6.metric("Emissions intensity (kgCO2e/ton)", f"{_get('vsm_emissions_intensity_kg_per_ton'):.2f}" if _get("vsm_emissions_intensity_kg_per_ton") is not None else "—")
    col7.metric("Total VSM energy cost (EUR)", f"{_get('vsm_total_cost_eur'):.0f}" if _get("vsm_total_cost_eur") is not None else "—")

    steps = vv[vv["variable_name"].str.startswith("vsm_step_emissions_tco2e::", na=False)].copy()
    if not steps.empty:
        steps["step_code"] = steps["variable_name"].str.split("::").str[1].fillna("UNKNOWN")
        steps = steps[["step_code", "value"]].groupby("step_code", as_index=False)["value"].sum()
        steps = steps.sort_values("value", ascending=False)
        st.markdown("### CO₂ hotspots by step (tCO2e)")
        st.bar_chart(steps.set_index("step_code")["value"])

    with st.expander("Show raw VSM-C measurements (vsm_*)"):
        st.dataframe(vv.sort_values("variable_name"), use_container_width=True)

# ============================================================================
# Section 4: Scenario Comparison (All vs BASE)
# ============================================================================

st.markdown("## Scenario Comparison")
st.caption("Compare multiple scenarios. BASE is reference for delta calculation.")

compare_scenarios = st.multiselect(
    "Select scenarios to compare",
    options=scenarios,
    default=scenarios[: min(3, len(scenarios))],
)

cmp = latest.copy()
if sel_dim != "All":
    cmp = cmp[cmp["dimension"] == sel_dim]
if sel_level != "All":
    cmp = cmp[cmp["decision_level"] == sel_level]
if sel_flow != "All":
    cmp = cmp[cmp["flow"] == sel_flow]
cmp = cmp[cmp["scenario_code"].isin(compare_scenarios)]

if cmp.empty:
    st.info("No data available for selected comparison filters.")
else:
    pivot = cmp.pivot_table(
        index=["kpi_code", "kpi_name", "unit", "is_benefit"],
        columns="scenario_code",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Add delta columns vs BASE
    if "BASE" in pivot.columns:
        for sc in compare_scenarios:
            if sc == "BASE" or sc not in pivot.columns:
                continue

            pivot[f"Δ {sc} vs BASE"] = pivot.apply(
                lambda r: (float(r[sc]) - float(r["BASE"]))
                if (pd.notna(r.get(sc)) and pd.notna(r.get("BASE")))
                else None,
                axis=1,
            )
            pivot[f"%Δ {sc} vs BASE"] = pivot.apply(
                lambda r: _pct_delta(r.get("BASE"), r.get(sc)), axis=1
            )
            pivot[f"Effect {sc} vs BASE"] = pivot.apply(
                lambda r: _effect_label(
                    r.get(f"Δ {sc} vs BASE"),
                    r.get("is_benefit")
                ),
                axis=1
            )

    st.subheader("Scenario comparison table")
    st.dataframe(pivot, use_container_width=True)

    # All vs BASE Summary
    if "BASE" in pivot.columns:
        st.markdown("### Summary: All vs BASE")
        
        base_col = "BASE"
        meta_cols = {"kpi_code", "kpi_name", "unit", "is_benefit"}
        derived_prefixes = ("Δ ", "%Δ ", "Effect ")

        scenario_cols = [
            c for c in pivot.columns
            if (c not in meta_cols) and (not str(c).startswith(derived_prefixes))
        ]
        scenario_cols = [c for c in scenario_cols if c in compare_scenarios]

        summary_rows = []
        long_rows = []

        for sc in scenario_cols:
            if sc == base_col:
                continue

            improved = worse = same = missing = 0

            for _, r in pivot.iterrows():
                base_val = r.get(base_col, None)
                sc_val = r.get(sc, None)

                if pd.isna(base_val) or pd.isna(sc_val):
                    missing += 1
                    effect = "Missing"
                    delta = None
                    pct = None
                else:
                    delta = float(sc_val) - float(base_val)
                    pct = _pct_delta(float(base_val), float(sc_val))
                    effect = _effect_label(delta, r.get("is_benefit", 0))

                    if effect == "Improved":
                        improved += 1
                    elif effect == "Worse":
                        worse += 1
                    else:
                        same += 1

                long_rows.append({
                    "scenario": sc,
                    "kpi_code": r["kpi_code"],
                    "kpi_name": r["kpi_name"],
                    "unit": r.get("unit", ""),
                    "BASE": base_val,
                    sc: sc_val,
                    "delta": delta,
                    "pct_delta": pct,
                    "effect": effect,
                })

            total_valid = improved + worse + same
            improved_pct = (improved / total_valid * 100.0) if total_valid > 0 else None

            summary_rows.append({
                "Scenario": sc,
                "Improved": improved,
                "Worse": worse,
                "Same": same,
                "Missing": missing,
                "Improved (%)": improved_pct,
                "Net score": improved - worse,
            })

        df_summary = pd.DataFrame(summary_rows).sort_values(
            ["Net score", "Improved"], ascending=False
        )

        st.dataframe(df_summary, use_container_width=True)

        for _, r in df_summary.iterrows():
            st.write(
                f"**{r['Scenario']}**: ✅ {int(r['Improved'])} | "
                f"❌ {int(r['Worse'])} | ➖ {int(r['Same'])} | ⚠️ {int(r['Missing'])}"
            )

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download summary (CSV)",
                df_summary.to_csv(index=False).encode("utf-8"),
                file_name="all_vs_base_summary.csv",
                mime="text/csv",
            )

        with col2:
            df_long = pd.DataFrame(long_rows)
            st.download_button(
                "📥 Download detailed (CSV)",
                df_long.to_csv(index=False).encode("utf-8"),
                file_name="all_vs_base_kpi_detail.csv",
                mime="text/csv",
            )