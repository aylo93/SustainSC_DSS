# ...existing code...
from __future__ import annotations

import os
import tempfile
from pathlib import Path

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

# Now safe to import project modules
from sustainsc.config import engine
from sustainsc.models import Base
try:
    # Si algún día mueves el loader dentro del paquete sustainsc/
    from sustainsc.data_loader import main as load_example_data_main
except ModuleNotFoundError:
    # Tu caso actual: loader en la raíz del repo
    from load_example_data import main as load_example_data_main

from sustainsc.kpi_engine import run_engine
from sustainsc.vsmc import main as vsmc_main

# VSM-C
VSM_CSV = Path(__file__).parent / "data" / "vsm_steps.csv"

# ============================================================================
# 1) DB helpers + bootstrap (cached)
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

def _count_milp_scenarios(engine) -> int:
    with engine.connect() as con:
        return int(con.execute(text(
            "SELECT COUNT(*) FROM sc_scenario WHERE code LIKE 'MILP_%'"
        )).scalar() or 0)

@st.cache_resource
def bootstrap_db(force: bool = False):
    """
    Initialize database: create schema + load data + compute KPIs (if needed).
    Cached to run only once per session.
    """
    Base.metadata.create_all(bind=engine)

    # 1) cargar datos base si no hay KPIs
    with engine.connect() as con:
        kpi_count = int(con.execute(text("SELECT COUNT(*) FROM sc_kpi")).scalar() or 0)

    if force or kpi_count == 0:
        load_example_data_main()   # BASE/S1/S2 + measurements base
        # IMPORTANTE: no hagas return aquí

    # 2) crear escenarios MILP si no existen
    milp_count = _count_milp_scenarios(engine)
    if force or milp_count == 0:
        from sustainsc.milp_interface import register_demo_milp_scenarios
        register_demo_milp_scenarios()

    # 3) recalcular KPIs (para que aparezcan en el dashboard)
    run_engine()

    return True

# ============================================================================
# 2) Data utilities
# ============================================================================

def _pct_delta(base, other):
    """Calculate percentage change"""
    try:
        if base is None or pd.isna(base) or float(base) == 0:
            return None
        return (float(other) - float(base)) / float(base) * 100.0
    except Exception:
        return None

def _effect_label(delta, is_benefit):
    """
    Label effect direction.
    is_benefit=1: higher is better
    is_benefit=0: lower is better
    """
    if delta is None or pd.isna(delta):
        return "Missing"

    is_benefit = int(is_benefit) if not pd.isna(is_benefit) else 0

    if float(delta) == 0:
        return "Same"

    if is_benefit == 1:
        return "Improved" if float(delta) > 0 else "Worse"
    else:
        return "Improved" if float(delta) < 0 else "Worse"

def _get_table_columns(table_name: str) -> list[str]:
    """Get column names from table"""
    with engine.connect() as con:
        if not _table_exists(con, table_name):
            return []
        rows = con.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        return [r[1] for r in rows]

@st.cache_data(ttl=60)
def load_kpi_data():
    """
    Load KPI metadata + results + scenarios.
    Cached with 60s TTL (refreshes every minute).
    """
    kpi_cols = _get_table_columns("sc_kpi")
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

    # Merge all data
    df = (
        res_df.merge(kpi_df, on="kpi_id", how="left")
              .merge(sc_df, on="scenario_id", how="left")
    )

    df["scenario_code"] = df["scenario_code"].fillna("NONE")
    df["scenario_name"] = df["scenario_name"].fillna("NoScenario")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    return df, kpi_df, sc_df

def latest_per_kpi_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Get latest result per (KPI, scenario) pair"""
    df2 = df.dropna(subset=["kpi_code"]).sort_values("period_end")
    return df2.groupby(["scenario_code", "kpi_code"], as_index=False).tail(1)

# --- VSM-C (optional, only if CSV exists) ---
try:
    if VSM_CSV.exists():
        # Esto escribe diagnósticos vsm_* y crea escenario Kaizen
        vsmc_main(kaizen=True, base_code="BASE", new_code="VSMC_KAIZEN_01")
except Exception as e:
    # No rompemos la app si falta el CSV o hay un problema
    st.warning(f"VSM-C skipped: {e}")

# ============================================================================
# 3) Streamlit App UI
# ============================================================================

st.set_page_config(page_title="SustainSCM DSS - KPI Dashboard", layout="wide")
st.title("SustainSCM DSS – KPI Dashboard (Prototype)")

# Initialize DB
if not bootstrap_db():
    st.error("Failed to bootstrap database")
    st.stop()

# Load data
df_all, kpi_meta, sc_meta = load_kpi_data()

if df_all.empty:
    st.warning(
        "⚠️ No KPI results found. Run: `python -m sustainsc.scripts.setup_demo` first."
    )
    st.stop()

latest = latest_per_kpi_scenario(df_all)

# ============================================================================
# Sidebar: Filters
# ============================================================================

st.sidebar.header("Filters")

# Agregar botón para reconstruir demo (MILP + KPIs)
if st.sidebar.button("🔄 Rebuild demo (MILP + KPIs)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    bootstrap_db(force=True)
    st.rerun()

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
# Section 3: Contribution Analysis (Drivers)
# ============================================================================

st.markdown("### Contribution analysis – Energy & Emissions drivers")
st.caption("Shows contributions of key variables (electricity, diesel, renewable energy).")

with engine.connect() as con:
    if _table_exists(con, "sc_measurement"):
        meas = pd.read_sql_query(
            text("SELECT variable_name, value, unit, timestamp, scenario_id FROM sc_measurement"),
            con
        )
    else:
        meas = pd.DataFrame()

if not meas.empty:
    meas["timestamp"] = pd.to_datetime(meas["timestamp"], errors="coerce")

    sc_map = {r["scenario_code"]: r["scenario_id"] for _, r in sc_meta.iterrows()}
    scenario_id = sc_map.get(sel_scenario)

    if scenario_id is None:
        st.info("No scenario_id found in database.")
    else:
        meas_sc = meas[meas["scenario_id"] == scenario_id].copy()
        contrib_vars = ["electricity_kwh", "diesel_kwh", "renewable_energy_kwh"]
        contrib = (
            meas_sc[meas_sc["variable_name"].isin(contrib_vars)]
            .groupby("variable_name", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
        )

        if contrib.empty:
            st.info("No MRV measurements found for energy variables.")
        else:
            st.bar_chart(contrib.set_index("variable_name")["value"])
else:
    st.info("No measurements available.")

# ============================================================================
# Section 4: Alerts & Thresholds
# ============================================================================

st.markdown("### Alerts & thresholds")
st.caption("Set KPI thresholds to flag potential issues.")

col_e1, col_ec1 = st.columns(2)
with col_e1:
    thresh_e1 = st.number_input(
        "Threshold for E1 (tCO2e) – alert if above",
        value=999999.0,
        step=1000.0
    )
with col_ec1:
    thresh_ec1 = st.number_input(
        "Threshold for EC1 (EUR/FU) – alert if above",
        value=999999.0,
        step=100.0
    )

alerts_view = latest[latest["scenario_code"] == sel_scenario].set_index("kpi_code")
msgs = []

if "E1" in alerts_view.index and pd.notna(alerts_view.loc["E1", "value"]):
    e1_val = float(alerts_view.loc["E1", "value"])
    if e1_val > thresh_e1:
        msgs.append(f"⚠️ E1 above threshold: {e1_val:.2f} tCO2e > {thresh_e1:.2f}")

if "EC1" in alerts_view.index and pd.notna(alerts_view.loc["EC1", "value"]):
    ec1_val = float(alerts_view.loc["EC1", "value"])
    if ec1_val > thresh_ec1:
        msgs.append(f"⚠️ EC1 above threshold: {ec1_val:.2f} EUR/FU > {thresh_ec1:.2f}")

if msgs:
    for m in msgs:
        st.warning(m)
else:
    st.success("✅ No alerts triggered under current thresholds.")

# ============================================================================
# Section 5: Scenario Comparison
# ============================================================================

st.markdown("## Scenario comparison")
st.caption("Compare multiple scenarios using latest KPI results.")
st.caption("📊 BASE is used as reference")

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

    st.subheader("Scenario comparison table (latest values)")
    st.dataframe(pivot, use_container_width=True)

    # All vs BASE Summary
    st.markdown("### All vs BASE summary (Improved / Worse / Same / Missing)")

    if "BASE" not in pivot.columns:
        st.warning("❌ BASE not available in comparison. Add BASE to scenarios.")
    else:
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
                    "is_benefit": r.get("is_benefit", 0),
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
                "Improved (count)": improved,
                "Worse (count)": worse,
                "Same (count)": same,
                "Missing (count)": missing,
                "Improved (%)": improved_pct,
                "Net score": improved - worse,
            })

        df_summary = pd.DataFrame(summary_rows).sort_values(
            ["Net score", "Improved (count)"], ascending=False
        )

        st.dataframe(df_summary, use_container_width=True)

        for _, r in df_summary.iterrows():
            st.write(
                f"**{r['Scenario']}**: "
                f"✅ Improved **{int(r['Improved (count)'])}** | "
                f"❌ Worse **{int(r['Worse (count)'])}** | "
                f"➖ Same **{int(r['Same (count)'])}** | "
                f"⚠️ Missing **{int(r['Missing (count)'])}**"
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
                "📥 Download detailed comparison (CSV)",
                df_long.to_csv(index=False).encode("utf-8"),
                file_name="all_vs_base_kpi_detail.csv",
                mime="text/csv",
            )

        with st.expander("📋 Show KPI-by-KPI detailed effects"):
            df_long = pd.DataFrame(long_rows)
            df_long["effect_order"] = df_long["effect"].map({
                "Improved": 0,
                "Worse": 1,
                "Same": 2,
                "Missing": 3
            }).fillna(9)
            df_long = df_long.sort_values(["scenario", "effect_order", "kpi_code"])

            st.dataframe(
                df_long[[
                    "scenario", "kpi_code", "kpi_name",
                    "BASE", "delta", "pct_delta", "effect"
                ]],
                use_container_width=True
            )

# --- VSM-C (optional, only if CSV exists) ---
try:
    if VSM_CSV.exists():
        # Esto escribe diagnósticos vsm_* y crea escenario Kaizen
        vsmc_main(kaizen=True, base_code="BASE", new_code="VSMC_KAIZEN_01")
except Exception as e:
    # No rompemos la app si falta el CSV o hay un problema
    st.warning(f"VSM-C skipped: {e}")

# --- VSM-C (optional, only if CSV exists) ---
try:
    if VSM_CSV.exists():
        # Esto escribe diagnósticos vsm_* y crea escenario Kaizen
        vsmc_main(kaizen=True, base_code="BASE", new_code="VSMC_KAIZEN_01")
except Exception as e:
    # No rompemos la app si falta el CSV o hay un problema
    st.warning(f"VSM-C skipped: {e}")

# --- VSM-C (optional, only if CSV exists) ---
try:
    if VSM_CSV.exists():
        # Esto escribe diagnósticos vsm_* y crea escenario Kaizen
        vsmc_main(kaizen=True, base_code="BASE", new_code="VSMC_KAIZEN_01")
except Exception as e:
    # No rompemos la app si falta el CSV o hay un problema
    st.warning(f"VSM-C skipped: {e}")