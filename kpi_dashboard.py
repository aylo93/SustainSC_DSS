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
# 0) Imports from sustainsc (antes de DB_URL setup para evitar duplicados)
# ============================================================================

from sustainsc.anylogistix_adapter import import_anylogistix_csv
from sustainsc.sim_export_adapter import read_any_file, normalize_any_export, upsert_measurements
from sustainsc.kpi_engine import run_engine

# ============================================================================
# 1) Set DB URL BEFORE importing other sustainsc modules
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
# 2) Bootstrap everything (imports inside to ensure DB_URL is set)
# ============================================================================

@st.cache_resource
def bootstrap_everything():
    """
    Bootstrap pipeline:
    1) Create schema
    2) Load example data (BASE/S1/S2 + measurements)
    3) Create MILP demo scenarios
    4) Create VSM-C demo scenarios (optional)
    5) Recompute KPIs
    """
    from sustainsc.config import engine
    from sustainsc.models import Base
    
    try:
        from sustainsc.data_loader import main as load_example_data_main
    except ModuleNotFoundError:
        from load_example_data import main as load_example_data_main

    try:
        # 1) Asegura esquema completo
        with st.spinner("Creating database schema..."):
            Base.metadata.create_all(bind=engine)

        # 2) Carga CSVs
        with st.spinner("Loading example data (BASE/S1/S2 + measurements)..."):
            load_example_data_main()

        # 3) Crea escenarios MILP
        with st.spinner("Registering demo MILP scenarios..."):
            try:
                from sustainsc.milp_interface import register_demo_milp_scenarios
                register_demo_milp_scenarios()
            except Exception as e:
                st.warning(f"⚠️ MILP registration skipped: {type(e).__name__}")

        # 4) Crea escenarios VSM-C (OPTIONAL - no falla si no existe)
        with st.spinner("Registering demo VSM-C scenarios..."):
            try:
                from sustainsc.vsmc import register_demo_vsmc_scenarios
                register_demo_vsmc_scenarios()
            except ImportError:
                # Module doesn't exist or import error - skip silently
                pass
            except FileNotFoundError:
                # vsm_steps.csv doesn't exist - skip silently
                pass
            except Exception as e:
                # Other errors - log but don't fail
                print(f"[INFO] VSM-C skipped: {type(e).__name__}: {e}")

        # 5) Recalcula KPIs
        with st.spinner("Computing KPI results..."):
            run_engine()

        return True

    except Exception as e:
        st.error(f"❌ Bootstrap failed: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

# ============================================================================
# 3) Data utilities & helpers
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

@st.cache_data(ttl=5)
def load_kpi_data():
    """Load KPI metadata + results + scenarios (cached 5s)"""
    from sustainsc.config import engine
    
    kpi_df = pd.read_sql(
        "SELECT id as kpi_id, code as kpi_code, name as kpi_name, dimension, decision_level, flow, unit, is_benefit "
        "FROM sc_kpi ORDER BY code",
        engine
    )

    res_df = pd.read_sql(
        "SELECT id as result_id, kpi_id, scenario_id, period_end, value "
        "FROM sc_kpi_result ORDER BY period_end",
        engine
    )

    sc_df = pd.read_sql(
        "SELECT id as scenario_id, code as scenario_code, name as scenario_name "
        "FROM sc_scenario ORDER BY id",
        engine
    )

    df = res_df.merge(kpi_df, on="kpi_id", how="left").merge(sc_df, on="scenario_id", how="left")
    df["scenario_code"] = df["scenario_code"].fillna("NONE")
    df["scenario_name"] = df["scenario_name"].fillna("NoScenario")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df, kpi_df, sc_df

def latest_per_kpi_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Get latest result per (KPI, scenario) pair"""
    df2 = df.dropna(subset=["kpi_code"]).sort_values("period_end")
    return df2.groupby(["scenario_code", "kpi_code"], as_index=False).tail(1)

@st.cache_data(ttl=5)
def load_vsm_measurements():
    """Load VSM-C measurements (cached 5s)"""
    from sustainsc.config import engine
    
    q = """
    SELECT m.variable_name, m.value, m.unit, m.timestamp, s.code as scenario_code
    FROM sc_measurement m
    LEFT JOIN sc_scenario s ON s.id = m.scenario_id
    WHERE m.variable_name LIKE 'vsm_%'
    """
    dfv = pd.read_sql(q, engine)
    dfv["timestamp"] = pd.to_datetime(dfv["timestamp"], errors="coerce")
    dfv["scenario_code"] = dfv["scenario_code"].fillna("NONE")
    return dfv

# ============================================================================
# 4) Streamlit App UI
# ============================================================================

st.set_page_config(page_title="SustainSCM DSS - KPI Dashboard", layout="wide")
st.title("SustainSCM DSS – KPI Dashboard")

# Bootstrap
if not bootstrap_everything():
    st.error("❌ Failed to bootstrap database")
    st.stop()

st.success("✅ Database initialized")

# Import engine after bootstrap
from sustainsc.config import engine, SessionLocal
try:
    from sustainsc.vsmc import main as vsmc_main
except ImportError:
    vsmc_main = None

VSM_CSV = Path(__file__).parent / "data" / "vsm_steps.csv"

# ============================================================================
# Sidebar: Controls & Filters
# ============================================================================

st.sidebar.header("Controls")

if st.sidebar.button("🔄 Rebuild demo (full)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ============================================================================
# AnyLogistix/AnyLogic upload (Cloud-friendly)
# ============================================================================
with st.sidebar.expander("📥 Import AnyLogistix results", expanded=False):
    st.write("Upload a CSV (long format recommended):")
    st.caption("Required columns: scenario_code, variable_name, value. "
               "Optional: timestamp, unit, source_system, comment.")

    up_alx = st.file_uploader("CSV file", type=["csv"], key="anylogistix_uploader")

    prefix = st.text_input("Scenario prefix (optional)", value="ALX_", key="alx_prefix")
    recalc_alx = st.checkbox("Recalculate KPIs after import", value=True, key="alx_recalc")

    if up_alx is not None:
        # Show quick preview
        try:
            df_preview = pd.read_csv(up_alx)
            st.write("Preview (first 10 rows):")
            st.dataframe(df_preview.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not preview CSV: {e}")

        if st.button("Import into DB", type="primary", key="alx_import_btn"):
            try:
                # 1) write uploaded file to a temp path (works in Streamlit Cloud)
                suffix = ".csv"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", newline="") as tmp:
                    tmp.write(up_alx.getvalue().decode("utf-8"))
                    tmp_path = Path(tmp.name)

                # 2) import -> creates/updates scenarios + measurements
                with st.spinner("Importing AnyLogistix data..."):
                    stats = import_anylogistix_csv(tmp_path, scenario_prefix=prefix)

                # 3) recalc KPIs so dashboard can see new scenario results
                if recalc_alx:
                    with st.spinner("Recalculating KPIs..."):
                        run_engine()

                # 4) clear Streamlit caches so new data is reloaded
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass

                st.success(
                    f"✅ Imported OK | Scenarios: {stats.scenarios_touched} "
                    f"| Measurements: {stats.measurements_written}"
                )

                # 5) rerun app to refresh filters/scenario list
                st.rerun()

            except FileNotFoundError as e:
                st.error(f"❌ File error: {e}")
            except Exception as e:
                st.error(f"❌ Import failed")
                st.exception(e)

# ============================================================================
# Simulation Export Import (AnyLogistix/AnyLogic)
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Import Simulation Export")

uploaded = st.sidebar.file_uploader(
    "Upload .xlsx or .csv (exported results)",
    type=["xlsx", "xls", "csv"],
    key="sim_export_uploader"
)

default_code = st.sidebar.text_input(
    "Default scenario_code (if file doesn't include one)",
    value="SIM_ALX",
    key="sim_default_code"
)

source_label = st.sidebar.text_input(
    "source_system label",
    value="AnyLogistix/AnyLogic",
    key="sim_source_label"
)

if uploaded is not None:
    if st.sidebar.button("Import → write MRV → recompute KPIs", key="sim_import_btn"):
        try:
            # 1) save upload to temp file
            suffix = "." + uploaded.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)

            # 2) read + normalize → MRV format
            with st.spinner("Reading file..."):
                df_raw = read_any_file(tmp_path)
            
            with st.spinner("Normalizing to MRV format..."):
                df_mrv = normalize_any_export(
                    df_raw,
                    default_scenario_code=default_code,
                    source_system=source_label,
                )

            # 3) write to DB + recompute KPIs
            with st.spinner("Writing measurements to database..."):
                session = SessionLocal()
                try:
                    written = upsert_measurements(session, df_mrv)
                finally:
                    session.close()

            with st.spinner("Recalculating KPIs..."):
                run_engine()

            # 4) refresh caches so new scenarios appear immediately
            try:
                st.cache_data.clear()
            except Exception:
                pass
            try:
                st.cache_resource.clear()
            except Exception:
                pass

            st.sidebar.success(f"✅ Imported {written} measurements. KPIs recalculated.")
            st.rerun()

        except FileNotFoundError as e:
            st.sidebar.error(f"❌ File not found: {e}")
        except Exception as e:
            st.sidebar.error(f"❌ Import failed")
            st.sidebar.exception(e)

st.sidebar.header("Filters")

# ============================================================================
# Load KPI data
# ============================================================================

try:
    df_all, kpi_meta, sc_meta = load_kpi_data()
except Exception as e:
    st.error(f"❌ Failed to load KPI data: {e}")
    st.stop()

if df_all.empty:
    st.warning("⚠️ No KPI results found. Try rebuilding or check logs.")
    st.stop()

latest = latest_per_kpi_scenario(df_all)

# Sidebar filters
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
if vsmc_main and VSM_CSV.exists():
    try:
        with st.spinner("Running VSM-C analysis..."):
            vsmc_main(kaizen=True, base_code="BASE", new_code="VSMC_KAIZEN_01")
    except Exception as e:
        st.warning(f"⚠️ VSM-C skipped: {e}")

vsm_df = load_vsm_measurements()

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
        st.markdown("### CO₂ Hotspots by Step (tCO2e)")
        st.bar_chart(steps.set_index("step_code")["value"])

    with st.expander("📋 Show raw VSM-C measurements (vsm_*)"):
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