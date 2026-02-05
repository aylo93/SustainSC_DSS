# kpi_dashboard.py
# SustainSC DSS – Streamlit KPI Dashboard (Cloud-proof)
# Features: filters + time series + contribution + alerts + scenario comparison + All vs BASE summary

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st
from sqlalchemy import text

# ------------------------------------------------------------
# 0) IMPORTANT: set DB URL BEFORE importing sustainsc.config
# ------------------------------------------------------------

def _default_db_url() -> str:
    # If user already defined it, respect it
    if os.getenv("SUSTAINSC_DB_URL"):
        return os.environ["SUSTAINSC_DB_URL"]

    # Streamlit Community Cloud typically runs your repo under /mount/src
    if Path("/mount/src").exists() or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true":
        return "sqlite:////tmp/sustainsc.db"  # writable on Cloud

    # Local default (Windows/Mac/Linux) in project folder
    return "sqlite:///sustainsc.db"

os.environ.setdefault("SUSTAINSC_DB_URL", _default_db_url())

# Now import your project modules (they will read the env var above)
from sustainsc.config import engine
from sustainsc.models import Base
from load_example_data import main as load_example_data_main
from sustainsc.kpi_engine import run_engine


# ------------------------------------------------------------
# 1) Bootstrap DB once (Cloud-safe, no subprocess)
# ------------------------------------------------------------

def _table_exists(con, table_name: str) -> bool:
    r = con.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
        {"t": table_name},
    ).fetchone()
    return r is not None

def _count_rows(con, table_name: str) -> int:
    if not _table_exists(con, table_name):
        return 0
    try:
        return int(con.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)
    except Exception:
        return 0

@st.cache_resource
def bootstrap_db() -> bool:
    """
    Creates schema + loads CSV example data + computes KPI results if needed.
    Cache resource prevents re-bootstrapping on every Streamlit rerun.
    """
    # 1) Create tables
    Base.metadata.create_all(bind=engine)

    with engine.connect() as con:
        kpi_count = _count_rows(con, "sc_kpi")
        meas_count = _count_rows(con, "sc_measurement")
        res_count = _count_rows(con, "sc_kpi_result")

    # 2) Load data if missing
    if kpi_count == 0 or meas_count == 0:
        load_example_data_main()

    # 3) Compute KPI results if missing
    with engine.connect() as con:
        res_count = _count_rows(con, "sc_kpi_result")
    if res_count == 0:
        run_engine()

    return True


# ------------------------------------------------------------
# 2) Data utilities
# ------------------------------------------------------------

def _pct_delta(base, other):
    try:
        if base is None or pd.isna(base) or float(base) == 0:
            return None
        return (float(other) - float(base)) / float(base) * 100.0
    except Exception:
        return None

def _effect_label(delta, is_benefit):
    """
    is_benefit = 1 => higher is better
    is_benefit = 0 => lower is better
    """
    if delta is None or pd.isna(delta):
        return "Missing"
    is_benefit = int(is_benefit) if not pd.isna(is_benefit) else 0
    if float(delta) == 0:
        return "Same"
    if is_benefit == 1:
        return "Improved" if float(delta) > 0 else "Worse"
    return "Improved" if float(delta) < 0 else "Worse"

def _get_table_columns(table_name: str) -> list[str]:
    with engine.connect() as con:
        if not _table_exists(con, table_name):
            return []
        rows = con.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        return [r[1] for r in rows]

@st.cache_data(ttl=30)
def load_kpi_data():
    """
    Load KPI metadata + KPI results + scenario metadata, robust to missing 'flow' column.
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

    df = (
        res_df.merge(kpi_df, on="kpi_id", how="left")
              .merge(sc_df, on="scenario_id", how="left")
    )

    df["scenario_code"] = df["scenario_code"].fillna("NONE")
    df["scenario_name"] = df["scenario_name"].fillna("NoScenario")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    return df, kpi_df, sc_df

def latest_per_kpi_scenario(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(subset=["kpi_code"]).sort_values("period_end")
    return df2.groupby(["scenario_code", "kpi_code"], as_index=False).tail(1)


# ------------------------------------------------------------
# 3) App
# ------------------------------------------------------------

st.set_page_config(page_title="SustainSCM DSS - KPI Dashboard", layout="wide")
st.title("SustainSCM DSS – KPI Dashboard (Prototype)")

bootstrap_db()

df_all, kpi_meta, sc_meta = load_kpi_data()

if df_all.empty:
    st.warning("No KPI results found. Check that load_example_data and kpi_engine ran correctly.")
    st.stop()

latest = latest_per_kpi_scenario(df_all)

# Sidebar filters
st.sidebar.header("Filters")
dimensions = ["All"] + sorted(latest["dimension"].dropna().unique().tolist())
decision_levels = ["All"] + sorted(latest["decision_level"].dropna().unique().tolist())
flows = ["All"] + sorted(latest["flow"].dropna().unique().tolist())
scenarios = sorted(latest["scenario_code"].dropna().unique().tolist())

sel_dim = st.sidebar.selectbox("Dimension", dimensions, index=0)
sel_level = st.sidebar.selectbox("Decision level", decision_levels, index=0)
sel_flow = st.sidebar.selectbox("Flow", flows, index=0)
sel_scenario = st.sidebar.selectbox("Scenario (main view)", scenarios, index=0)

view = latest.copy()
if sel_dim != "All":
    view = view[view["dimension"] == sel_dim]
if sel_level != "All":
    view = view[view["decision_level"] == sel_level]
if sel_flow != "All":
    view = view[view["flow"] == sel_flow]
view = view[view["scenario_code"] == sel_scenario]

st.subheader(f"Latest KPI values – Scenario: {sel_scenario}")
st.caption("This table shows the most recent KPIResult per KPI for the selected scenario.")

display_cols = ["kpi_code", "kpi_name", "dimension", "decision_level", "flow", "value", "unit"]
st.dataframe(view[display_cols].sort_values("kpi_code"), width="stretch")

# ------------------------------------------------------------
# Trend time series
# ------------------------------------------------------------
st.markdown("### Trend time series (per KPI)")
kpi_list = sorted(view["kpi_code"].unique().tolist())
if not kpi_list:
    st.info("No KPIs available under current filters.")
else:
    sel_kpi = st.selectbox("Select KPI for trend", kpi_list, index=0)
    ts = df_all[(df_all["scenario_code"] == sel_scenario) & (df_all["kpi_code"] == sel_kpi)].copy()
    ts = ts.sort_values("period_end").dropna(subset=["period_end"])
    if ts.empty:
        st.info("No time series data for selected KPI.")
    else:
        ts_plot = ts[["period_end", "value"]].rename(columns={"period_end": "timestamp"})
        st.line_chart(ts_plot.set_index("timestamp"))

# ------------------------------------------------------------
# Contribution analysis (demo)
# ------------------------------------------------------------
st.markdown("### Contribution analysis (demo) – Emissions & Energy drivers")
st.caption("This uses MRV measurements to show contributions of key drivers. Expand variables as needed.")

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
        st.info("No scenario_id found for contribution analysis.")
    else:
        meas_sc = meas[meas["scenario_id"] == scenario_id].copy()
        contrib_vars = ["electricity_kwh", "diesel_kwh", "renewable_energy_kwh"]
        contrib = (
            meas_sc[meas_sc["variable_name"].isin(contrib_vars)]
            .groupby("variable_name", as_index=False)["value"].sum()
            .sort_values("value", ascending=False)
        )
        if contrib.empty:
            st.info("No MRV measurements found for contribution variables.")
        else:
            st.bar_chart(contrib.set_index("variable_name")["value"])
else:
    st.info("No measurements available for contribution analysis.")

# ------------------------------------------------------------
# Alerts & thresholds
# ------------------------------------------------------------
st.markdown("### Alerts & thresholds (simple)")
st.caption("Set thresholds for key KPIs; flags show potential issues (prototype).")

colA, colB = st.columns(2)
with colA:
    thresh_e1 = st.number_input("Threshold for E1 (tCO2e) – alert if above", value=999999.0)
with colB:
    thresh_ec1 = st.number_input("Threshold for EC1 (EUR/FU) – alert if above", value=999999.0)

alerts_view = latest[latest["scenario_code"] == sel_scenario].set_index("kpi_code")
msgs = []
if "E1" in alerts_view.index and pd.notna(alerts_view.loc["E1", "value"]) and float(alerts_view.loc["E1", "value"]) > thresh_e1:
    msgs.append(f"⚠️ E1 above threshold: {alerts_view.loc['E1','value']} > {thresh_e1}")
if "EC1" in alerts_view.index and pd.notna(alerts_view.loc["EC1", "value"]) and float(alerts_view.loc["EC1", "value"]) > thresh_ec1:
    msgs.append(f"⚠️ EC1 above threshold: {alerts_view.loc['EC1','value']} > {thresh_ec1}")

if msgs:
    for m in msgs:
        st.warning(m)
else:
    st.success("No alerts triggered under current thresholds.")

# ------------------------------------------------------------
# Scenario comparison
# ------------------------------------------------------------
st.markdown("## Scenario comparison")
st.caption("Compare multiple scenarios using latest KPI results. BASE is used as reference when available.")
st.caption("Tip: si quieres también %Δ, ya está incluido cuando BASE está presente (muy útil para tesis).")

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
    st.info("No data available for the selected scenario comparison filters.")
else:
    pivot = cmp.pivot_table(
        index=["kpi_code", "kpi_name", "unit", "is_benefit"],
        columns="scenario_code",
        values="value",
        aggfunc="first",
    ).reset_index()

    if "BASE" in pivot.columns:
        for sc in compare_scenarios:
            if sc == "BASE" or sc not in pivot.columns:
                continue
            pivot[f"Δ {sc} vs BASE"] = pivot.apply(
                lambda r: (r[sc] - r["BASE"]) if (pd.notna(r.get(sc)) and pd.notna(r.get("BASE"))) else None,
                axis=1,
            )
            pivot[f"%Δ {sc} vs BASE"] = pivot.apply(lambda r: _pct_delta(r.get("BASE"), r.get(sc)), axis=1)
            pivot[f"Effect {sc} vs BASE"] = pivot.apply(
                lambda r: _effect_label(r.get(f"Δ {sc} vs BASE"), r.get("is_benefit")), axis=1
            )

    st.subheader("Scenario comparison table (latest values)")
    st.dataframe(pivot, width="stretch")

    # --------------------------------------------------------
    # All vs BASE summary
    # --------------------------------------------------------
    st.markdown("### All vs BASE summary (Improved / Worse / Same / Missing)")
    if "BASE" not in pivot.columns:
        st.warning("BASE is not available in the comparison table. Add BASE scenario results first.")
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

                long_rows.append(
                    {
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
                    }
                )

            total_valid = improved + worse + same
            improved_pct = (improved / total_valid * 100.0) if total_valid > 0 else None

            summary_rows.append(
                {
                    "Scenario": sc,
                    "Improved (count)": improved,
                    "Worse (count)": worse,
                    "Same (count)": same,
                    "Missing (count)": missing,
                    "Improved (%)": improved_pct,
                    "Net score (Improved - Worse)": improved - worse,
                }
            )

        df_summary = pd.DataFrame(summary_rows).sort_values(
            ["Net score (Improved - Worse)", "Improved (count)"], ascending=False
        )
        st.dataframe(df_summary, width="stretch")

        for _, r in df_summary.iterrows():
            st.write(
                f"**{r['Scenario']}**: Improved **{int(r['Improved (count)'])}** / "
                f"Worse **{int(r['Worse (count)'])}** / Same **{int(r['Same (count)'])}** "
                f"(Missing **{int(r['Missing (count)'])}**)"
            )

        st.download_button(
            "Download All vs BASE summary (CSV)",
            df_summary.to_csv(index=False).encode("utf-8"),
            file_name="all_vs_base_summary.csv",
            mime="text/csv",
        )

        df_long = pd.DataFrame(long_rows)
        with st.expander("Show detailed KPI-by-KPI effects (All vs BASE)"):
            df_long2 = df_long.copy()
            df_long2["effect_order"] = (
                df_long2["effect"].map({"Improved": 0, "Worse": 1, "Same": 2, "Missing": 3}).fillna(9)
            )
            df_long2 = df_long2.sort_values(["scenario", "effect_order", "kpi_code"])
            st.dataframe(
                df_long2[["scenario", "kpi_code", "kpi_name", "is_benefit", "BASE", "delta", "pct_delta", "effect"]],
                width="stretch"
            )
            st.download_button(
                "Download KPI-by-KPI comparison (CSV)",
                df_long2.to_csv(index=False).encode("utf-8"),
                file_name="all_vs_base_kpi_detail.csv",
                mime="text/csv",
            )
