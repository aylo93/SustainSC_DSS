# kpi_dashboard.py
import os
import streamlit as st

# Asegura DB en /tmp en Cloud
os.environ["SUSTAINSC_DB_URL"] = "sqlite:////tmp/sustainsc.db"

from sustainsc.config import SessionLocal
from sustainsc.models import Base
from sustainsc.config import engine

import create_db
import load_example_data
from sustainsc import kpi_engine

def bootstrap_db_once():
    # Evita que Streamlit re-ejecute esto 100 veces al recargar
    if st.session_state.get("_db_bootstrapped", False):
        return

    # 1) Crea tablas
    Base.metadata.create_all(bind=engine)

    # 2) Carga datos CSV (si tu script load_example_data ya lo hace)
    # Ideal: que load_example_data.main() cargue todo: escenarios, factores, kpis, measurements
    load_example_data.main()

    # 3) Calcula KPIs
    kpi_engine.run_engine()

    st.session_state["_db_bootstrapped"] = True

# kpi_dashboard.py
# SustainSC DSS - Streamlit KPI Dashboard (26 KPIs + filters + scenario comparison)
# Requires: streamlit, pandas, sqlalchemy

import os
import streamlit as st

DB_PATH = "sustainsc.db"

def ensure_db_ready():
    if os.path.exists(DB_PATH):
        return

    # Ejecuta todo en el mismo proceso (más robusto en Cloud)
    import create_db
    import load_example_data
    from sustainsc.kpi_engine import run_engine

    create_db.main()
    load_example_data.main()
    run_engine()

ensure_db_ready()


import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = "sqlite:///sustainsc.db"

def _pct_delta(base, other):
    try:
        if base is None or pd.isna(base) or base == 0:
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
    if delta == 0:
        return "Same"
    if is_benefit == 1:
        return "Improved" if delta > 0 else "Worse"
    return "Improved" if delta < 0 else "Worse"

@st.cache_data(ttl=5)
def load_kpi_data():
    engine = create_engine(DB_PATH)

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

def latest_per_kpi_scenario(df):
    df2 = df.dropna(subset=["kpi_code"]).sort_values("period_end")
    return df2.groupby(["scenario_code", "kpi_code"], as_index=False).tail(1)

st.set_page_config(page_title="SustainSCM DSS - KPI Dashboard", layout="wide")
st.title("SustainSCM DSS – KPI Dashboard (Prototype)")
st.set_page_config(page_title="SustainSC DSS", layout="wide")
bootstrap_db_once()

df_all, kpi_meta, sc_meta = load_kpi_data()

if df_all.empty:
    st.warning("No KPI results found. Run: python -m sustainsc.kpi_engine")
    st.stop()

latest = latest_per_kpi_scenario(df_all)

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

st.markdown("### Contribution analysis (demo) – Emissions & Cost drivers")
st.caption("This uses MRV measurements to show contributions of key drivers. Expand variables as needed.")

engine = create_engine(DB_PATH)
meas = pd.read_sql(text("SELECT variable_name, value, unit, timestamp, scenario_id FROM sc_measurement"), engine)
meas["timestamp"] = pd.to_datetime(meas["timestamp"], errors="coerce")

sc_map = {r["scenario_code"]: r["scenario_id"] for _, r in sc_meta.iterrows()}
scenario_id = sc_map.get(sel_scenario, None)

if scenario_id is None:
    st.info("No scenario_id found for contribution analysis.")
else:
    meas_sc = meas[meas["scenario_id"] == scenario_id].copy()
    contrib_vars = ["electricity_kwh", "diesel_kwh", "renewable_energy_kwh"]
    contrib = meas_sc[meas_sc["variable_name"].isin(contrib_vars)].groupby("variable_name", as_index=False)["value"].sum()
    if contrib.empty:
        st.info("No MRV measurements found for contribution analysis variables.")
    else:
        contrib = contrib.sort_values("value", ascending=False)
        st.bar_chart(contrib.set_index("variable_name")["value"])

st.markdown("### Alerts & thresholds (simple)")
st.caption("Set thresholds for key KPIs; flags show potential issues (prototype).")

alert_kpis = ["E1", "E2", "EC1", "S1"]
available_alerts = [k for k in alert_kpis if k in latest["kpi_code"].unique()]
if not available_alerts:
    st.info("Alert KPIs not available in current dataset.")
else:
    colA, colB = st.columns(2)
    with colA:
        thresh_e1 = st.number_input("Threshold for E1 (tCO2e) – alert if above", value=999999.0)
    with colB:
        thresh_ec1 = st.number_input("Threshold for EC1 (EUR/FU) – alert if above", value=999999.0)

    alerts_view = latest[latest["scenario_code"] == sel_scenario].set_index("kpi_code")
    msgs = []
    if "E1" in alerts_view.index and float(alerts_view.loc["E1", "value"]) > thresh_e1:
        msgs.append(f"⚠️ E1 above threshold: {alerts_view.loc['E1','value']} > {thresh_e1}")
    if "EC1" in alerts_view.index and float(alerts_view.loc["EC1", "value"]) > thresh_ec1:
        msgs.append(f"⚠️ EC1 above threshold: {alerts_view.loc['EC1','value']} > {thresh_ec1}")

    if msgs:
        for m in msgs:
            st.warning(m)
    else:
        st.success("No alerts triggered under current thresholds.")

st.markdown("## Scenario comparison")
st.caption("Compare multiple scenarios using latest KPI results. BASE is used as reference when available.")

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

    st.markdown("### All vs BASE summary (Improved / Worse / Same / Missing)")
    if "BASE" not in pivot.columns:
        st.warning("BASE is not available in the comparison table. Add BASE scenario results first.")
    else:
        base_col = "BASE"
        meta_cols = {"kpi_code", "kpi_name", "unit", "is_benefit"}
        derived_prefixes = ("Δ ", "%Δ ", "Effect ")
        scenario_cols = [
            c for c in pivot.columns if (c not in meta_cols) and (not str(c).startswith(derived_prefixes))
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
            st.dataframe(df_long2[["scenario", "kpi_code", "kpi_name", "is_benefit", "BASE", "delta", "pct_delta", "effect"]], width="stretch")
            st.download_button(
                "Download KPI-by-KPI comparison (CSV)",
                df_long2.to_csv(index=False).encode("utf-8"),
                file_name="all_vs_base_kpi_detail.csv",
                mime="text/csv",
            )
