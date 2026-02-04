# kpi_dashboard_advanced.py
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

DB_PATH = "sustainsc.db"
THRESHOLDS_CSV = Path("data/kpi_thresholds.csv")


@st.cache_data
def load_db():
    con = sqlite3.connect(DB_PATH)

    kpi = pd.read_sql_query(
        """
        SELECT id AS kpi_id, code, name, description, dimension, decision_level, flow, unit, is_benefit, formula_id
        FROM sc_kpi
        ORDER BY code
        """,
        con,
    )

    scen = pd.read_sql_query(
        """
        SELECT id AS scenario_id, code AS scenario_code, name AS scenario_name
        FROM sc_scenario
        ORDER BY id
        """,
        con,
    )

    res = pd.read_sql_query(
        """
        SELECT
            r.id AS kpi_result_id,
            r.kpi_id,
            r.scenario_id,
            r.value,
            r.period_end AS timestamp
        FROM sc_kpi_result r
        """,
        con,
    )

    meas = pd.read_sql_query(
        """
        SELECT scenario_id, variable_name, value, unit, timestamp
        FROM sc_measurement
        """,
        con,
    )

    ef = pd.read_sql_query(
        """
        SELECT activity_type, value AS ef_value, unit
        FROM sc_emission_factor
        """,
        con,
    )

    con.close()

    # normalize
    for col in ["dimension", "decision_level", "flow"]:
        if col in kpi.columns:
            kpi[col] = kpi[col].astype(str).str.lower().str.strip()

    res["timestamp"] = pd.to_datetime(res["timestamp"], errors="coerce")
    meas["timestamp"] = pd.to_datetime(meas["timestamp"], errors="coerce")

    return kpi, scen, res, meas, ef


def apply_filters_kpi_catalog(kpi_df, dim_sel, lvl_sel, flow_sel):
    df = kpi_df.copy()
    if dim_sel != "All":
        df = df[df["dimension"] == dim_sel]
    if lvl_sel != "All":
        df = df[df["decision_level"] == lvl_sel]
    if flow_sel != "All":
        df = df[df["flow"] == flow_sel]
    return df


def asof_latest_results(kpi_df, scen_df, res_df, asof_ts, dim_sel, lvl_sel, flow_sel, scenario_sel):
    """
    Returns: latest KPI results as-of selected timestamp, under filters.
    """
    df = res_df.merge(scen_df, on="scenario_id", how="left").merge(kpi_df, on="kpi_id", how="left")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] <= asof_ts]

    # Filters on KPI meta
    if dim_sel != "All":
        df = df[df["dimension"] == dim_sel]
    if lvl_sel != "All":
        df = df[df["decision_level"] == lvl_sel]
    if flow_sel != "All":
        df = df[df["flow"] == flow_sel]

    # Scenario filter (optional for summary)
    if scenario_sel != "All":
        df = df[df["scenario_code"] == scenario_sel]

    if df.empty:
        return df

    # latest per KPI code per scenario (as-of)
    df = df.sort_values(["scenario_code", "code", "timestamp"], ascending=[True, True, False])
    df = df.groupby(["scenario_code", "code"], as_index=False).first()

    cols = ["scenario_code", "code", "name", "dimension", "decision_level", "flow", "value", "unit", "timestamp"]
    return df[cols].sort_values(["scenario_code", "code"])


def trend_timeseries(res_df, kpi_df, scen_df, kpi_code_sel):
    df = res_df.merge(scen_df, on="scenario_id", how="left").merge(kpi_df, on="kpi_id", how="left")
    df = df[df["code"] == kpi_code_sel].dropna(subset=["timestamp"])
    if df.empty:
        return df

    df = df.sort_values("timestamp")
    # pivot: time index, scenario columns
    pivot = df.pivot_table(index="timestamp", columns="scenario_code", values="value", aggfunc="last")
    return pivot


def month_window(ts: pd.Timestamp):
    start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
    end = (start + pd.offsets.MonthBegin(1))
    return start, end


def contribution_co2e(meas_df, ef_df, scenario_id, asof_ts):
    """
    CO2e contribution (kgCO2e) by activity variables that have emission factors.
    Uses measurements within the month of asof_ts.
    """
    start, end = month_window(asof_ts)

    m = meas_df[(meas_df["scenario_id"] == scenario_id) & (meas_df["timestamp"] >= start) & (meas_df["timestamp"] < end)].copy()
    if m.empty:
        return pd.DataFrame()

    # join factors on activity_type = variable_name
    ef = ef_df.copy()
    ef["activity_type"] = ef["activity_type"].astype(str).str.strip()

    m["variable_name"] = m["variable_name"].astype(str).str.strip()
    merged = m.merge(ef, left_on="variable_name", right_on="activity_type", how="inner")

    if merged.empty:
        return pd.DataFrame()

    merged["kgco2e"] = merged["value"].astype(float) * merged["ef_value"].astype(float)
    out = merged.groupby("variable_name", as_index=False)["kgco2e"].sum().sort_values("kgco2e", ascending=False)

    total = out["kgco2e"].sum()
    out["share_%"] = (out["kgco2e"] / total * 100.0) if total > 0 else 0.0
    return out


def contribution_cost(meas_df, scenario_id, asof_ts):
    """
    Cost contribution using direct cost measurements (EUR).
    Uses measurements within the month of asof_ts.
    """
    start, end = month_window(asof_ts)
    m = meas_df[(meas_df["scenario_id"] == scenario_id) & (meas_df["timestamp"] >= start) & (meas_df["timestamp"] < end)].copy()
    if m.empty:
        return pd.DataFrame()

    cost_vars = [
        "fuel_cost_eur",
        "electricity_cost_eur",
        "maintenance_cost_eur",
        "logistics_cost_eur",
        "operating_cost_eur",
    ]
    c = m[m["variable_name"].isin(cost_vars)].copy()
    if c.empty:
        return pd.DataFrame()

    out = c.groupby("variable_name", as_index=False)["value"].sum()
    out = out.rename(columns={"value": "eur"}).sort_values("eur", ascending=False)

    total = out["eur"].sum()
    out["share_%"] = (out["eur"] / total * 100.0) if total > 0 else 0.0
    return out


def load_thresholds():
    if not THRESHOLDS_CSV.exists():
        return pd.DataFrame(columns=["kpi_code", "operator", "threshold", "severity", "message"])
    df = pd.read_csv(THRESHOLDS_CSV).dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    return df


def eval_alerts(latest_df, thresholds_df):
    """
    latest_df: as-of results with columns [scenario_code, code, value, unit, ...]
    thresholds_df: kpi_code, operator, threshold, severity, message
    """
    if latest_df.empty or thresholds_df.empty:
        return pd.DataFrame()

    t = thresholds_df.copy()
    t["kpi_code"] = t["kpi_code"].astype(str).str.strip()

    df = latest_df.copy()
    df["code"] = df["code"].astype(str).str.strip()

    merged = df.merge(t, left_on="code", right_on="kpi_code", how="inner")
    if merged.empty:
        return pd.DataFrame()

    def check(op, v, thr):
        if pd.isna(v) or pd.isna(thr):
            return False
        if op == ">":
            return v > thr
        if op == ">=":
            return v >= thr
        if op == "<":
            return v < thr
        if op == "<=":
            return v <= thr
        if op == "==":
            return v == thr
        return False

    merged["triggered"] = merged.apply(lambda r: check(str(r["operator"]).strip(), float(r["value"]), float(r["threshold"])), axis=1)
    alerts = merged[merged["triggered"]].copy()

    cols = ["severity", "scenario_code", "code", "name", "value", "unit", "operator", "threshold", "message", "timestamp"]
    alerts = alerts[cols].sort_values(["severity", "scenario_code", "code"])
    return alerts


def scenario_matrix(asof_df, kpi_catalog_filtered):
    """
    Pivot KPIs (rows) x scenarios (columns) for as-of values
    """
    if asof_df.empty:
        return pd.DataFrame()

    allowed = set(kpi_catalog_filtered["code"].tolist())
    df = asof_df[asof_df["code"].isin(allowed)].copy()

    piv = df.pivot_table(index="code", columns="scenario_code", values="value", aggfunc="first")
    piv = piv.sort_index()
    return piv


def main():
    st.set_page_config(page_title="SustainSC DSS – Advanced KPI Dashboard", layout="wide")
    st.title("SustainSC DSS – Advanced KPI Dashboard (Prototype)")

    kpi_df, scen_df, res_df, meas_df, ef_df = load_db()
    thresholds_df = load_thresholds()

    # Sidebar filters
    st.sidebar.header("Filters")

    scenario_options = ["All"] + scen_df["scenario_code"].tolist()
    scenario_sel = st.sidebar.selectbox("Scenario (for summary & contributions)", scenario_options, index=0)

    dim_options = ["All"] + sorted(kpi_df["dimension"].dropna().unique().tolist())
    dim_sel = st.sidebar.selectbox("Dimension", dim_options, index=0)

    lvl_options = ["All"] + sorted(kpi_df["decision_level"].dropna().unique().tolist())
    lvl_sel = st.sidebar.selectbox("Decision level", lvl_options, index=0)

    flow_options = ["All"] + sorted(kpi_df["flow"].dropna().unique().tolist())
    flow_sel = st.sidebar.selectbox("Flow", flow_options, index=0)

    # As-of timestamp selector (needed for matrix + alerts)
    all_ts = sorted(res_df["timestamp"].dropna().unique())
    if not all_ts:
        st.error("No KPI results found in sc_kpi_result.")
        return

    asof_ts = st.sidebar.selectbox("As-of (result timestamp)", all_ts, index=len(all_ts) - 1)

    # Filtered KPI catalog (for coverage + matrix)
    kpi_filtered = apply_filters_kpi_catalog(kpi_df, dim_sel, lvl_sel, flow_sel)

    # Latest as-of values
    asof_df = asof_latest_results(kpi_df, scen_df, res_df, asof_ts, dim_sel, lvl_sel, flow_sel, scenario_sel)

    # ---- Coverage ----
    st.subheader("1) KPI Coverage (as-of)")

    computed_n = len(asof_df[["scenario_code", "code"]].drop_duplicates()) if not asof_df.empty else 0
    total_kpis = len(kpi_filtered["code"].unique())
    st.write(f"Computed KPI entries under current filters: **{computed_n}** (Total KPIs in catalog under filters: **{total_kpis}**)")

    with st.expander("Show as-of KPI table"):
        st.dataframe(asof_df, width="stretch")

    # ---- Trend time series ----
    st.subheader("2) Trend time series")
    kpi_code_options = kpi_filtered["code"].tolist() if not kpi_filtered.empty else kpi_df["code"].tolist()
    kpi_code_sel = st.selectbox("Select KPI code for trend chart", kpi_code_options, index=0)

    pivot = trend_timeseries(res_df, kpi_df, scen_df, kpi_code_sel)
    if pivot.empty:
        st.info("No time series points found for this KPI.")
    else:
        st.line_chart(pivot)

    # ---- Contribution analysis ----
    st.subheader("3) Contribution analysis (CO₂e and cost)")

    # choose scenario id
    if scenario_sel == "All":
        st.info("Select a specific scenario in the sidebar to see contribution charts.")
    else:
        sc_row = scen_df[scen_df["scenario_code"] == scenario_sel]
        if sc_row.empty:
            st.warning("Selected scenario not found in DB.")
        else:
            scenario_id = int(sc_row.iloc[0]["scenario_id"])

            co2 = contribution_co2e(meas_df, ef_df, scenario_id, pd.Timestamp(asof_ts))
            if co2.empty:
                st.warning("No CO₂e contributions could be computed (missing emission factors or measurements in that month).")
            else:
                st.write("CO₂e contributions (kgCO₂e) by activity:")
                st.bar_chart(co2.set_index("variable_name")["kgco2e"])

            cost = contribution_cost(meas_df, scenario_id, pd.Timestamp(asof_ts))
            if cost.empty:
                st.warning("No cost contribution could be computed (missing cost measurements in that month).")
            else:
                st.write("Cost contributions (EUR) by cost component:")
                st.bar_chart(cost.set_index("variable_name")["eur"])

    # ---- Alerts & thresholds ----
    st.subheader("4) Alerts & thresholds")

    if thresholds_df.empty:
        st.info("No thresholds file found. Create data/kpi_thresholds.csv to enable alerts.")
    else:
        alerts = eval_alerts(asof_df, thresholds_df)
        if alerts.empty:
            st.success("No alerts triggered under the current filters and as-of timestamp.")
        else:
            st.warning(f"Alerts triggered: {len(alerts)}")
            st.dataframe(alerts, width="stretch")

    # ---- Scenario comparison matrix ----
    st.subheader("5) Scenario comparison matrix (as-of)")

    # For matrix we need "All scenarios" regardless of summary filter
    asof_all_scen = asof_latest_results(kpi_df, scen_df, res_df, asof_ts, dim_sel, lvl_sel, flow_sel, scenario_sel="All")
    matrix = scenario_matrix(asof_all_scen, kpi_filtered)

    if matrix.empty:
        st.info("No data for scenario comparison under current filters.")
    else:
        st.dataframe(matrix, width="stretch")

    with st.expander("Show KPI catalog (filtered)"):
        st.dataframe(kpi_filtered[["code", "name", "dimension", "decision_level", "flow", "unit"]], width="stretch")


if __name__ == "__main__":
    main()
