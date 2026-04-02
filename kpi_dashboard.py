from __future__ import annotations

import os
import tempfile
from pathlib import Path

# En Streamlit Cloud existe /mount/src
if os.path.exists("/mount/src"):
    os.environ["SUSTAINSC_DB_URL"] = "sqlite:////tmp/sustainsc.db"

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

# ============================================================================
# 0) Imports from sustainsc (antes de DB_URL setup para evitar duplicados)
# ============================================================================

from sustainsc.anylogistix_adapter import import_anylogistix_csv
from sustainsc.sim_export_adapter import read_any_file, normalize_any_export, upsert_measurements
from sustainsc.kpi_engine import run_full_pipeline

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

@st.cache_resource(show_spinner=False)
def bootstrap_everything():
    try:
        from load_example_data import main as load_example_data_main
        from sustainsc.kpi_engine import run_full_pipeline

        load_example_data_main()
        run_full_pipeline(debug_missing=True)
        return True
    except Exception as e:
        print(f"[BOOTSTRAP ERROR] {e}")
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

COMPOSITE_CODES = {"ENV_INDEX", "ECO_INDEX", "SOC_INDEX", "TECH_INDEX", "SUSTAIN_INDEX"}

def exclude_composites(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "kpi_code" not in df.columns:
        return df.copy()
    return df[~df["kpi_code"].isin(COMPOSITE_CODES)].copy()

def latest_norm_for_scenario(norm_df: pd.DataFrame, scenario_code: str) -> pd.DataFrame:
    if norm_df.empty:
        return pd.DataFrame()

    nn = norm_df[norm_df["scenario_code"] == scenario_code].copy()
    if nn.empty:
        return nn

    nn["period_end"] = pd.to_datetime(nn["period_end"], errors="coerce")
    nn = nn.sort_values("period_end").groupby("kpi_code", as_index=False).tail(1)
    nn = exclude_composites(nn)

    keep = ["kpi_code", "normalized_value", "semaforo", "baseline_value", "normalization_method"]
    keep = [c for c in keep if c in nn.columns]
    return nn[keep].copy()

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

def list_scenarios_by_prefix(prefixes=("ALX_", "SIM_", "VSMC_", "MILP_")):
    with engine.connect() as con:
        rows = con.execute(text("SELECT code FROM sc_scenario ORDER BY code")).fetchall()
    codes = [r[0] for r in rows]
    picked = [c for c in codes if any(c.startswith(p) for p in prefixes)]
    return picked

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
# Import Data (AnyLogistix / AnyLogic) - SINGLE unified sidebar section
# ============================================================================
import io
import tempfile
from pathlib import Path

st.sidebar.subheader("📥 Import Data")

with st.sidebar.expander("AnyLogistix / AnyLogic Simulation Results", expanded=False):
    tab_alx, tab_export = st.tabs(["AnyLogistix CSV (long)", "Export (.xlsx/.csv)"])

    # ------------------------------------------------------------------------
    # TAB 1: AnyLogistix long CSV importer
    # ------------------------------------------------------------------------
    with tab_alx:
        st.write("Upload a CSV (long format recommended).")
        st.caption(
            "Required columns: scenario_code, variable_name, value. "
            "Optional: timestamp, unit, source_system, comment."
        )

        up_alx = st.file_uploader(
            "CSV file",
            type=["csv"],
            key="anylogistix_uploader_long",
        )

        prefix = st.text_input(
            "Scenario prefix (optional)",
            value="ALX_",
            key="alx_prefix",
        )

        recalc_alx = st.checkbox(
            "Recalculate KPIs after import",
            value=True,
            key="alx_recalc",
        )

        if up_alx is not None:
            # Preview robust (doesn't consume the stream)
            try:
                data = up_alx.getvalue()
                df_preview = pd.read_csv(io.BytesIO(data))
                st.write("Preview (first 10 rows):")
                st.dataframe(df_preview.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Could not preview CSV: {e}")

            if st.button("Import into DB", type="primary", key="alx_import_btn"):
                try:
                    # 1) Save upload to temp file (Cloud-friendly)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(up_alx.getvalue())
                        tmp_path = Path(tmp.name)

                    # 2) Import (creates/updates scenarios + measurements)
                    with st.spinner("Importing AnyLogistix data..."):
                        stats = import_anylogistix_csv(tmp_path, scenario_prefix=prefix)

                    # 3) Recalc KPIs if requested
                    if recalc_alx:
                        with st.spinner("Recalculating KPIs..."):
                            run_full_pipeline(debug_missing=False)

                    # 4) Clear caches so filters/scenario lists refresh
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    try:
                        st.cache_resource.clear()
                    except Exception:
                        pass

                    # 5) Auto-select imported scenarios in comparison section
                    if hasattr(stats, 'scenario_codes') and stats.scenario_codes:
                        st.session_state["compare_autoselect"] = ["BASE"] + list(stats.scenario_codes)
                        st.session_state["scroll_to_compare"] = True

                    st.success(
                        f"✅ Imported OK | Scenarios: {stats.scenarios_touched} "
                        f"| Measurements: {stats.measurements_written}"
                    )
                    st.success("Imported. Go to [Scenario comparison](#scenario-compare) section (scroll down) to compare vs BASE.")
                    st.rerun()

                except FileNotFoundError as e:
                    st.error(f"❌ File error: {e}")
                except Exception as e:
                    st.error("❌ Import failed")
                    st.exception(e)

    # ------------------------------------------------------------------------
    # TAB 2: AnyLogistix/AnyLogic export importer (.xlsx/.csv "raw" exports)
    # ------------------------------------------------------------------------
    with tab_export:
        st.caption("Upload exported simulation results file (raw format).")

        uploaded = st.file_uploader(
            "Upload .xlsx or .csv (exported results)",
            type=["xlsx", "xls", "csv"],
            key="sim_export_uploader",
        )

        default_code = st.text_input(
            "Default scenario_code (if file doesn't include one)",
            value="SIM_ALX",
            key="sim_default_code",
        )

        source_label = st.text_input(
            "source_system label",
            value="AnyLogistix/AnyLogic",
            key="sim_source_label",
        )
st.markdown("---")
st.caption("Quick actions")

if st.button("🆚 Compare imported scenarios vs BASE", key="btn_compare_vs_base"):
    candidates = list_scenarios_by_prefix(prefixes=("ALX_", "SIM_", "VSMC_", "MILP_"))
    if not candidates:
        st.warning("No imported scenarios found (ALX_/SIM_/VSMC_/MILP_). Import data or rebuild demo first.")
    else:
        st.session_state["compare_scenarios_default"] = ["BASE"] + candidates[:4]
        st.success(f"Loaded {len(candidates)} scenario(s) for comparison.")
        st.rerun()


        if uploaded is not None:
            if st.button("Import → write MRV → recompute KPIs", key="sim_import_btn", type="primary"):
                try:
                    # 1) Save upload to temp file
                    suffix = "." + uploaded.name.split(".")[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getbuffer())
                        tmp_path = Path(tmp.name)

                    # 2) Read + normalize → MRV format
                    with st.spinner("Reading file..."):
                        df_raw = read_any_file(tmp_path)

                    with st.spinner("Normalizing to MRV format..."):
                        df_mrv = normalize_any_export(
                            df_raw,
                            default_scenario_code=default_code,
                            source_system=source_label,
                        )

                    # 3) Write to DB + recompute KPIs
                    with st.spinner("Writing measurements to database..."):
                        session = SessionLocal()
                        try:
                            written = upsert_measurements(session, df_mrv)
                        finally:
                            session.close()

                    with st.spinner("Recalculating KPIs..."):
                        run_full_pipeline(debug_missing=False)

                    # 4) Clear caches
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    try:
                        st.cache_resource.clear()
                    except Exception:
                        pass

                    # 5) Auto-select imported scenarios in comparison section
                    # Extract unique scenario codes from the imported data
                    imported_codes = sorted(df_mrv["scenario_code"].unique().tolist())
                    if imported_codes:
                        st.session_state["compare_autoselect"] = ["BASE"] + imported_codes
                        st.session_state["scroll_to_compare"] = True

                    st.success(f"✅ Imported {written} measurements. KPIs recalculated.")
                    st.success("Imported. Go to [Scenario comparison](#scenario-compare) section (scroll down) to compare vs BASE.")
                    st.rerun()

                except FileNotFoundError as e:
                    st.error(f"❌ File not found: {e}")
                except Exception as e:
                    st.error("❌ Import failed")
                    st.exception(e)

# ----------------------------------------------------------------------------
# Continue with the rest of your sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("Controls")

if st.sidebar.button("🔄 Rebuild demo (full)"):
    from load_example_data import main as load_example_data_main
    from sustainsc.kpi_engine import run_full_pipeline

    load_example_data_main()
    run_full_pipeline(debug_missing=True)

    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ============================================================================
# Load normalized KPI results + rules
# ============================================================================

@st.cache_data(ttl=30)
def load_normalized_results():
    from sustainsc.config import engine

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


def latest_norm_per_kpi_scenario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df2 = df.dropna(subset=["kpi_code", "scenario_code"]).sort_values("period_end")
    return df2.groupby(["scenario_code", "kpi_code"], as_index=False).tail(1)


def _default_base_index(options):
    if not options:
        return 0
    for i, s in enumerate(options):
        if "BASE" in str(s).upper():
            return i
    return 0


def _apply_common_filters(df, dim_sel, level_sel, flow_sel):
    out = df.copy()
    if dim_sel != "All":
        out = out[out["dimension"] == dim_sel]
    if level_sel != "All":
        out = out[out["decision_level"] == level_sel]
    if flow_sel != "All":
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

    # geometric (corrected)
    return float(100.0 * np.prod((np.maximum(vals, 1e-6) / 100.0) ** ws))


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
        detailed.groupby("scenario")
        .apply(lambda g: pd.Series({
            "Improved": int((g["effect"] == "Improved").sum()),
            "Worse": int((g["effect"] == "Worse").sum()),
            "Same": int((g["effect"] == "Same").sum()),
            "Missing": int((g["effect"] == "Missing").sum()),
            "Mean Δ (pts)": float(g["delta_pts"].dropna().mean()) if g["delta_pts"].notna().any() else np.nan,
            "Median Δ (pts)": float(g["delta_pts"].dropna().median()) if g["delta_pts"].notna().any() else np.nan,
            "Net score": int((g["effect"] == "Improved").sum()) - int((g["effect"] == "Worse").sum()),
        }))
        .reset_index()
        .sort_values(["Net score", "Mean Δ (pts)"], ascending=False)
    )

    by_dim = (
        detailed.groupby(["scenario", "dimension"])
        .apply(lambda g: pd.Series({
            "Improved": int((g["effect"] == "Improved").sum()),
            "Worse": int((g["effect"] == "Worse").sum()),
            "Same": int((g["effect"] == "Same").sum()),
            "Mean Δ (pts)": float(g["delta_pts"].dropna().mean()) if g["delta_pts"].notna().any() else np.nan,
        }))
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

    # usar solo KPI completos en todos los escenarios seleccionados
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


# ============================================================================
# Load normalized data
# ============================================================================

norm_df = load_normalized_results()
rules_df = load_normalization_rules()

if norm_df.empty:
    st.warning("⚠️ No normalized KPI results found. Rebuild demo or rerun the full pipeline.")
    st.stop()

norm_latest = latest_norm_per_kpi_scenario(norm_df)

if norm_latest.empty:
    st.warning("⚠️ Normalized KPI table is empty after latest-value selection.")
    st.stop()

# Sidebar filters based on normalized data
dimensions = ["All"] + sorted(norm_latest["dimension"].dropna().unique().tolist())
decision_levels = ["All"] + sorted(norm_latest["decision_level"].dropna().unique().tolist())
flows = ["All"] + sorted(norm_latest["flow"].dropna().unique().tolist())
scenarios = sorted(norm_latest["scenario_code"].dropna().unique().tolist())

sel_dim = st.sidebar.selectbox("Dimension", dimensions, index=0)
sel_level = st.sidebar.selectbox("Decision level", decision_levels, index=0)
sel_flow = st.sidebar.selectbox("Flow", flows, index=0)
sel_scenario = st.sidebar.selectbox("Scenario (main view)", scenarios, index=_default_base_index(scenarios))

norm_view = _apply_common_filters(norm_latest, sel_dim, sel_level, sel_flow)
norm_view = norm_view[norm_view["scenario_code"] == sel_scenario].copy()

# ============================================================================
# Section 1: Normalized KPI view (replaces raw KPI table)
# ============================================================================

st.subheader(f"Normalized KPI view – Scenario: {sel_scenario}")
st.caption("All analytical comparisons in this dashboard use normalized KPI scores (0–100), not raw KPI values.")

show_cols = [
    "kpi_code", "kpi_name", "dimension", "decision_level", "flow", "unit",
    "raw_value", "normalized_value", "semaforo", "baseline_value",
    "lower_ref", "upper_ref", "normalization_method"
]
show_cols = [c for c in show_cols if c in norm_view.columns]

styled_main = norm_view[show_cols].sort_values(["dimension", "kpi_code"]).style.map(
    color_semaforo, subset=["semaforo"]
)
st.dataframe(styled_main, use_container_width=True)

# ============================================================================
# Section 2: VSM-C Diagnostics (keep as separate diagnostic)
# ============================================================================

st.markdown("## VSM-C Diagnostics")
st.caption("VSM-C Diagnostic (lead time, VA ratio, hotspots) and auto-generated Kaizen scenario.")

if vsmc_main is not None and VSM_CSV is not None and VSM_CSV.exists():
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
# Section 3: Normalized scenario comparison vs reference
# ============================================================================

st.markdown('<div id="scenario-compare"></div>', unsafe_allow_html=True)
st.markdown("## Normalized Scenario Comparison")
st.caption("All scenario deviations are computed using normalized KPI scores, so directionality is already encoded.")

base_like = [s for s in scenarios if "BASE" in s.upper()]
ref_default = base_like[0] if base_like else scenarios[0]

reference_scenario = st.selectbox(
    "Reference scenario",
    scenarios,
    index=scenarios.index(ref_default),
    key="reference_scenario_norm"
)

default_compare = [s for s in scenarios if s != reference_scenario][:4]
compare_scenarios = st.multiselect(
    "Scenarios to compare against the reference",
    options=[s for s in scenarios if s != reference_scenario],
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
    norm_latest=norm_latest,
    reference_scenario=reference_scenario,
    selected_scenarios=compare_scenarios,
    dim_sel=sel_dim,
    level_sel=sel_level,
    flow_sel=sel_flow,
    tol=same_tolerance,
)

if detailed_cmp.empty:
    st.info("No normalized comparison data available for the selected filters.")
else:
    st.markdown("### Summary: improved / worse / same (normalized KPI scores)")
    st.dataframe(summary_cmp, use_container_width=True)

    st.markdown("### Summary by dimension")
    st.dataframe(by_dim_cmp.sort_values(["scenario", "dimension"]), use_container_width=True)

    st.markdown("### Detailed KPI effects (normalized)")
    det_show = detailed_cmp[[
        "scenario", "kpi_code", "kpi_name", "dimension",
        "reference_score", "scenario_score", "delta_pts",
        "reference_semaforo", "scenario_semaforo", "effect"
    ]].sort_values(["scenario", "dimension", "kpi_code"])
    st.dataframe(det_show, use_container_width=True)

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
        st.dataframe(top_imp[["kpi_code", "kpi_name", "dimension", "delta_pts", "effect"]], use_container_width=True)

    with col_b:
        st.write("**Top worsenings**")
        top_wrs = focus_df.sort_values("delta_pts", ascending=True).head(10)
        st.dataframe(top_wrs[["kpi_code", "kpi_name", "dimension", "delta_pts", "effect"]], use_container_width=True)

    st.markdown("### Traffic-light distribution by scenario")
    traffic_df = (
        _apply_common_filters(norm_latest, sel_dim, sel_level, sel_flow)
        .query("scenario_code in @([reference_scenario] + compare_scenarios)")
        .groupby(["scenario_code", "semaforo"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    st.dataframe(traffic_df, use_container_width=True)

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

# ============================================================================
# Section 4: Composite indices, corrected Sustain Index, sensitivity and MCDA
# ============================================================================

st.markdown("## Composite Indices, Sensitivity & MCDA")
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

dim_long_df, dim_wide_df = compute_dimension_indices(norm_latest, rules_df, dim_weights)

if dim_wide_df.empty:
    st.info("No composite/dimension indices could be computed from normalized KPI results.")
else:
    st.markdown("### Corrected composite index cards")
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
    dim_show = dim_wide_df[[
        "scenario_code", "environmental", "economic", "social", "technological",
        "SUSTAIN_INDEX_GEOM", "SUSTAIN_INDEX_ARITH"
    ]].sort_values("SUSTAIN_INDEX_GEOM", ascending=False)
    st.dataframe(dim_show, use_container_width=True)

    st.markdown("### Corrected Sustain Index ranking")
    st.bar_chart(dim_show.set_index("scenario_code")["SUSTAIN_INDEX_GEOM"])

    st.markdown("### Sensitivity analysis for selected scenario")
    sens_df = build_one_way_sensitivity(r)
    if not sens_df.empty:
        geom_chart = sens_df.pivot(index="focus_weight", columns="focus_dimension", values="SUSTAIN_INDEX_GEOM")
        arith_chart = sens_df.pivot(index="focus_weight", columns="focus_dimension", values="SUSTAIN_INDEX_ARITH")

        st.write("**One-way sensitivity (geometric Sustain Index)**")
        st.line_chart(geom_chart)

        st.write("**One-way sensitivity (arithmetic alternative)**")
        st.line_chart(arith_chart)

        with st.expander("📋 Show sensitivity table"):
            st.dataframe(sens_df, use_container_width=True)

    st.markdown("### MCDA (normalized KPI scores)")
    st.caption(
        "WSM uses the normalized KPI scores and the rule weights. "
        "TOPSIS is computed on the subset of KPI that are complete across the selected scenarios."
    )

    default_mcda = [reference_scenario] + compare_scenarios if compare_scenarios else [reference_scenario]
    mcda_scenarios = st.multiselect(
        "Scenarios for MCDA ranking",
        options=scenarios,
        default=[s for s in default_mcda if s in scenarios],
        key="mcda_scenarios"
    )

    global_weights = build_global_kpi_weights(rules_df, dim_weights)
    wsm_df = compute_wsm_scores(norm_latest, global_weights, mcda_scenarios)
    topsis_df = compute_topsis_scores(norm_latest, global_weights, mcda_scenarios)

    mcda_df = pd.merge(wsm_df, topsis_df, on="scenario_code", how="outer")
    if not mcda_df.empty:
        mcda_df["Rank_WSM"] = mcda_df["WSM_score"].rank(ascending=False, method="dense")
        if "TOPSIS_score" in mcda_df.columns:
            mcda_df["Rank_TOPSIS"] = mcda_df["TOPSIS_score"].rank(ascending=False, method="dense")
        mcda_df = mcda_df.sort_values(["Rank_WSM", "scenario_code"])

        st.dataframe(mcda_df, use_container_width=True)

        if "WSM_score" in mcda_df.columns:
            st.write("**WSM ranking**")
            st.bar_chart(mcda_df.set_index("scenario_code")["WSM_score"])

        if "TOPSIS_score" in mcda_df.columns:
            st.write("**TOPSIS ranking**")
            st.bar_chart(mcda_df.set_index("scenario_code")["TOPSIS_score"])
    else:
        st.info("MCDA ranking could not be computed with the current scenario selection.")
