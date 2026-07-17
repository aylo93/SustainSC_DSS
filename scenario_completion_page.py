"""Streamlit UI for completing and importing MRV scenario workbooks."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

import streamlit as st

from batch_completion_engine import BatchCompletionResult, BatchScenarioCompletionEngine


def render_scenario_completion_page(
    *,
    config_dir: str | Path,
    on_commit: Optional[Callable[[BatchCompletionResult], None]] = None,
) -> None:
    st.subheader("MRV Causal Scenario Completion")
    st.caption(
        "Complete and validate all workbook scenarios before running the KPI pipeline."
    )

    uploaded = st.file_uploader(
        "Upload the SustainSCM MRV batch workbook",
        type=["xlsx"],
        key="mrv_completion_workbook",
    )
    if uploaded is None:
        st.info(
            "Required sheets: 01_SCENARIOS, 02_DIRECT_MRV_INPUT, 03_NATIVE_OUTPUTS, "
            "04_APPROVED_ASSUMPTIONS, 05_REFERENCE_BASE and 11_EXPECTED_CH7_MRV."
        )
        return

    engine = BatchScenarioCompletionEngine(config_dir)
    with NamedTemporaryFile(suffix=".xlsx", delete=False) as temp:
        temp.write(uploaded.getbuffer())
        temp_path = Path(temp.name)

    try:
        result = engine.complete_batch_from_excel(temp_path)
    except Exception as exc:
        st.error(f"The completion run could not be executed: {exc}")
        return
    finally:
        temp_path.unlink(missing_ok=True)

    fail_count = int(((result.qa_report["severity"] == "Critical") & (result.qa_report["status"] == "FAIL")).sum())
    warn_count = int((result.qa_report["status"] == "WARN").sum())
    l6_count = int((result.completion_review["rule_level"] == "L6").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenarios", len(result.scenario_results))
    c2.metric("Critical failures", fail_count)
    c3.metric("Warnings", warn_count)
    c4.metric("BASE-retained values", l6_count)

    tabs = st.tabs(["Completion review", "QA report", "Software upload", "Chapter 7 comparison"])
    with tabs[0]:
        st.dataframe(result.completion_review, width="stretch")
    with tabs[1]:
        st.dataframe(result.qa_report, width="stretch")
    with tabs[2]:
        st.dataframe(result.software_upload, width="stretch")
        csv_bytes = result.software_upload.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download completed MRV CSV",
            data=csv_bytes,
            file_name="completed_mrv_scenarios.csv",
            mime="text/csv",
            disabled=result.has_critical_failures,
        )
    with tabs[3]:
        st.dataframe(result.comparison_report, width="stretch")

    if on_commit is not None:
        if st.button(
            "Import scenarios and run KPI pipeline",
            type="primary",
            disabled=result.has_critical_failures,
        ):
            on_commit(result)
    else:
        st.info("Database import is not configured for this page.")
