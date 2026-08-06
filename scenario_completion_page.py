"""Streamlit UI for completing and importing MRV scenario workbooks."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

import streamlit as st

from batch_completion_engine import BatchCompletionResult, BatchScenarioCompletionEngine
from sustainsc.ui import (
    render_data_status_panel,
    render_downloadable_table,
    render_section_header,
    render_workflow_progress,
)


def render_scenario_completion_page(
    *,
    config_dir: str | Path,
    on_commit: Optional[Callable[[BatchCompletionResult], None]] = None,
) -> None:
    render_section_header(
        "MRV Scenario Workbook v2",
        "Auditable L1–L6 completion and conflict validation before KPI calculation.",
    )
    render_workflow_progress(
        {
            "Scenario setup": "complete",
            "Input evidence": "ready",
            "Causal completion": "pending",
            "QA and conflicts": "pending",
            "Review and commit": "pending",
        }
    )

    uploaded = st.file_uploader(
        "Upload the SustainSCM MRV measurement-completion workbook",
        type=["xlsx"],
        key="mrv_completion_workbook",
    )
    if uploaded is None:
        st.info(
            "Native v2 workbooks declare template_schema_version in 18_CASE_METADATA. "
            "Recognized legacy workbooks are handled only through the compatibility adapter."
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

    parsed = result.parsed_workbook
    fail_count = int(((result.qa_report["severity"] == "Critical") & (result.qa_report["status"] == "FAIL")).sum())
    warn_count = int((result.qa_report["status"] == "WARN").sum())
    regression_count = int(
        result.comparison_report["comparison_status"].isin(
            ["UNRESOLVED_DIFFERENCE", "MISSING_EXPECTED_VALUE"]
        ).sum()
    )
    l6_count = int((result.completion_review["rule_level"] == "L6").sum())

    level_counts = result.completion_review["rule_level"].value_counts().to_dict()
    render_data_status_panel(
        {
            "Schema": parsed.schema.version if parsed else "unknown",
            "Case ID": parsed.metadata.get("case_id", "") if parsed else "",
            "Dataset ID": parsed.metadata.get("dataset_id", "") if parsed else "",
            "Scenarios": len(result.scenario_results),
            "Common variables": int(parsed.variable_dictionary["common_upload_variable"].astype(str).str.lower().isin({"yes", "true", "1"}).sum()) if parsed else 0,
            "Direct evidence": len(parsed.direct_inputs) if parsed else 0,
            "Native outputs": len(parsed.native_outputs) if parsed else 0,
            "Assumptions": len(parsed.assumptions) if parsed else 0,
            "Critical failures": fail_count,
            "Warnings": warn_count,
            "Regression differences": regression_count,
            "L1 Direct": int(level_counts.get("L1", 0)),
            "L2 Derived": int(level_counts.get("L2", 0)),
            "L3 Scaled": int(level_counts.get("L3", 0)),
            "L4 Bridge": int(level_counts.get("L4", 0)),
            "L5 Assumed": int(level_counts.get("L5", 0)),
            "L6 BASE": l6_count,
        }
    )

    tabs = st.tabs(["Causal completion", "Production QA", "Review and commit", "Regression comparison"])
    with tabs[0]:
        review = result.completion_review.copy()
        levels = sorted(review["rule_level"].dropna().unique().tolist())
        selected_levels = st.multiselect(
            "Completion levels",
            levels,
            default=levels,
            key="completion_rule_levels",
        )
        search = st.text_input(
            "Search variables",
            placeholder="Variable name or source",
            key="completion_search",
        )
        review = review[review["rule_level"].isin(selected_levels)]
        if search:
            mask = review.astype(str).apply(
                lambda column: column.str.contains(search, case=False, na=False)
            ).any(axis=1)
            review = review[mask]
        render_downloadable_table(
            review,
            filename="mrv_completion_review.csv",
            key="download_mrv_completion_review",
            height=430,
        )
    with tabs[1]:
        critical = result.qa_report[
            (result.qa_report["severity"] == "Critical")
            & (result.qa_report["status"] == "FAIL")
        ]
        with st.expander("Critical failure details", expanded=not critical.empty):
            if critical.empty:
                st.success("The dataset passed production-critical QA.")
            else:
                st.error(f"Commit blocked by {len(critical)} production-critical failures.")
                render_downloadable_table(
                    critical,
                    filename="mrv_production_critical_failures.csv",
                    key="download_mrv_production_critical_failures",
                )
        warnings = result.qa_report[result.qa_report["status"] == "WARN"]
        with st.expander("Warnings by category"):
            if warnings.empty:
                st.info("No production QA warnings.")
            else:
                render_downloadable_table(
                    warnings.groupby("check_id", as_index=False).size(),
                    filename="mrv_warnings_by_category.csv",
                    key="download_mrv_warnings_by_category",
                )
        statuses = sorted(result.qa_report["status"].dropna().unique().tolist())
        selected_statuses = st.multiselect(
            "QA status",
            statuses,
            default=statuses,
            key="completion_qa_status",
        )
        qa_view = result.qa_report[result.qa_report["status"].isin(selected_statuses)]
        render_downloadable_table(
            qa_view,
            filename="mrv_qa_report.csv",
            key="download_mrv_qa_report",
            height=430,
        )
    with tabs[2]:
        if result.has_critical_failures:
            st.error(f"Commit blocked by {fail_count} production-critical failures.")
        elif regression_count:
            st.warning(
                "The dataset passed production QA. Historical regression comparison "
                f"contains {regression_count} differences. Review the comparison report "
                "before accepting a new baseline."
            )
        else:
            st.success("The dataset passed production QA and regression comparison.")
        render_downloadable_table(
            result.software_upload,
            filename="completed_mrv_scenarios.csv",
            key="download_mrv_software_upload_table",
        )
        csv_bytes = result.software_upload.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download completed MRV CSV",
            data=csv_bytes,
            file_name="completed_mrv_scenarios.csv",
            mime="text/csv",
            disabled=result.has_critical_failures,
        )
    with tabs[3]:
        differences = result.comparison_report[
            result.comparison_report["comparison_status"] != "MATCH"
        ]
        st.caption(
            "Historical/reference comparison only. These rows do not block commit "
            "unless a separate strict-regression workflow is configured."
        )
        render_downloadable_table(
            differences,
            filename="mrv_validation_comparison.csv",
            key="download_mrv_validation_comparison",
        )

    if on_commit is not None:
        if st.button(
            "Import scenarios and run KPI pipeline",
            type="primary",
            disabled=result.has_critical_failures,
        ):
            on_commit(result)
    else:
        st.info("Database import is not configured for this page.")
