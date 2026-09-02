"""Streamlit UI for completing and importing MRV scenario workbooks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional

import streamlit as st

from batch_completion_engine import BatchCompletionResult, BatchScenarioCompletionEngine
from sustainsc.numerical import NUMERICAL_COMPARISON
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
        "MRV Scenario Workbook",
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
            "Current workbooks declare their schema in 18_CASE_METADATA. "
            "Recognized earlier workbooks are handled through an explicit compatibility adapter."
        )
        return

    payload = uploaded.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    parser_key = (
        f"{checksum}:schema-2.0:parser-2:completion-5:rules-5:transport-boundary-1:"
        f"normalization-2:ec2-guard-1:tolerance-{NUMERICAL_COMPARISON.version}"
    )
    if st.session_state.get("mrv_workbook_key") != parser_key:
        st.cache_data.clear()
        for key in (
            "mrv_completion_result", "mrv_commit_result", "kpi_result",
            "mcda_result", "dpp_import_summary", "last_import_run_id",
            "normalization_result", "traffic_light_result",
        ):
            st.session_state.pop(key, None)
        st.session_state["mrv_workbook_key"] = parser_key
        engine = BatchScenarioCompletionEngine(config_dir)
        with NamedTemporaryFile(suffix=".xlsx", delete=False) as temp:
            temp.write(payload)
            temp_path = Path(temp.name)
        try:
            completed = engine.complete_batch_from_excel(temp_path)
            completed.source_filename = uploaded.name
            st.session_state["mrv_completion_result"] = completed
        except Exception as exc:
            st.error(f"The completion run could not be executed: {exc}")
            return
        finally:
            temp_path.unlink(missing_ok=True)
    result = st.session_state["mrv_completion_result"]

    parsed = result.parsed_workbook
    if parsed and parsed.schema.migration_required:
        st.warning(
            "This recognized earlier workbook was migrated to the current internal "
            "measurement contract. Deprecated evidence labels remain visible in diagnostics."
        )
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
            "Schema": (
                "Compatible legacy" if parsed and parsed.schema.migration_required
                else "Current" if parsed else "Unsupported"
            ),
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

    structure = result.structural_summary or {}
    if structure.get("complete"):
        st.success("MRV completion structure: Complete")
    else:
        st.error("MRV completion structure: Incomplete")
    render_data_status_panel({
        "Structural completion status": "Complete" if structure.get("complete") else "Incomplete",
        "Production QA status": "Passed" if not result.has_critical_failures else "Blocked",
        "DPP status": "PRE_DPP_PROVISIONAL" if result.completion_review["provisional"].any() else "POST_DPP_VALIDATED",
        "Regression comparison status": "Matched" if regression_count == 0 else f"{regression_count} differences",
        "Eligibility for KPI calculation": "Eligible" if result.can_commit else "Blocked",
    })

    with st.expander("Import diagnostics", expanded=False):
        review_provenance = result.completion_review["provenance"].fillna("").astype(str)
        exact_l3 = int(review_provenance.str.contains("permission_source=exact override", regex=False).sum())
        scoped_l3 = int(review_provenance.str.contains("permission_source=strategy scope", regex=False).sum())
        factor_rows = int(result.completion_review["rule_id"].eq("MRV_R_GHG_S1S2_FACTORS").sum())
        bridge_status = "not configured"
        if parsed is not None and not parsed.bridge_rules.empty:
            sd_bridge = parsed.bridge_rules[
                parsed.bridge_rules["bridge_rule_id"].astype(str).eq("BR_SD_MRV_COVERAGE")
            ]
            if not sd_bridge.empty and str(sd_bridge.iloc[0]["rule_status"]).upper() != "ACTIVE":
                bridge_status = "inactive — native index retained for audit"
        diagnostics = {
            "Detected workbook family": parsed.schema.schema_family if parsed else "unsupported",
            "Uploaded filename": result.source_filename or uploaded.name,
            "Uploaded SHA-256": result.workbook_sha256 or checksum,
            "Uploaded file size": result.workbook_size or len(payload),
            "Parser version": result.parser_version,
            "Completion-engine version": result.completion_engine_version,
            "Detected schema version": parsed.schema.schema_version if parsed else "unknown",
            "Detection source": parsed.schema.detected_from if parsed else "",
            "Migration adapter used": parsed.migration_adapter or "None" if parsed else "None",
            "Case ID": parsed.metadata.get("case_id", "") if parsed else "",
            "Dataset ID": parsed.metadata.get("dataset_id", "") if parsed else "",
            "Required sheets found": len(parsed.schema.required_sheets) if parsed else 0,
            "Parsed scenario count": len(parsed.scenarios) if parsed else 0,
            "Parsed dictionary variables": len(parsed.variable_dictionary) if parsed else 0,
            "Direct evidence count": len(parsed.direct_inputs) if parsed else 0,
            "Native output count": len(parsed.native_outputs) if parsed else 0,
            "Assumption count": len(parsed.assumptions) if parsed else 0,
            "Production QA failures": fail_count,
            "Regression differences": regression_count,
            "L3 permission source — exact override": exact_l3,
            "L3 permission source — strategy scope": scoped_l3,
            "Factor-based GHG status": f"executed for {factor_rows} scenarios" if factor_rows else "direct evidence preserved / configuration missing",
            "SD MRV bridge status": bridge_status,
            "EC2 denominator guard": "applied during KPI normalization when corroboration fails",
        }
        render_data_status_panel(diagnostics)
        if parsed is not None:
            factor_roles = parsed.factor_register["analytical_role"].astype(str)
            factor_scopes = parsed.factor_register["scope"].astype(str)
            transport_factors = parsed.factor_register[
                factor_roles.str.contains(
                    "transport-scope|active analytical", case=False, na=False, regex=True
                )
                & factor_scopes.str.contains("transport", case=False, na=False)
            ]
            reference_code = str(parsed.metadata.get("default_reference_scenario", ""))
            reference_rows = result.completion_review[
                result.completion_review["scenario_code"].eq(reference_code)
            ].set_index("variable_name")
            factor = transport_factors.iloc[0] if len(transport_factors) == 1 else None
            transport_qa = result.qa_report[
                result.qa_report["check_id"].eq("TRANSPORT_GHG_DOUBLE_COUNT_RISK")
            ]
            render_data_status_panel({
                "Transport GHG boundary": "outbound road transport",
                "Direct DES Vehicle CO2": "used" if result.completion_review["rule_id"].eq("BR_DES_TRANSPORT_GHG").any() else "not used",
                "Transport factor set": factor["factor_set_id"] if factor is not None else "not configured",
                "Transport factor code": factor["factor_code"] if factor is not None else "not configured",
                "Transport factor value": factor["value"] if factor is not None else "not configured",
                "Transport factor unit": factor["unit"] if factor is not None else "not configured",
                "E7 numerator": reference_rows.at["transport_ghg_tco2e", "completed_value"] if not reference_rows.empty else "unavailable",
                "E7 output denominator": reference_rows.at["output_qty_fu", "completed_value"] if not reference_rows.empty else "unavailable",
                "E7 result (kgCO2e/FU)": reference_rows.at["transport_ghg_intensity_fu", "completed_value"] if not reference_rows.empty else "unavailable",
                "Double-count status": "PASS" if not transport_qa.status.eq("FAIL").any() else "FAIL",
            })
        if result.evidence_outcomes is not None and not result.evidence_outcomes.empty:
            render_downloadable_table(
                result.evidence_outcomes.groupby("outcome", as_index=False).size(),
                filename="mrv_evidence_outcomes.csv",
                key="download_mrv_evidence_outcomes",
            )
        if result.failure_diagnostics is not None and not result.failure_diagnostics.empty:
            render_downloadable_table(
                result.failure_diagnostics,
                filename="mrv_import_failure_diagnostics.csv",
                key="download_mrv_import_failure_diagnostics",
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
            disabled=not result.can_commit,
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
            disabled=not result.can_commit,
        ):
            on_commit(result)
    else:
        st.info("Database import is not configured for this page.")
