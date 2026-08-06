"""SustainSCM MRV Causal Scenario Completion Engine.

This module sits immediately before the existing SustainSCM KPI engine. It converts
partial scenario evidence into a complete, auditable seven-column MRV dataset.
It deliberately does not calculate normalized KPIs, composite indices, sensitivity
results, or MCDA rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
import math
import re
import uuid

import pandas as pd

UPLOAD_COLUMNS = [
    "variable_name",
    "value",
    "unit",
    "timestamp",
    "scenario_code",
    "source_system",
    "comment",
]

INPUT_SHEETS = {
    "scenario": "01_SCENARIO_CONFIG",
    "direct": "02_DIRECT_MRV_INPUT",
    "native": "03_NATIVE_MODEL_OUTPUTS",
    "assumptions": "04_APPROVED_ASSUMPTIONS",
    "reference": "05_REFERENCE_BASE",
}

OUTPUT_SHEETS = {
    "review": "11_COMPLETION_REVIEW",
    "upload": "12_SOFTWARE_UPLOAD",
    "qa": "13_QA_REPORT",
}

RULE_SOURCE_SYSTEM = {
    "L1": "direct_mrv_input",
    "L2": "mrv_derived_recalculation",
    "L3": "causal_baseline_scaling",
    "L4": "native_to_mrv_bridge",
    "L5": "approved_strategy_assumption",
    "L6": "causal_base_retention",
}


@dataclass(frozen=True)
class Permission:
    strategy_code: str
    variable_name: str
    influence_status: str
    permitted_rules: str
    priority: int
    justification: str

    def allows(self, level: str) -> bool:
        return level.upper() in _expand_rule_expression(self.permitted_rules)

    @property
    def blocks_change(self) -> bool:
        return self.influence_status == "Retain_BASE" or self.permitted_rules == "L6"


@dataclass
class CompletionResult:
    scenario_code: str
    run_id: str
    completion_review: pd.DataFrame
    software_upload: pd.DataFrame
    qa_report: pd.DataFrame
    rejected_inputs: pd.DataFrame

    @property
    def has_critical_failures(self) -> bool:
        if self.qa_report.empty:
            return False
        return bool(
            ((self.qa_report["severity"] == "Critical") & (self.qa_report["status"] == "FAIL")).any()
        )

    def export_csv(self, path: str | Path, *, force: bool = False) -> Path:
        if self.has_critical_failures and not force:
            raise ValueError("Critical QA failures block CSV export. Review qa_report first.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.software_upload.loc[:, UPLOAD_COLUMNS].to_csv(target, index=False)
        return target


class ScenarioCompletionEngine:
    """Complete scenario-level MRV measurements using a versioned L1-L6 protocol."""

    def __init__(
        self,
        config_dir: str | Path,
        *,
        config_frames: Mapping[str, pd.DataFrame] | None = None,
        engine_version: str = "1.0.0",
        rule_version: str = "1.0.0",
        strict_approval: bool = True,
        numerical_tolerance: float = 1e-6,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.engine_version = engine_version
        self.rule_version = rule_version
        self.strict_approval = strict_approval
        self.numerical_tolerance = float(numerical_tolerance)

        supplied = config_frames or {}
        self.dictionary = _clean_frame(supplied.get("dictionary", self._read_config("mrv_dictionary.csv")))
        self.scope = _clean_frame(supplied.get("scope", self._read_config("strategy_scope.csv")))
        self.overrides = _clean_frame(supplied.get("overrides", self._read_config("variable_overrides.csv")))
        self.rules = _clean_frame(supplied.get("rules", self._read_config("mrv_rules.csv")))
        self.bridges = _clean_frame(supplied.get("bridges", self._read_config("bridge_rules.csv")))

        self._validate_config()
        self.dictionary = self.dictionary.set_index("variable_name", drop=False)
        self._rule_order = self._topological_rule_order()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete_from_excel(
        self,
        workbook_path: str | Path,
        *,
        reference_csv: str | Path | None = None,
        output_csv: str | Path | None = None,
        write_back_workbook: str | Path | None = None,
        force_export: bool = False,
    ) -> CompletionResult:
        inputs = self.load_excel_inputs(workbook_path)
        if reference_csv is not None:
            reference = pd.read_csv(reference_csv)
        else:
            reference = inputs["reference"]

        result = self.complete(
            scenario=inputs["scenario"],
            direct_inputs=inputs["direct"],
            native_outputs=inputs["native"],
            assumptions=inputs["assumptions"],
            reference=reference,
        )

        if write_back_workbook is not None:
            self.write_outputs_to_workbook(
                source_workbook=workbook_path,
                destination_workbook=write_back_workbook,
                result=result,
            )
        if output_csv is not None:
            result.export_csv(output_csv, force=force_export)
        return result

    def load_excel_inputs(self, workbook_path: str | Path) -> dict[str, pd.DataFrame | pd.Series]:
        source = Path(workbook_path)
        if not source.exists():
            raise FileNotFoundError(source)
        xls = pd.ExcelFile(source)
        missing = [sheet for sheet in INPUT_SHEETS.values() if sheet not in xls.sheet_names]
        if missing:
            raise ValueError(f"Workbook is missing required sheets: {missing}")

        scenario_df = _clean_frame(pd.read_excel(source, sheet_name=INPUT_SHEETS["scenario"], header=3))
        if scenario_df.empty:
            raise ValueError("01_SCENARIO_CONFIG does not contain an active scenario row.")

        return {
            "scenario": scenario_df.iloc[0],
            "direct": _clean_frame(pd.read_excel(source, sheet_name=INPUT_SHEETS["direct"], header=3)),
            "native": _clean_frame(pd.read_excel(source, sheet_name=INPUT_SHEETS["native"], header=3)),
            "assumptions": _clean_frame(pd.read_excel(source, sheet_name=INPUT_SHEETS["assumptions"], header=3)),
            "reference": _clean_frame(pd.read_excel(source, sheet_name=INPUT_SHEETS["reference"], header=3)),
        }

    def complete(
        self,
        *,
        scenario: pd.Series | Mapping[str, Any],
        direct_inputs: pd.DataFrame,
        native_outputs: pd.DataFrame,
        assumptions: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> CompletionResult:
        scenario = pd.Series(dict(scenario))
        scenario_code = _required_text(scenario.get("scenario_code"), "scenario_code")
        reference_code = _required_text(scenario.get("reference_scenario"), "reference_scenario")
        timestamp = _required_text(scenario.get("evaluation_timestamp"), "evaluation_timestamp")
        run_id = uuid.uuid4().hex[:16]
        selected_strategies = _selected_strategies(scenario)

        qa: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        variable_issues: dict[str, list[tuple[str, str]]] = {}

        def issue(
            check_id: str,
            severity: str,
            status: str,
            message: str,
            *,
            variable: str = "",
            action: str = "Review the input or configuration.",
        ) -> None:
            qa.append(
                {
                    "check_id": check_id,
                    "severity": severity,
                    "status": status,
                    "affected_variable": variable,
                    "message": message,
                    "required_action": action,
                    "scenario_code": scenario_code,
                    "run_id": run_id,
                }
            )
            if variable and status in {"WARN", "FAIL"}:
                variable_issues.setdefault(variable, []).append((severity, message))

        if not selected_strategies:
            issue("QA_SCENARIO_STRATEGY", "Critical", "FAIL", "At least one strategy must be selected.")
        else:
            issue("QA_SCENARIO_STRATEGY", "Critical", "PASS", "At least one strategy is selected.")

        if _text(scenario.get("approval_status")) not in {"Approved", "Not required"}:
            issue(
                "QA_SCENARIO_APPROVAL",
                "Warning",
                "WARN",
                "The scenario configuration is not yet approved.",
                action="Approve the scenario before committing it to the database.",
            )
        else:
            issue("QA_SCENARIO_APPROVAL", "Warning", "PASS", "Scenario approval status is acceptable.")

        reference = self._prepare_reference(reference, reference_code)
        base_values = dict(zip(reference["variable_name"], reference["value"]))
        base_units = dict(zip(reference["variable_name"], reference["unit"]))
        common_variables = self.dictionary[
            self.dictionary["common_upload_variable"].map(_yes)
        ]["variable_name"].tolist()

        missing_reference = [v for v in common_variables if v not in base_values]
        if missing_reference:
            issue(
                "QA_REFERENCE_COMPLETENESS",
                "Critical",
                "FAIL",
                f"Reference scenario is missing {len(missing_reference)} common variables.",
                action="Load a complete validated BASE/reference dataset.",
            )
        else:
            issue("QA_REFERENCE_COMPLETENESS", "Critical", "PASS", "Reference scenario is complete.")

        direct_inputs = _clean_frame(direct_inputs)
        native_outputs = _clean_frame(native_outputs)
        assumptions = _clean_frame(assumptions)

        self._check_duplicates(direct_inputs, "variable_name", "direct input", issue)
        self._check_duplicates(native_outputs, "native_variable", "native output", issue)
        self._check_duplicates(assumptions, "assumption_id", "assumption", issue)

        # Every common variable begins as an L6 reference value. Higher evidence
        # levels replace it only when the causal scope permits the change.
        state: dict[str, dict[str, Any]] = {}
        for variable in common_variables:
            state[variable] = {
                "value": _float_or_none(base_values.get(variable)),
                "rule_level": "L6",
                "rule_id": "BASE_RETENTION",
                "selected_strategy": "",
                "source_module": "Reference MRV",
                "source_reference": reference_code,
                "direct_input": None,
                "native_input": None,
                "provenance": f"Reference value retained from {reference_code}.",
            }

        # L1 direct inputs -------------------------------------------------
        direct_map: dict[str, float] = {}
        for _, row in direct_inputs.iterrows():
            variable = _text(row.get("variable_name"))
            value = _float_or_none(row.get("scenario_value"))
            if not variable or value is None:
                continue
            direct_map[variable] = value
            if variable not in self.dictionary.index:
                issue("QA_UNKNOWN_DIRECT_VARIABLE", "Critical", "FAIL", f"Unknown direct-input variable: {variable}", variable=variable)
                rejected.append({"input_type": "direct", "variable_name": variable, "value": value, "reason": "Unknown variable"})
                continue
            meta = self.dictionary.loc[variable]
            if not _yes(meta.get("user_input_allowed")):
                issue("QA_DERIVED_DIRECT_INPUT", "Critical", "FAIL", "Direct input is not allowed for a derived/alias variable.", variable=variable)
                rejected.append({"input_type": "direct", "variable_name": variable, "value": value, "reason": "User input not allowed"})
                continue
            supplied_unit = _text(row.get("canonical_unit"))
            canonical_unit = _text(meta.get("canonical_unit"))
            if supplied_unit and canonical_unit and supplied_unit != canonical_unit:
                issue("QA_UNIT_MISMATCH", "Critical", "FAIL", f"Unit {supplied_unit!r} does not match canonical unit {canonical_unit!r}.", variable=variable)
                rejected.append({"input_type": "direct", "variable_name": variable, "value": value, "reason": "Unit mismatch"})
                continue
            if self.strict_approval and _text(row.get("approval_status")) != "Approved":
                issue("QA_UNAPPROVED_DIRECT_INPUT", "Warning", "WARN", "Direct input was ignored because it is not approved.", variable=variable, action="Approve the input or remove it.")
                rejected.append({"input_type": "direct", "variable_name": variable, "value": value, "reason": "Not approved"})
                continue
            permission = self.resolve_permission(selected_strategies, variable)
            if permission.blocks_change or not permission.allows("L1"):
                issue("QA_UNAUTHORIZED_DIRECT_INPUT", "Critical", "FAIL", f"Selected strategies do not permit an L1 change to {variable}.", variable=variable, action="Remove the value or add a scientifically defensible strategy/rule.")
                rejected.append({"input_type": "direct", "variable_name": variable, "value": value, "reason": "Outside causal scope"})
                continue
            state[variable].update(
                value=value,
                rule_level="L1",
                rule_id="DIRECT_INPUT",
                selected_strategy=permission.strategy_code,
                source_module=_text(row.get("source_module")) or "Direct MRV",
                source_reference=_text(row.get("source_reference")),
                direct_input=value,
                provenance=f"Approved direct input under {permission.strategy_code}.",
            )

        # L4 bridge transformations ---------------------------------------
        bridge_candidates: dict[str, list[dict[str, Any]]] = {}
        for _, row in native_outputs.iterrows():
            native_variable = _text(row.get("native_variable"))
            native_value = _float_or_none(row.get("scenario_value"))
            use = _yes(row.get("use_in_completion"))
            if not native_variable or native_value is None or not use:
                continue
            if self.strict_approval and _text(row.get("approval_status")) != "Approved":
                issue("QA_UNAPPROVED_NATIVE_OUTPUT", "Warning", "WARN", "Native output was ignored because it is not approved.", variable=native_variable)
                rejected.append({"input_type": "native", "variable_name": native_variable, "value": native_value, "reason": "Not approved"})
                continue
            rule_id = _text(row.get("bridge_rule_id"))
            target = _text(row.get("proposed_target_mrv"))
            bridge = self._find_bridge(rule_id=rule_id, native_variable=native_variable, target_variable=target)
            if bridge is None:
                issue("QA_MISSING_BRIDGE", "Critical", "FAIL", "No active bridge rule was found for the native output.", variable=native_variable, action="Select or configure a documented bridge rule.")
                rejected.append({"input_type": "native", "variable_name": native_variable, "value": native_value, "reason": "Missing bridge"})
                continue
            target = _text(bridge["target_mrv_variable"])
            permission = self.resolve_permission(selected_strategies, target)
            if permission.blocks_change or not permission.allows("L4"):
                issue("QA_UNAUTHORIZED_BRIDGE", "Critical", "FAIL", f"Selected strategies do not permit an L4 bridge to {target}.", variable=target)
                rejected.append({"input_type": "native", "variable_name": native_variable, "value": native_value, "reason": "Bridge outside causal scope"})
                continue
            try:
                transformed = self._evaluate_bridge(bridge, native_value, base_values)
            except Exception as exc:
                issue("QA_BRIDGE_ERROR", "Critical", "FAIL", f"Bridge {bridge['bridge_rule_id']} failed: {exc}", variable=target)
                continue
            bridge_candidates.setdefault(target, []).append(
                {
                    "value": transformed,
                    "native_value": native_value,
                    "bridge": bridge,
                    "permission": permission,
                    "source_reference": _text(row.get("source_reference")),
                }
            )

        for target, candidates in bridge_candidates.items():
            if len(candidates) > 1:
                unique = {round(float(c["value"]), 12) for c in candidates}
                if len(unique) > 1:
                    issue("QA_BRIDGE_CONFLICT", "Critical", "FAIL", "Multiple bridge rules produced conflicting values.", variable=target, action="Select one approved source or an integrated model result.")
                    continue
            candidate = candidates[0]
            if state[target]["rule_level"] == "L1":
                issue("QA_BRIDGE_SUPERSEDED", "Warning", "WARN", "Bridge result was superseded by a direct L1 input.", variable=target)
                continue
            bridge = candidate["bridge"]
            permission = candidate["permission"]
            state[target].update(
                value=candidate["value"],
                rule_level="L4",
                rule_id=_text(bridge["bridge_rule_id"]),
                selected_strategy=permission.strategy_code,
                source_module=_text(bridge["source_module"]),
                source_reference=candidate["source_reference"],
                native_input=candidate["native_value"],
                provenance=f"Native output transformed by bridge {_text(bridge['bridge_rule_id'])}.",
            )

        # L5 approved assumptions ----------------------------------------
        for _, row in assumptions.iterrows():
            status = _text(row.get("status"))
            variable = _text(row.get("target_variable"))
            value = _float_or_none(row.get("proposed_value"))
            if status != "Approved" or not variable or value is None:
                continue
            strategy = _text(row.get("strategy_code"))
            if strategy and strategy not in selected_strategies:
                issue("QA_UNUSED_ASSUMPTION", "Warning", "WARN", "Approved assumption was not used because its strategy is not selected.", variable=variable)
                continue
            permission = self.resolve_permission(selected_strategies, variable)
            if permission.blocks_change or not permission.allows("L5"):
                issue("QA_UNAUTHORIZED_ASSUMPTION", "Critical", "FAIL", "The approved assumption is outside the selected causal domain.", variable=variable)
                rejected.append({"input_type": "assumption", "variable_name": variable, "value": value, "reason": "Outside causal scope"})
                continue
            if state[variable]["rule_level"] in {"L1", "L4"}:
                issue("QA_ASSUMPTION_SUPERSEDED", "Warning", "WARN", "L5 assumption was superseded by stronger evidence.", variable=variable)
                continue
            state[variable].update(
                value=value,
                rule_level="L5",
                rule_id=_text(row.get("assumption_id")) or "APPROVED_ASSUMPTION",
                selected_strategy=permission.strategy_code,
                source_module="Approved assumption",
                source_reference=_text(row.get("evidence_or_expert_source")),
                provenance=f"Approved L5 assumption: {_text(row.get('justification'))}",
            )

        # L3 baseline-intensity scaling ----------------------------------
        if _yes(scenario.get("allow_baseline_scaling")):
            driver = _text(scenario.get("scaling_driver_variable"))
            base_driver = _float_or_none(base_values.get(driver))
            scenario_driver = _float_or_none(state.get(driver, {}).get("value"))
            if base_driver in (None, 0) or scenario_driver is None:
                issue("QA_SCALING_DRIVER", "Critical", "FAIL", "Baseline scaling was enabled but the scaling driver is unavailable or zero.", variable=driver)
            else:
                ratio = scenario_driver / base_driver
                for _, row in direct_inputs.iterrows():
                    variable = _text(row.get("variable_name"))
                    if not variable or not _yes(row.get("allow_l3_scaling")) or variable not in state:
                        continue
                    if state[variable]["rule_level"] != "L6":
                        continue
                    permission = self.resolve_permission(selected_strategies, variable)
                    if permission.allows("L3") and not permission.blocks_change:
                        state[variable].update(
                            value=float(base_values[variable]) * ratio,
                            rule_level="L3",
                            rule_id=f"BASE_SCALE_BY_{driver}",
                            selected_strategy=permission.strategy_code,
                            source_module="Baseline scaling",
                            source_reference=driver,
                            provenance=f"BASE intensity scaled by {driver}; activity ratio={ratio:.8g}.",
                        )
        else:
            issue("QA_SCALING_SETTING", "Information", "PASS", "Baseline scaling is disabled.")

        # L2 MRV identities and aliases in topological order -------------
        rules_by_target = self.rules.set_index("target_variable", drop=False)
        for target in self._rule_order:
            if target not in state or target not in rules_by_target.index:
                continue
            # Stronger explicit evidence is never silently overwritten.
            if state[target]["rule_level"] in {"L1", "L4", "L5"}:
                continue
            rule = rules_by_target.loc[target]
            if isinstance(rule, pd.DataFrame):
                rule = rule.iloc[0]
            try:
                value = self._evaluate_mrv_rule(rule, state, base_values)
            except Exception as exc:
                issue("QA_MRV_RULE_ERROR", "Critical", "FAIL", f"MRV rule {_text(rule['rule_id'])} failed: {exc}", variable=target)
                continue
            state[target].update(
                value=value,
                rule_level="L2",
                rule_id=_text(rule["rule_id"]),
                selected_strategy="",
                source_module="MRV calculation",
                source_reference=";".join(_rule_sources(rule)),
                provenance=_text(rule.get("formula_description")) or "MRV identity recalculation.",
            )

        # A direct customer-acceptance index remains authoritative, but its
        # consistency with the canonical equal-weight formula is auditable.
        if "customer_acceptance_index" in direct_map:
            components = [
                _float_or_none(state.get(name, {}).get("value"))
                for name in (
                    "sustainable_sales_share",
                    "customer_survey_score",
                    "contract_renewal_rate",
                )
            ]
            if all(value is not None for value in components):
                calculated = sum(components) / 3.0 * 100.0
                difference = abs(direct_map["customer_acceptance_index"] - calculated)
                if difference > self.numerical_tolerance:
                    issue(
                        "QA_CUSTOMER_ACCEPTANCE_CONFLICT",
                        "Warning",
                        "WARN",
                        (
                            "Direct customer acceptance differs from the canonical "
                            f"equal-weight calculation by {difference:.12g} points; "
                            "the approved L1 value was retained."
                        ),
                        variable="customer_acceptance_index",
                    )

        # QA bounds and cross-variable consistency -----------------------
        for variable, record in state.items():
            value = _float_or_none(record.get("value"))
            if value is None or not math.isfinite(value):
                issue("QA_MISSING_COMPLETED_VALUE", "Critical", "FAIL", "No finite completed value is available.", variable=variable)
                continue
            meta = self.dictionary.loc[variable]
            min_value = _float_or_none(meta.get("min_value"))
            max_value = _float_or_none(meta.get("max_value"))
            if min_value is not None and value < min_value:
                issue("QA_MIN_BOUND", "Critical", "FAIL", f"Value {value} is below minimum {min_value}.", variable=variable)
            if max_value is not None and value > max_value:
                issue("QA_MAX_BOUND", "Critical", "FAIL", f"Value {value} is above maximum {max_value}.", variable=variable)

        self._relationship_check(state, "renewable_energy_kwh", "electricity_kwh", "Renewable energy exceeds total electricity.", issue)
        self._relationship_check(state, "waste_recovered_t", "waste_generated_t", "Recovered waste exceeds waste generated.", issue)
        self._relationship_check(state, "material_circular_t", "material_total_t", "Circular material exceeds total material.", issue)
        self._relationship_check(state, "dpp_valid_volume", "shipped_volume_total", "DPP-valid volume exceeds shipped volume.", issue)
        self._relationship_check(state, "mrv_points_active_valid", "mrv_points_required", "Valid MRV points exceed required MRV points.", issue)

        if state.get("ghg_total_s1s2", {}).get("rule_level") in {"L1", "L4"} and state.get("transport_ghg_tco2e", {}).get("rule_level") in {"L1", "L4"}:
            issue(
                "QA_GHG_BOUNDARY",
                "Warning",
                "WARN",
                "Both total Scope 1+2 GHG and transport GHG changed. Confirm whether transport is already included in the selected GHG boundary to avoid double counting.",
                variable="ghg_total_s1s2",
                action="Document the emissions boundary before KPI calculation.",
            )

        # Build outputs ---------------------------------------------------
        review_rows: list[dict[str, Any]] = []
        upload_rows: list[dict[str, Any]] = []
        for variable in common_variables:
            record = state[variable]
            issues = variable_issues.get(variable, [])
            qa_status = "FAIL" if any(sev == "Critical" for sev, _ in issues) else "WARN" if issues else "PASS"
            qa_message = " | ".join(message for _, message in issues)
            rule_level = record["rule_level"]
            source_system = self._source_system(rule_level, record.get("source_module"))
            comment = (
                f"rule={rule_level}; rule_id={record['rule_id']}; "
                f"strategy={record.get('selected_strategy') or 'none'}; "
                f"provenance={record.get('provenance', '')}"
            )
            review_rows.append(
                {
                    "variable_name": variable,
                    "unit": base_units.get(variable, _text(self.dictionary.loc[variable, "canonical_unit"])),
                    "base_value": _float_or_none(base_values.get(variable)),
                    "direct_input": record.get("direct_input"),
                    "native_input": record.get("native_input"),
                    "completed_value": record.get("value"),
                    "rule_level": rule_level,
                    "rule_id": record.get("rule_id"),
                    "selected_strategy": record.get("selected_strategy"),
                    "source_module": record.get("source_module"),
                    "source_reference": record.get("source_reference"),
                    "provenance": record.get("provenance"),
                    "qa_status": qa_status,
                    "qa_message": qa_message,
                    "timestamp": timestamp,
                    "scenario_code": scenario_code,
                    "engine_version": self.engine_version,
                    "rule_version": self.rule_version,
                }
            )
            upload_rows.append(
                {
                    "variable_name": variable,
                    "value": record.get("value"),
                    "unit": base_units.get(variable, _text(self.dictionary.loc[variable, "canonical_unit"])),
                    "timestamp": timestamp,
                    "scenario_code": scenario_code,
                    "source_system": source_system,
                    "comment": comment,
                }
            )

        completed_count = sum(_float_or_none(r["value"]) is not None for r in upload_rows)
        if completed_count == len(common_variables):
            issue("QA_COMPLETION_COUNT", "Critical", "PASS", f"All {len(common_variables)} common variables were completed.")
        else:
            issue("QA_COMPLETION_COUNT", "Critical", "FAIL", f"Only {completed_count} of {len(common_variables)} common variables were completed.")

        qa_df = pd.DataFrame(qa, columns=[
            "check_id", "severity", "status", "affected_variable", "message", "required_action", "scenario_code", "run_id"
        ])
        review_df = pd.DataFrame(review_rows)
        upload_df = pd.DataFrame(upload_rows, columns=UPLOAD_COLUMNS)
        rejected_df = pd.DataFrame(rejected, columns=["input_type", "variable_name", "value", "reason"])

        return CompletionResult(
            scenario_code=scenario_code,
            run_id=run_id,
            completion_review=review_df,
            software_upload=upload_df,
            qa_report=qa_df,
            rejected_inputs=rejected_df,
        )

    def resolve_permission(self, selected_strategies: Iterable[str], variable_name: str) -> Permission:
        if variable_name not in self.dictionary.index:
            return Permission("", variable_name, "Retain_BASE", "L6", 0, "Unknown variable.")
        group = _text(self.dictionary.loc[variable_name, "variable_group"])
        candidates: list[Permission] = []
        for strategy in selected_strategies:
            if not strategy:
                continue
            exact = self.overrides[
                (self.overrides["strategy_code"] == strategy)
                & (self.overrides["active"].map(_yes))
                & (self.overrides["variable_name"].isin([variable_name, "*"]))
            ]
            if not exact.empty:
                # Exact variable row wins over wildcard row.
                exact = exact.assign(_specific=(exact["variable_name"] == variable_name).astype(int))
                row = exact.sort_values(["_specific", "priority"], ascending=False).iloc[0]
                candidates.append(
                    Permission(
                        strategy,
                        variable_name,
                        _text(row["influence_status"]),
                        _text(row["permitted_rules"]),
                        int(row["priority"]),
                        _text(row["scientific_justification"]),
                    )
                )
                continue
            scope = self.scope[
                (self.scope["strategy_code"] == strategy)
                & (self.scope["variable_group"] == group)
            ]
            if not scope.empty:
                row = scope.sort_values("priority", ascending=False).iloc[0]
                candidates.append(
                    Permission(
                        strategy,
                        variable_name,
                        _text(row["influence_status"]),
                        _text(row["permitted_rules"]),
                        int(row["priority"]),
                        _text(row["scientific_interpretation"]),
                    )
                )
        if not candidates:
            return Permission("", variable_name, "Retain_BASE", "L6", 0, "No selected strategy supplied a causal permission.")
        return sorted(candidates, key=lambda p: p.priority, reverse=True)[0]

    def write_outputs_to_workbook(
        self,
        *,
        source_workbook: str | Path,
        destination_workbook: str | Path,
        result: CompletionResult,
    ) -> Path:
        """Write review, upload, and QA tables while preserving the template layout."""
        from copy import copy
        from openpyxl import load_workbook

        source = Path(source_workbook)
        destination = Path(destination_workbook)
        destination.parent.mkdir(parents=True, exist_ok=True)
        wb = load_workbook(source)

        self._write_dataframe_to_sheet(wb[OUTPUT_SHEETS["review"]], result.completion_review)
        self._write_dataframe_to_sheet(wb[OUTPUT_SHEETS["upload"]], result.software_upload)
        self._write_dataframe_to_sheet(wb[OUTPUT_SHEETS["qa"]], result.qa_report)
        wb.save(destination)
        return destination

    @staticmethod
    def _write_dataframe_to_sheet(worksheet: Any, dataframe: pd.DataFrame) -> None:
        from copy import copy

        header_row = 4
        start_row = 5
        headers = [worksheet.cell(header_row, col).value for col in range(1, worksheet.max_column + 1)]
        expected = [str(h) for h in headers if h is not None]
        missing = [c for c in expected if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Output DataFrame does not contain workbook columns: {missing}")

        # Clear old values but keep styles.
        for row in worksheet.iter_rows(min_row=start_row, max_row=max(worksheet.max_row, start_row)):
            for cell in row:
                cell.value = None

        template_row = start_row
        for ridx, record in enumerate(dataframe.loc[:, expected].itertuples(index=False, name=None), start=start_row):
            if ridx > worksheet.max_row:
                worksheet.insert_rows(ridx)
            for cidx, value in enumerate(record, start=1):
                cell = worksheet.cell(ridx, cidx)
                if ridx != template_row:
                    source_cell = worksheet.cell(template_row, cidx)
                    if source_cell.has_style:
                        cell._style = copy(source_cell._style)
                    if source_cell.number_format:
                        cell.number_format = source_cell.number_format
                    cell.alignment = copy(source_cell.alignment)
                    cell.fill = copy(source_cell.fill)
                    cell.font = copy(source_cell.font)
                    cell.border = copy(source_cell.border)
                cell.value = None if pd.isna(value) else value

    # ------------------------------------------------------------------
    # Configuration and calculations
    # ------------------------------------------------------------------
    def _read_config(self, filename: str) -> pd.DataFrame:
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing engine configuration: {path}")
        return _clean_frame(pd.read_csv(path))

    def _validate_config(self) -> None:
        required = {
            "dictionary": {"variable_name", "canonical_unit", "variable_group", "common_upload_variable", "user_input_allowed", "min_value", "max_value"},
            "scope": {"strategy_code", "variable_group", "influence_status", "permitted_rules", "priority"},
            "overrides": {"strategy_code", "variable_name", "influence_status", "permitted_rules", "priority", "active"},
            "rules": {"rule_id", "target_variable", "operation", "source_1", "source_2", "source_3", "parameter", "rule_status"},
            "bridges": {"bridge_rule_id", "source_module", "native_variable", "target_mrv_variable", "operation", "parameter", "base_multiplier_variable", "rule_status"},
        }
        frames = {
            "dictionary": self.dictionary,
            "scope": self.scope,
            "overrides": self.overrides,
            "rules": self.rules,
            "bridges": self.bridges,
        }
        for name, columns in required.items():
            missing = columns - set(frames[name].columns)
            if missing:
                raise ValueError(f"{name} configuration is missing columns: {sorted(missing)}")
        if self.dictionary["variable_name"].duplicated().any():
            raise ValueError("mrv_dictionary.csv contains duplicate variable_name values.")

    def _topological_rule_order(self) -> list[str]:
        active = self.rules[self.rules["rule_status"].astype(str).str.upper() == "ACTIVE"]
        targets = set(active["target_variable"])
        dependencies: dict[str, set[str]] = {}
        for _, row in active.iterrows():
            target = _text(row["target_variable"])
            dependencies[target] = {src for src in _rule_sources(row) if src in targets}
        order: list[str] = []
        remaining = {k: set(v) for k, v in dependencies.items()}
        while remaining:
            ready = sorted([target for target, deps in remaining.items() if not deps])
            if not ready:
                raise ValueError(f"Circular MRV-rule dependency detected: {remaining}")
            for target in ready:
                order.append(target)
                remaining.pop(target)
                for deps in remaining.values():
                    deps.discard(target)
        return order

    def _prepare_reference(self, reference: pd.DataFrame, reference_code: str) -> pd.DataFrame:
        required = set(UPLOAD_COLUMNS)
        missing = required - set(reference.columns)
        if missing:
            raise ValueError(f"Reference data is missing required CSV columns: {sorted(missing)}")
        ref = reference.copy()
        if "scenario_code" in ref.columns:
            filtered = ref[ref["scenario_code"].astype(str) == reference_code]
            if not filtered.empty:
                ref = filtered
        ref = ref.loc[:, UPLOAD_COLUMNS].copy()
        ref["value"] = pd.to_numeric(ref["value"], errors="coerce")
        ref = ref.dropna(subset=["variable_name", "value"])
        if ref["variable_name"].duplicated().any():
            duplicates = ref.loc[ref["variable_name"].duplicated(), "variable_name"].tolist()
            raise ValueError(f"Reference scenario contains duplicate variables: {duplicates}")
        return ref

    @staticmethod
    def _check_duplicates(frame: pd.DataFrame, key: str, label: str, issue: Callable[..., None]) -> None:
        if key not in frame.columns or frame.empty:
            return
        populated = frame[frame[key].notna() & (frame[key].astype(str).str.strip() != "")]
        duplicates = populated[populated[key].duplicated(keep=False)][key].astype(str).unique().tolist()
        if duplicates:
            issue("QA_DUPLICATE_INPUT", "Critical", "FAIL", f"Duplicate {label} keys: {duplicates}")
        else:
            issue(f"QA_DUPLICATE_{key.upper()}", "Critical", "PASS", f"No duplicate {label} keys were found.")

    def _find_bridge(self, *, rule_id: str, native_variable: str, target_variable: str) -> Optional[pd.Series]:
        active = self.bridges[self.bridges["rule_status"].astype(str).str.upper() == "ACTIVE"]
        if rule_id:
            match = active[active["bridge_rule_id"] == rule_id]
        else:
            match = active[active["native_variable"] == native_variable]
            if target_variable:
                match = match[match["target_mrv_variable"] == target_variable]
        if match.empty:
            return None
        if len(match) > 1 and not target_variable:
            return None
        return match.iloc[0]

    @staticmethod
    def _evaluate_bridge(bridge: pd.Series, native_value: float, base_values: Mapping[str, float]) -> float:
        operation = _text(bridge["operation"]).upper()
        parameter = _float_or_none(bridge.get("parameter"))
        if operation == "ALIAS":
            return float(native_value)
        if operation == "MULTIPLY_CONSTANT":
            if parameter is None:
                raise ValueError("MULTIPLY_CONSTANT requires parameter.")
            return float(native_value) * parameter
        if operation == "DIVIDE_CONSTANT":
            if parameter in (None, 0):
                raise ValueError("DIVIDE_CONSTANT requires a nonzero parameter.")
            return float(native_value) / parameter
        if operation == "MULTIPLY_BASE":
            base_variable = _text(bridge.get("base_multiplier_variable"))
            base_value = _float_or_none(base_values.get(base_variable))
            if base_value is None:
                raise ValueError(f"BASE multiplier variable {base_variable!r} is unavailable.")
            return float(native_value) * base_value
        raise ValueError(f"Unsupported bridge operation: {operation}")

    @staticmethod
    def _evaluate_mrv_rule(rule: pd.Series, state: Mapping[str, Mapping[str, Any]], base_values: Mapping[str, float]) -> float:
        operation = _text(rule["operation"]).upper()
        sources = _rule_sources(rule)
        values = []
        for source in sources:
            value = _float_or_none(state.get(source, {}).get("value"))
            if value is None:
                raise ValueError(f"Source variable {source!r} is unavailable.")
            values.append(value)
        parameter = _float_or_none(rule.get("parameter"))
        if operation == "ALIAS":
            return values[0]
        if operation == "SUM":
            return sum(values)
        if operation == "PRODUCT":
            return values[0] * values[1]
        if operation == "RATIO":
            return _safe_divide(values[0], values[1])
        if operation == "RATIO_PCT":
            return _safe_divide(values[0], values[1]) * 100.0
        if operation == "RATIO_X1000":
            return _safe_divide(values[0] * 1000.0, values[1])
        if operation == "RATIO_X1M":
            return _safe_divide(values[0] * 1_000_000.0, values[1])
        if operation == "RATIO_DIV24":
            return _safe_divide(values[0], values[1]) / 24.0
        if operation == "AVERAGE3_PCT":
            return sum(values[:3]) / 3.0 * 100.0
        if operation == "PRODUCT3_PCT":
            return values[0] * values[1] * values[2] * 100.0
        if operation == "MIN2":
            return min(values[0], values[1])
        if operation == "MULTIPLY_CONSTANT":
            if parameter is None:
                raise ValueError("MULTIPLY_CONSTANT requires parameter.")
            return values[0] * parameter
        if operation == "DIVIDE_CONSTANT":
            if parameter in (None, 0):
                raise ValueError("DIVIDE_CONSTANT requires a nonzero parameter.")
            return values[0] / parameter
        if operation == "SUM_RATIO_PCT":
            return _safe_divide(values[0] + values[1], values[2]) * 100.0
        if operation == "ROSI_PCT":
            return _safe_divide(values[0] - values[1], values[1]) * 100.0
        raise ValueError(f"Unsupported MRV operation: {operation}")

    @staticmethod
    def _relationship_check(
        state: Mapping[str, Mapping[str, Any]],
        numerator: str,
        denominator: str,
        message: str,
        issue: Callable[..., None],
    ) -> None:
        if numerator not in state or denominator not in state:
            return
        left = _float_or_none(state[numerator].get("value"))
        right = _float_or_none(state[denominator].get("value"))
        if left is None or right is None:
            return
        if left > right + 1e-9:
            issue("QA_RELATIONSHIP", "Critical", "FAIL", message, variable=numerator)

    @staticmethod
    def _source_system(rule_level: str, source_module: Any) -> str:
        if rule_level == "L1":
            module = _slug(_text(source_module))
            return f"direct_model_output_{module}" if module else "direct_mrv_input"
        return RULE_SOURCE_SYSTEM.get(rule_level, "scenario_completion_engine")


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(c).strip() for c in result.columns]
    result = result.dropna(how="all")
    return result


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"Required scenario field {field!r} is blank.")
    return text


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _yes(value: Any) -> bool:
    return _text(value).lower() in {"yes", "true", "1", "y", "approved", "active"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _selected_strategies(scenario: pd.Series) -> list[str]:
    values = [
        _text(scenario.get("primary_strategy")),
        _text(scenario.get("secondary_strategy_1")),
        _text(scenario.get("secondary_strategy_2")),
    ]
    return list(dict.fromkeys(v for v in values if v))


def _expand_rule_expression(expression: Any) -> set[str]:
    text = _text(expression).upper().replace(" ", "")
    result: set[str] = set()
    for start, end in re.findall(r"L([1-6])(?:-L?([1-6]))?", text):
        first = int(start)
        last = int(end) if end else first
        for number in range(min(first, last), max(first, last) + 1):
            result.add(f"L{number}")
    return result


def _rule_sources(rule: Mapping[str, Any]) -> list[str]:
    sources = []
    for column in ("source_1", "source_2", "source_3"):
        value = _text(rule.get(column))
        if value:
            sources.append(value)
    return sources


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ZeroDivisionError("MRV rule denominator is zero.")
    return numerator / denominator
