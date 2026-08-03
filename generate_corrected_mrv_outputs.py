"""Generate auditable corrected MRV and KPI regression CSV outputs.

The repository's current long-format measurements are used only as a fallback
fixture when the named Cuba workbook is unavailable.  DPP measures are rebuilt
from batch passports and all units are aligned to the active MRV dictionary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sustainsc.config import Base
from sustainsc.dpp_service import PASSPORT_TYPE, DPP_SCHEMA_VERSION, validate_dpp_core
from sustainsc.kpi_engine import Ctx, compute_formula
from sustainsc.models import Measurement, Scenario
from sustainsc.mrv_validation import MRV_COLUMNS, validate_completed_mrv

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "corrected_mrv"
DICTIONARY = ROOT / "config" / "mrv_dictionary.csv"
SOURCE = ROOT / "data" / "measurements.csv"

BASE_EXPECTED = {
    "EC1": 17.3575,
    "EC2": 20.00416086546,
    "EC3": 75.078,
    "EC4": -80.0,
    "EC5": 7.0,
    "EC6": 0.02,
    "EC7": 80.0,
    "EC8": 0.354166667,
    "E1": 749.5,
    "E2": 2.60243055556,
    "E3": 5.20833333333,
    "E4": 25.0,
    "E5": 0.06944444444,
    "E6": 40.0,
    "E7": 0.46319444444,
    "E8": 20.0,
    "E9": 0.20833333333,
    "S1": 2.4,
    "S2": 20.0,
    "S3": 1.0,
    "S4": 7.0,
    "S5": 66.6666666667,
    "S6": 55.0,
    "T1": 75.0,
    "T2": 50.0,
    "T3": 70.0,
    "T4": 80.0,
    "T5": 88.3333333333,
    "T6": 3.5,
    "T7": 66.6666666667,
}


def _dpp_summaries(scenarios: list[str]) -> dict[str, dict[str, float]]:
    batches = pd.read_csv(ROOT / "data" / "product_batches.csv")
    events = pd.read_csv(ROOT / "data" / "traceability_events.csv")
    summaries: dict[str, dict[str, float]] = {}
    for scenario_code in scenarios:
        selected = batches[batches["scenario_code"] == scenario_code].sort_values(
            "batch_code"
        )
        valid = 0
        volume = 0.0
        valid_volume = 0.0
        completeness = 0.0
        for batch in selected.itertuples():
            quantity = pd.to_numeric(batch.quantity, errors="coerce")
            batch_events = events[events["batch_code"] == batch.batch_code].sort_values(
                "timestamp"
            )
            passport = {
                "passport_type": PASSPORT_TYPE,
                "schema_version": DPP_SCHEMA_VERSION,
                "dpp_id": f"dpp:batch:{batch.batch_code}",
                "product_identity": {
                    "product_code": batch.product_code,
                    "batch_code": batch.batch_code,
                    "scenario_code": batch.scenario_code,
                    "origin_facility": batch.facility_name,
                    "production_date": batch.production_date,
                    "quantity": float(quantity) if pd.notna(quantity) else None,
                    "unit": batch.unit,
                },
                "traceability_events": [
                    {
                        "event_type": row.event_type,
                        "timestamp": row.timestamp,
                        "quantity": row.quantity,
                        "unit": row.unit,
                        "source_system": row.source_system,
                        "comment": row.comment,
                    }
                    for row in batch_events.itertuples()
                ],
            }
            result = validate_dpp_core(passport)
            positive_quantity = float(quantity) if pd.notna(quantity) and quantity > 0 else 0.0
            volume += positive_quantity
            completeness += result.completeness_score
            if result.is_valid:
                valid += 1
                valid_volume += positive_quantity
        count = len(selected)
        summaries[scenario_code] = {
            "dpp_batches_total": float(count),
            "dpp_batches_valid": float(valid),
            "dpp_volume": volume,
            "dpp_valid_volume": valid_volume,
            "dpp_completeness_average": completeness / count if count else 0.0,
        }
    return summaries


def _calculate_kpis(completed: pd.DataFrame) -> pd.DataFrame:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    catalog = pd.read_csv(ROOT / "data" / "kpis.csv")
    rows: list[dict[str, object]] = []
    with Session(engine) as session:
        for scenario_code, frame in completed.groupby("scenario_code", sort=True):
            scenario = Scenario(code=scenario_code, name=scenario_code)
            session.add(scenario)
            session.flush()
            for row in frame.itertuples():
                session.add(
                    Measurement(
                        variable_name=row.variable_name,
                        value=float(row.value),
                        unit=row.unit,
                        timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                        scenario_id=scenario.id,
                        source_system=row.source_system,
                        comment=row.comment,
                    )
                )
            session.flush()
            ctx = Ctx(session=session, scenario_id=scenario.id, cache={})
            for kpi in catalog.itertuples():
                rows.append(
                    {
                        "scenario_code": scenario_code,
                        "kpi_code": kpi.code,
                        "kpi_name": kpi.name,
                        "formula_id": kpi.formula_id,
                        "unit": kpi.unit,
                        "value": compute_formula(ctx, kpi.formula_id),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenario_dir = output / "scenarios"
    scenario_dir.mkdir(exist_ok=True)

    dictionary = pd.read_csv(DICTIONARY)
    common = dictionary[
        dictionary["common_upload_variable"].astype(str).str.lower() == "yes"
    ].copy()
    required = set(common["variable_name"])
    units = common.set_index("variable_name")["canonical_unit"].to_dict()

    if args.workbook:
        from batch_completion_engine import BatchScenarioCompletionEngine

        batch_result = BatchScenarioCompletionEngine(
            ROOT / "config"
        ).complete_batch_from_excel(args.workbook)
        source = batch_result.software_upload.copy()
        source_label = str(args.workbook.resolve())
    else:
        source = pd.read_csv(SOURCE, sep="\t")
        source_label = str(SOURCE.relative_to(ROOT))
    source["_input_order"] = range(len(source))
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    source = source[source["variable_name"].isin(required)].copy()
    source = source.sort_values(
        ["scenario_code", "variable_name", "timestamp", "_input_order"]
    )
    duplicate_count = int(
        source.duplicated(["scenario_code", "variable_name"], keep=False).sum()
    )
    completed = source.drop_duplicates(
        ["scenario_code", "variable_name"], keep="last"
    ).copy()

    original_units = completed["unit"].copy()
    completed["unit"] = completed["variable_name"].map(units)
    unit_corrections = int((original_units != completed["unit"]).sum())
    summaries = _dpp_summaries(sorted(completed["scenario_code"].unique()))
    for index, row in completed.iterrows():
        scenario_summary = summaries[row["scenario_code"]]
        if row["variable_name"] in {"dpp_volume", "dpp_valid_volume"}:
            completed.at[index, "value"] = scenario_summary[row["variable_name"]]
            completed.at[index, "source_system"] = "dpp_generation_module"
            completed.at[index, "comment"] = (
                f"DPP schema {DPP_SCHEMA_VERSION}; prototype validation; "
                f"batches={int(scenario_summary['dpp_batches_total'])}; "
                f"valid={int(scenario_summary['dpp_batches_valid'])}"
            )
        elif row["variable_name"] == "dpp_coverage":
            shipped = completed[
                (completed["scenario_code"] == row["scenario_code"])
                & (completed["variable_name"] == "shipped_volume_total")
            ]["value"]
            denominator = float(shipped.iloc[0]) if not shipped.empty else 0.0
            completed.at[index, "value"] = (
                scenario_summary["dpp_valid_volume"] / denominator * 100
                if denominator > 0
                else 0.0
            )
            completed.at[index, "source_system"] = "mrv_dependency_engine"
            completed.at[index, "comment"] = (
                "dpp_valid_volume / shipped_volume_total * 100; "
                "depends on validated DPP batches"
            )

    completed = completed[MRV_COLUMNS].sort_values(
        ["scenario_code", "variable_name"]
    )
    validation = validate_completed_mrv(
        completed, dictionary_path=DICTIONARY, raise_on_error=True
    )
    completed.to_csv(output / "SustainSCM_CORRECTED_COMPLETED_MRV.csv", index=False)
    for scenario_code, frame in completed.groupby("scenario_code", sort=True):
        frame.to_csv(scenario_dir / f"{scenario_code}_completed_mrv.csv", index=False)

    provenance = completed.copy()
    provenance.insert(
        0, "run_id", f"corrected-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    provenance["rule_level"] = provenance["source_system"].map(
        lambda source_name: "DPP" if source_name == "dpp_generation_module" else "retained"
    )
    provenance.to_csv(output / "COMPLETION_PROVENANCE.csv", index=False)

    qa = validation.report.copy()
    qa = pd.concat(
        [
            qa,
            pd.DataFrame(
                [
                    {
                        "scenario_code": "ALL",
                        "finding": (
                            "source_fixture_workbook"
                            if args.workbook
                            else "source_fixture_fallback"
                        ),
                        "variable_name": source_label,
                        "severity": "Pass" if args.workbook else "Warning",
                    },
                    {
                        "scenario_code": "ALL",
                        "finding": "input_duplicate_rows_resolved",
                        "variable_name": str(duplicate_count),
                        "severity": "Warning" if duplicate_count else "Pass",
                    },
                    {
                        "scenario_code": "ALL",
                        "finding": "canonical_unit_corrections",
                        "variable_name": str(unit_corrections),
                        "severity": "Warning" if unit_corrections else "Pass",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    qa.to_csv(output / "QA_REPORT.csv", index=False)

    comparison_source = (
        pd.read_csv(args.reference)
        if args.reference
        else source.copy()
    )
    comparison_source = comparison_source[
        comparison_source["variable_name"].isin(required)
    ].copy()
    source_compare = comparison_source.drop_duplicates(
        ["scenario_code", "variable_name"], keep="last"
    )[["scenario_code", "variable_name", "value", "unit"]].rename(
        columns={"value": "source_value", "unit": "source_unit"}
    )
    comparison = completed.merge(
        source_compare, on=["scenario_code", "variable_name"], how="left"
    )
    comparison = comparison.rename(
        columns={"value": "corrected_value", "unit": "corrected_unit"}
    )
    comparison["difference"] = (
        pd.to_numeric(comparison["corrected_value"])
        - pd.to_numeric(comparison["source_value"])
    )
    comparison["classification"] = "exact_match"
    comparison.loc[
        comparison["source_unit"] != comparison["corrected_unit"], "classification"
    ] = "canonical_unit_correction"
    comparison.loc[
        comparison["difference"].abs() > 1e-9, "classification"
    ] = "scientific_rule_correction"
    comparison["reference_source"] = (
        str(args.reference.resolve()) if args.reference else source_label
    )
    comparison.to_csv(
        output
        / ("CH7_COMPARISON.csv" if args.reference else "CH7_COMPARISON_FALLBACK.csv"),
        index=False,
    )

    kpis = _calculate_kpis(completed)
    kpis.to_csv(output / "ALL_SCENARIOS_30_KPI_VALUES.csv", index=False)
    base = kpis[kpis["scenario_code"] == "BASE"].copy()
    base["expected_value"] = base["kpi_code"].map(BASE_EXPECTED)
    base["difference"] = base["value"] - base["expected_value"]
    base["status"] = base.apply(
        lambda row: (
            "match"
            if abs(row["difference"]) <= 1e-6
            else "rounding_only"
            if abs(row["difference"]) <= 1e-3
            else "scientific_rule_correction"
            if row["kpi_code"] == "T4"
            else "unresolved_mismatch"
        ),
        axis=1,
    )
    base.to_csv(output / "BASE_30_KPI_COMPARISON.csv", index=False)
    print(
        f"Generated {len(completed)} rows for {validation.scenario_count} scenarios; "
        f"{len(kpis)} KPI values; outputs={output}"
    )


if __name__ == "__main__":
    main()
