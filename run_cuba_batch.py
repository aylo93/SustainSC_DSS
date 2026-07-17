import argparse
from pathlib import Path
from batch_completion_engine import BatchScenarioCompletionEngine

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete a SustainSCM MRV batch workbook.")
    parser.add_argument("workbook", type=Path, help="Input .xlsx workbook")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    args = parser.parse_args()

    result = BatchScenarioCompletionEngine(args.config_dir).complete_batch_from_excel(args.workbook)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.export_combined_csv(args.output_dir / "ALL_SCENARIOS_COMPLETED_MRV.csv")
    result.export_scenario_csv_zip(args.output_dir / "COMPLETED_MRV_BY_SCENARIO.zip")
    result.completion_review.to_csv(args.output_dir / "COMPLETION_REVIEW.csv", index=False)
    result.qa_report.to_csv(args.output_dir / "QA_REPORT.csv", index=False)
    result.comparison_report.to_csv(args.output_dir / "CH7_COMPARISON.csv", index=False)

    critical = int(((result.qa_report.severity == "Critical") & (result.qa_report.status == "FAIL")).sum())
    warning = int((result.qa_report.status == "WARN").sum())
    print(f"Scenarios: {len(result.scenario_results)}")
    print(f"Completed common-MRV rows: {len(result.software_upload)}")
    print(f"Critical failures: {critical}")
    print(f"Warnings: {warning}")


if __name__ == "__main__":
    main()
