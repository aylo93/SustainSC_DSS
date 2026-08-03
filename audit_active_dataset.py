"""Read-only administrative audit of dataset/run isolation."""

from sustainsc.config import SessionLocal
from sustainsc.dataset_scope import audit_dataset, ensure_dataset_schema
from sqlalchemy import text


def main() -> None:
    ensure_dataset_schema()
    with SessionLocal() as session:
        audit = audit_dataset(session)
        rows = session.execute(
            text(
                """
                SELECT s.code AS scenario_code,
                  (SELECT COUNT(*) FROM sc_measurement m WHERE m.scenario_id=s.id) measurement_count,
                  (SELECT COUNT(*) FROM sc_kpi_result r WHERE r.scenario_id=s.id) raw_kpi_count,
                  (SELECT COUNT(*) FROM sc_kpi_normalized_result n WHERE n.scenario_id=s.id) normalized_kpi_count,
                  (SELECT COUNT(*) FROM sc_product_batch b WHERE b.scenario_id=s.id) batch_count,
                  (SELECT COUNT(*) FROM sc_traceability_event e
                    JOIN sc_product_batch b ON b.id=e.batch_id WHERE b.scenario_id=s.id) traceability_event_count,
                  CASE WHEN EXISTS (
                    SELECT 1 FROM sc_import_run_scenario l
                    JOIN sc_import_run ir ON ir.id=l.import_run_id
                    WHERE l.scenario_id=s.id AND ir.is_active=1
                  ) THEN 1 ELSE 0 END active_dataset_membership
                FROM sc_scenario s ORDER BY s.code
                """
            )
        ).mappings().all()
    print(f"active_run_id={audit.active_run_id}")
    print(f"active_scenarios={','.join(audit.active_scenarios)}")
    print(f"inactive_historical_scenarios={','.join(audit.inactive_scenarios)}")
    print(f"orphan_measurements={audit.orphan_measurements}")
    print(f"orphan_kpi_results={audit.orphan_kpi_results}")
    print(f"orphan_normalized_results={audit.orphan_normalized_results}")
    print(
        "scenario_code|measurement_count|raw_kpi_count|normalized_kpi_count|"
        "batch_count|traceability_event_count|active_dataset_membership"
    )
    for row in rows:
        print("|".join(str(row[key]) for key in row.keys()))


if __name__ == "__main__":
    main()
