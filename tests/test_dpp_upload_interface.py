from pathlib import Path


def test_guided_import_has_one_integrated_dpp_uploader():
    source = (Path(__file__).parents[1] / "kpi_dashboard.py").read_text(encoding="utf-8")

    assert source.count('"DPP & Traceability workbook"') == 1
    assert 'type=["xlsx"]' in source
    assert 'key="dpp_traceability_workbook_upload"' in source
    assert "Product batches CSV (optional)" not in source
    assert "Traceability events CSV (optional)" not in source
    assert "initial_product_batches_csv" not in source
    assert "initial_traceability_events_csv" not in source
    assert "dpp_workbook_bytes=workbook_bytes" in source
    assert "batches_file" not in source
    assert "events_file" not in source
