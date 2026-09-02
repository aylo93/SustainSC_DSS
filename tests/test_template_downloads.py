from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from sustainsc.mrv_schema_v2 import parse_mrv_workbook
from sustainsc.template_downloads import DPP_TEMPLATE, MRV_TEMPLATE, load_template_bytes


def test_downloadable_templates_are_intact_xlsx_assets() -> None:
    for template in (DPP_TEMPLATE, MRV_TEMPLATE):
        payload = load_template_bytes(template)
        assert payload.startswith(b"PK")
        assert sha256(payload).hexdigest() == template.sha256
        with ZipFile(BytesIO(payload)) as archive:
            assert archive.testzip() is None


def test_dpp_download_template_contains_required_input_blocks() -> None:
    workbook = pd.ExcelFile(BytesIO(load_template_bytes(DPP_TEMPLATE)))
    assert workbook.sheet_names == [
        "00_GUIDE",
        "01_PRODUCT_BATCHES",
        "02_TRACEABILITY_EVENTS",
        "03_DATA_DICTIONARY",
        "04_REFERENCE_LISTS",
    ]


def test_mrv_download_template_uses_the_supported_schema() -> None:
    parsed = parse_mrv_workbook(BytesIO(load_template_bytes(MRV_TEMPLATE)))
    assert parsed.schema.workbook_type == "MRV_V2"
    assert parsed.schema.version == "2.0"
    assert not parsed.schema.migration_required


def test_import_blocks_render_template_download_buttons() -> None:
    dpp_source = Path("kpi_dashboard.py").read_text(encoding="utf-8")
    mrv_source = Path("scenario_completion_page.py").read_text(encoding="utf-8")
    assert '"Download DPP & Traceability template"' in dpp_source
    assert 'key="download_dpp_traceability_template"' in dpp_source
    assert '"Download MRV input template"' in mrv_source
    assert 'key="download_mrv_input_template"' in mrv_source
