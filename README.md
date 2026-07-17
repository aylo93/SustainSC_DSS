# SustainSC DSS

Decision Support System for Sustainable Supply Chain Management. The Streamlit
application combines MRV scenario completion, KPI calculation and normalization,
composite sustainability indices, MCDA comparison, traceability and DPP export.

## Run locally

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run kpi_dashboard.py
```

The application creates and seeds its SQLite database automatically when needed.
Set `SUSTAINSC_DB_URL` to use a different SQLAlchemy database URL.

The first screen remains empty until the MRV Excel template has passed through
the completion engines and its completed scenarios are imported. The generated
CSV is an output of that process, not the primary application input.

To delete all operational data and return to that initial screen:

```bash
python create_db.py --reset
```

This command permanently removes the current local database contents and
recreates only the empty schema. The KPI catalog and calculation factors are
loaded automatically when Streamlit starts; scenarios and measurements are not.

## Main structure

- `kpi_dashboard.py`: Streamlit entry point.
- `scenario_completion_engine.py`: auditable causal MRV completion engine.
- `batch_completion_engine.py`: multi-scenario workbook orchestrator.
- `scenario_completion_page.py`: workbook validation and import UI.
- `config/`: MRV dictionary, strategy scope and completion rules.
- `sustainsc/`: database models and KPI, normalization and DPP services.
- `data/`: demo and reference datasets loaded by the application.

## MRV workbook import

Open **Complete and import MRV scenario workbook** in the application, upload an
`.xlsx` workbook, review QA results and import it. Critical QA failures disable
the import. A successful import replaces measurements for the included scenarios
and recalculates the complete KPI pipeline.

The expected workbook sheets are:

- `01_SCENARIOS`
- `02_DIRECT_MRV_INPUT`
- `03_NATIVE_OUTPUTS`
- `04_APPROVED_ASSUMPTIONS`
- `05_REFERENCE_BASE`
- `11_EXPECTED_CH7_MRV`

For command-line batch processing:

```bash
python run_cuba_batch.py path/to/workbook.xlsx --output-dir generated
```

## Data maintenance

- `create_db.py`: ensure the schema exists.
- `load_example_data.py`: reload the core demo catalog and measurements.
- `seed_dpp_demo.py`: seed the traceability/DPP demo.
- `load_measurements_only.py`: reload only measurements.
- `load_product_batches.py` and `load_traceability_events.py`: import DPP data.

Generated databases, Python caches, local editor settings and batch output files
are excluded from version control.
