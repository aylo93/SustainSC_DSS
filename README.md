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
- The guided Streamlit workflow imports DPP data from one integrated XLSX workbook
  containing `01_PRODUCT_BATCHES` and `02_TRACEABILITY_EVENTS`.
- `load_product_batches.py` and `load_traceability_events.py` are legacy command-line
  compatibility utilities; they are not used by the normal Streamlit commit path.

Generated databases, Python caches, local editor settings and batch output files
are excluded from version control.

## Two-phase DPP workflow

The DPP module implements a **DPP-ready batch-level prototype**:

1. Build and validate KPI-independent DPP cores.
2. Summarize valid batch quantities into `dpp_volume` and
   `dpp_valid_volume` MRV measurements.
3. Recalculate KPIs from the finalized MRV layer.
4. Enrich passports with product-scenario raw KPI results and scenario-level
   normalized decision-support results.

Normalized scores and traffic lights are not represented as physical batch
properties. They retain their scenario decision-support scope.

`run_scenario_pipeline_with_dpp` supports a session-aware KPI runner so all
steps can share one transaction. The current legacy `run_full_pipeline`
function creates independent sessions; callers using it must first commit the
returned DPP MRV records and then run that global pipeline. No new DPP
persistence table was added because the repository has no migration framework
and the existing `ProductPassport` model is only a link/metadata record.

For tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Streamlit workflow

Data are imported only from the initial **Data Import** view. The MRV workbook
is required; batch and traceability CSV files are optional and are processed in
that same view. After import, targeted dashboard caches are invalidated and the
application reruns against committed database records.

The sidebar filters apply only to the detailed KPI table. When a restrictive
dimension, decision-level or flow filter is active, integrated indices,
sensitivity and ranking sections are skipped because they require the complete
dataset. DPP and traceability appear after the integrated analyses and are
generated directly from the currently imported batch records.

## User-interface design system

Reusable presentation code lives in `sustainsc/ui/`:

- `theme.py` defines semantic colors, spacing, surfaces and accessible CSS.
- `components.py` contains page headers, workflow progress, data-status panels,
  empty states and filter summaries.
- `chart_theme.py` registers the shared `sustainscm` Plotly template.
- `assets/supply_chain.svg` is an original lightweight vector illustration
  created for this repository; it has no external licensing dependency.

The application uses a polished light theme configured in
`.streamlit/config.toml`. A custom runtime dark-mode switch was intentionally
not added because it would conflict with Streamlit widget theming and reruns.

## MCDA56 Decision-Evidence Layer

The ASCA page retrieves a compact, read-only subset of the completed Romanian
56-anchor experiment: eight structured experimental archetypes, BASE plus six
intervention strategies, and the common 30-KPI SustainSCM architecture. BASE is
the within-archetype benchmark and is excluded from the competitive rankings.

The evidence layer displays the stored within-archetype WSM and TOPSIS ranks,
four dimension indices and global geometric index, cross-archetype rank
statistics, deterministic and bounded-random weight sensitivity, completion
sensitivity, and VSM-C/MCDA agreement. Runtime CSV files live in
`data/asca/mcda56/`; the full raw and normalized 1,680-observation tables remain
in the external reproducibility package and are not loaded on Streamlit reruns.

ASCA does not recalculate KPIs, normalization, WSM or TOPSIS and does not modify
the parent-model, metamodel, applicability-domain or routing results. It
configures scenarios, checks applicability, retrieves the completed SustainSCM
evidence and applies deterministic wording rules. The archetypes are structured
experimental anchors, not statistical sector averages or empirical company
observations. Social and technological completion bridges remain synthetic
design quantities; the displayed completion-sensitivity tests bound reliance on
those assumptions but do not turn them into observations.
