# SustainSC DSS – KPI Dashboard (Prototype)

This repository contains a prototype Decision Support System (DSS) for Sustainable Supply Chain Management (SustainSCM).  
It includes:
- A SQLite database schema (SQLAlchemy)
- Example datasets (CSV)
- A KPI computation engine (26 core KPIs)
- A Streamlit dashboard with filters (Scenario / Dimension / Decision level / Flow) and scenario comparison.

## Project structure

- `sustainsc/` → Python package (models, KPI engine, dashboard utilities)
- `data/` → Example CSV datasets
  - `scenarios.csv`
  - `emission_factors.csv`
  - `cost_factors.csv`
  - `kpis.csv`
  - `measurements.csv`
- `create_db.py` → creates the SQLite schema
- `load_example_data.py` → loads CSV data into the DB
- `sustainsc/kpi_engine.py` → computes KPI results and stores them in DB
- `kpi_dashboard.py` → Streamlit dashboard

## Requirements

- Python 3.10+ (recommended)
- Dependencies in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
