"""
Adapter for reading simulation exports (CSV/XLSX) and converting to MRV long format.

Provides three main functions:
1. read_any_file(path) → reads CSV/XLSX
2. normalize_any_export(df_raw, mapping_csv_path, ...) → converts to MRV long format
3. import_long_mrv_csv(path, scenario_prefix, ...) → validates and writes directly
4. upsert_measurements(session, df_mrv) → writes to sc_measurement + creates scenarios
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from .models import Scenario, Measurement
from .config import SessionLocal


# ============================================================================
# 1) READ ANY FILE (CSV/XLSX)
# ============================================================================

def read_any_file(path: Path | str) -> pd.DataFrame:
    """
    Read CSV or XLSX file into DataFrame.
    
    Args:
        path: Path to CSV or XLSX file
    
    Returns:
        DataFrame with raw data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not recognized
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    suffix = path.suffix.lower()
    
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            # Try to read first sheet; if multiple sheets, let user specify
            df = pd.read_excel(path, sheet_name=0)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Expected .csv, .xlsx, or .xls")
        
        print(f"✅ Read {len(df)} rows from {path.name}")
        return df
    
    except Exception as e:
        raise ValueError(f"Error reading file {path.name}: {e}")


# ============================================================================
# 2) NORMALIZE TO MRV LONG FORMAT
# ============================================================================

def normalize_any_export(
    df_raw: pd.DataFrame,
    mapping_csv_path: Optional[Path | str] = None,
    default_scenario_code: str = "SIM_ALX",
    source_system: str = "AnyLogistix/AnyLogic",
    default_unit: str = "unit",
    default_timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert raw export to MRV long format.
    
    Expected MRV structure (output):
    - scenario_code: str (e.g., "BASE", "S1", "ALX_001")
    - variable_name: str (e.g., "total_cost_eur", "co2_emissions_tco2e")
    - value: float
    - unit: str (e.g., "EUR", "tCO2e")
    - timestamp: datetime (optional)
    - source_system: str
    - comment: str (optional)
    
    Args:
        df_raw: Raw DataFrame from read_any_file()
        mapping_csv_path: Optional path to column mapping CSV.
                         Format: raw_column, variable_name, unit
        default_scenario_code: Scenario code if not in data (e.g., "SIM_ALX")
        source_system: Source system label (e.g., "AnyLogistix/AnyLogic")
        default_unit: Default unit if not found in mapping
        default_timestamp: Default timestamp if not in data (ISO format)
    
    Returns:
        DataFrame in MRV long format with columns:
        [scenario_code, variable_name, value, unit, timestamp, source_system, comment]
    
    Raises:
        ValueError: If required columns missing or invalid data
    """
    
    df = df_raw.copy()
    
    # Infer scenario_code from column if exists, else use default
    if "scenario_code" not in df.columns:
        # Try to find scenario column by name pattern
        scenario_cols = [c for c in df.columns if "scenario" in c.lower()]
        if scenario_cols:
            df.rename(columns={scenario_cols[0]: "scenario_code"}, inplace=True)
        else:
            df["scenario_code"] = default_scenario_code
    
    # Fill missing scenario_code
    df["scenario_code"] = df["scenario_code"].fillna(default_scenario_code)
    df["scenario_code"] = df["scenario_code"].astype(str).str.strip()
    
    # Load mapping if provided
    mapping = {}
    if mapping_csv_path:
        mapping_path = Path(mapping_csv_path)
        if mapping_path.exists():
            try:
                mapping_df = pd.read_csv(mapping_path)
                # Expected columns: raw_column, variable_name, unit
                for _, row in mapping_df.iterrows():
                    raw_col = str(row.get("raw_column", "")).strip()
                    var_name = str(row.get("variable_name", "")).strip()
                    unit = str(row.get("unit", default_unit)).strip()
                    if raw_col and var_name:
                        mapping[raw_col] = {"variable_name": var_name, "unit": unit}
                print(f"✅ Loaded mapping from {mapping_path.name}: {len(mapping)} columns")
            except Exception as e:
                print(f"[WARN] Could not load mapping CSV: {e}")
    
    # Convert to long format
    if "variable_name" not in df.columns:
        # If not already in long format, melt numeric columns
        id_cols = ["scenario_code"]
        if "timestamp" in df.columns:
            id_cols.append("timestamp")
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        if not numeric_cols:
            raise ValueError("No numeric columns found to melt into long format")
        
        df = df.melt(
            id_vars=id_cols,
            value_vars=numeric_cols,
            var_name="variable_name",
            value_name="value"
        )
    
    # Apply mapping (rename variable_name and add unit)
    df["unit"] = default_unit
    for raw_col, map_info in mapping.items():
        mask = df["variable_name"] == raw_col
        if mask.any():
            df.loc[mask, "variable_name"] = map_info["variable_name"]
            df.loc[mask, "unit"] = map_info.get("unit", default_unit)
    
    # Handle timestamp
    if "timestamp" not in df.columns:
        if default_timestamp:
            df["timestamp"] = pd.to_datetime(default_timestamp, errors="coerce")
        else:
            df["timestamp"] = datetime.now()
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Add metadata
    df["source_system"] = source_system
    if "comment" not in df.columns:
        df["comment"] = None
    
    # Select and order columns
    output_cols = ["scenario_code", "variable_name", "value", "unit", "timestamp", "source_system", "comment"]
    df = df[output_cols].copy()
    
    # Clean up
    df = df.dropna(subset=["scenario_code", "variable_name", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    
    print(f"✅ Normalized to MRV long format: {len(df)} measurements")
    return df


# ============================================================================
# 3) UPSERT MEASUREMENTS (writes to DB + creates scenarios)
# ============================================================================

def upsert_measurements(
    session: Session,
    df_mrv: pd.DataFrame,
    auto_create_scenario: bool = True,
) -> int:
    """
    Upsert measurements from MRV DataFrame into sc_measurement table.
    Also creates scenarios if they don't exist.
    
    Args:
        session: SQLAlchemy session
        df_mrv: DataFrame in MRV long format
                [scenario_code, variable_name, value, unit, timestamp, source_system, comment]
        auto_create_scenario: If True, create scenario if not exists
    
    Returns:
        Number of measurements written/updated
    
    Raises:
        ValueError: If required columns missing
    """
    
    required_cols = ["scenario_code", "variable_name", "value", "unit"]
    missing = [c for c in required_cols if c not in df_mrv.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    written = 0
    
    for _, row in df_mrv.iterrows():
        scenario_code = str(row["scenario_code"]).strip()
        variable_name = str(row["variable_name"]).strip()
        value = float(row["value"])
        unit = str(row.get("unit", "unit")).strip()
        timestamp = row.get("timestamp", datetime.now())
        source_system = str(row.get("source_system", "Unknown")).strip()
        comment = row.get("comment", None)
        
        # Ensure scenario exists
        scenario = session.query(Scenario).filter_by(code=scenario_code).first()
        if not scenario:
            if auto_create_scenario:
                scenario = Scenario(
                    code=scenario_code,
                    name=f"Scenario {scenario_code}",
                    description=f"Auto-created from {source_system}"
                )
                session.add(scenario)
                session.flush()
                print(f"  📌 Created scenario: {scenario_code}")
            else:
                print(f"[WARN] Scenario {scenario_code} not found, skipping measurement")
                continue
        
        # Upsert measurement
        existing = session.query(Measurement).filter_by(
            scenario_id=scenario.id,
            variable_name=variable_name,
            timestamp=timestamp
        ).first()
        
        if existing:
            existing.value = value
            existing.unit = unit
            existing.source_system = source_system
            existing.comment = comment
        else:
            measurement = Measurement(
                scenario_id=scenario.id,
                variable_name=variable_name,
                value=value,
                unit=unit,
                timestamp=timestamp,
                source_system=source_system,
                comment=comment
            )
            session.add(measurement)
        
        written += 1
    
    try:
        session.commit()
        print(f"✅ Upserted {written} measurements")
    except Exception as e:
        session.rollback()
        raise ValueError(f"Error writing measurements: {e}")
    
    return written


# ============================================================================
# 4) IMPORT LONG MRV CSV (validates and writes directly)
# ============================================================================

def import_long_mrv_csv(
    path: Path | str,
    scenario_prefix: str = "MRV_",
    auto_create_scenario: bool = True,
    session: Optional[Session] = None,
) -> int:
    """
    Import CSV already in MRV long format directly into database.
    
    Expected CSV columns:
    - scenario_code: str
    - variable_name: str
    - value: float
    - unit: str
    - timestamp: datetime (optional)
    - source_system: str (optional, defaults to "CSV Import")
    - comment: str (optional)
    
    Args:
        path: Path to MRV CSV file
        scenario_prefix: Optional prefix to add to scenario codes
        auto_create_scenario: If True, create scenario if not exists
        session: Optional SQLAlchemy session. If None, creates new session.
    
    Returns:
        Number of measurements written
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV structure invalid
    """
    
    path = Path(path)
    
    # Read CSV
    df_mrv = pd.read_csv(path)
    
    # Validate structure
    required = ["scenario_code", "variable_name", "value", "unit"]
    missing = [c for c in required if c not in df_mrv.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    # Add scenario prefix if not already present
    if scenario_prefix and scenario_prefix != "":
        df_mrv["scenario_code"] = df_mrv["scenario_code"].apply(
            lambda x: f"{scenario_prefix}{x}" if not str(x).startswith(scenario_prefix) else x
        )
    
    # Default source_system
    if "source_system" not in df_mrv.columns:
        df_mrv["source_system"] = "CSV Import"
    
    # Parse timestamp if exists
    if "timestamp" in df_mrv.columns:
        df_mrv["timestamp"] = pd.to_datetime(df_mrv["timestamp"], errors="coerce")
    
    # Create session if not provided
    if session is None:
        session = SessionLocal()
        own_session = True
    else:
        own_session = False
    
    try:
        written = upsert_measurements(session, df_mrv, auto_create_scenario=auto_create_scenario)
        return written
    finally:
        if own_session:
            session.close()


# ============================================================================
# HELPER: Validate MRV Format
# ============================================================================

def validate_mrv_format(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate DataFrame is in valid MRV long format.
    
    Returns:
        (is_valid: bool, message: str)
    """
    required_cols = ["scenario_code", "variable_name", "value", "unit"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return False, f"Missing columns: {missing}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for NaN in critical columns
    if df[required_cols].isna().any().any():
        return False, "Found NaN in required columns"
    
    # Check value is numeric
    try:
        pd.to_numeric(df["value"], errors="raise")
    except Exception as e:
        return False, f"Column 'value' not numeric: {e}"
    
    return True, "✅ Valid MRV format"


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sim_export_adapter.py <csv_or_xlsx_file> [--mapping <mapping_csv>]")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    mapping_path = None
    
    if "--mapping" in sys.argv:
        mapping_idx = sys.argv.index("--mapping")
        if mapping_idx + 1 < len(sys.argv):
            mapping_path = Path(sys.argv[mapping_idx + 1])
    
    # Read → Normalize → Upsert
    try:
        print(f"📖 Reading {file_path.name}...")
        df_raw = read_any_file(file_path)
        
        print(f"🔄 Normalizing to MRV format...")
        df_mrv = normalize_any_export(
            df_raw,
            mapping_csv_path=mapping_path,
            default_scenario_code="SIM_ALX"
        )
        
        # Validate
        is_valid, msg = validate_mrv_format(df_mrv)
        print(f"{'✅' if is_valid else '❌'} {msg}")
        
        if not is_valid:
            sys.exit(1)
        
        print(f"💾 Upserting measurements...")
        session = SessionLocal()
        written = upsert_measurements(session, df_mrv)
        session.close()
        
        print(f"✅ Done! {written} measurements written.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
