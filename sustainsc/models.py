# sustainsc/models.py
from __future__ import annotations

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .config import Base


class ImportRun(Base):
    __tablename__ = "sc_import_run"
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(255), nullable=False)
    source_filename = Column(String(500), nullable=True)
    import_timestamp = Column(DateTime, nullable=False)
    status = Column(String(30), nullable=False, default="importing")
    reference_scenario_code = Column(String(50), nullable=True)
    scenario_count = Column(Integer, nullable=False, default=0)
    measurement_count = Column(Integer, nullable=False, default=0)
    checksum = Column(String(64), nullable=True)
    factor_set_id = Column(String(100), nullable=True)
    last_kpi_calculation = Column(DateTime, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False, index=True)

    scenario_links = relationship(
        "ImportRunScenario", back_populates="import_run", cascade="all, delete-orphan"
    )
    measurements = relationship("Measurement", back_populates="import_run")
    kpi_results = relationship("KPIResult", back_populates="import_run")
    normalized_results = relationship("KPINormalizedResult", back_populates="import_run")


class ImportRunScenario(Base):
    __tablename__ = "sc_import_run_scenario"
    __table_args__ = (
        UniqueConstraint("import_run_id", "scenario_id", name="uq_import_run_scenario"),
    )
    id = Column(Integer, primary_key=True)
    import_run_id = Column(
        Integer, ForeignKey("sc_import_run.id"), nullable=False, index=True
    )
    scenario_id = Column(Integer, ForeignKey("sc_scenario.id"), nullable=False, index=True)

    import_run = relationship("ImportRun", back_populates="scenario_links")
    scenario = relationship("Scenario", back_populates="import_run_links")


class Scenario(Base):
    __tablename__ = "sc_scenario"
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    measurements = relationship("Measurement", back_populates="scenario", cascade="all, delete-orphan")
    kpi_results = relationship("KPIResult", back_populates="scenario", cascade="all, delete-orphan")
    import_run_links = relationship(
        "ImportRunScenario", back_populates="scenario", cascade="all, delete-orphan"
    )


# ---------- DPP / Master data ----------
class Product(Base):
    __tablename__ = "sc_product"
    id = Column(Integer, primary_key=True)
    code = Column(String(80), unique=True, nullable=False)  # e.g., AGG_0_4, AGG_4_8
    name = Column(String(255), nullable=False)
    fu_unit = Column(String(50), nullable=True)  # e.g., "t" (toneladas)

    # DPP “handle” (link o identificador)
    dpp_ref = Column(String(255), nullable=True)

    measurements = relationship("Measurement", back_populates="product")
    kpi_results = relationship("KPIResult", back_populates="product")


class ProductPassport(Base):
    """
    DPP / “pasaporte” (opcional): permite guardar metadata por producto/lote.
    """
    __tablename__ = "sc_product_passport"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("sc_product.id"), nullable=False)

    batch_code = Column(String(120), nullable=True)
    passport_url = Column(String(500), nullable=True)
    certificate_ref = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)

    product = relationship("Product")


class Facility(Base):
    __tablename__ = "sc_facility"
    id = Column(Integer, primary_key=True)
    code = Column(String(80), unique=True, nullable=False)  # e.g., PLANT_A, PLANT_B
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)  # city/region
    facility_type = Column(String(80), nullable=True)  # quarry/plant/warehouse

    processes = relationship("Process", back_populates="facility")
    measurements = relationship("Measurement", back_populates="facility")
    kpi_results = relationship("KPIResult", back_populates="facility")


class Process(Base):
    __tablename__ = "sc_process"
    id = Column(Integer, primary_key=True)
    code = Column(String(80), unique=True, nullable=False)  # e.g., CRUSHING, SCREENING
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)

    facility = relationship("Facility", back_populates="processes")
    measurements = relationship("Measurement", back_populates="process")


class TransportLeg(Base):
    __tablename__ = "sc_transport_leg"
    id = Column(Integer, primary_key=True)
    code = Column(String(80), unique=True, nullable=False)  # e.g., A_TO_MARKET, A_TO_B
    name = Column(String(255), nullable=True)

    from_facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)
    to_facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)

    mode = Column(String(50), nullable=True)  # truck/rail/ship
    distance_km = Column(Float, nullable=True)

    measurements = relationship("Measurement", back_populates="transport_leg")


# ---------- KPI catalog ----------
class KPI(Base):
    __tablename__ = "sc_kpi"
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    dimension = Column(String(50), nullable=False)
    decision_level = Column(String(50), nullable=False)
    flow = Column(String(50), nullable=False)
    unit = Column(String(50), nullable=True)
    is_benefit = Column(Boolean, nullable=False, default=False)
    formula_id = Column(String(100), nullable=False)
    protocol_notes = Column(Text, nullable=True)

    results = relationship("KPIResult", back_populates="kpi")


# ---------- MRV measurement ----------
class Measurement(Base):
    __tablename__ = "sc_measurement"
    id = Column(Integer, primary_key=True)

    variable_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    timestamp = Column(DateTime, nullable=False)

    scenario_id = Column(Integer, ForeignKey("sc_scenario.id"), nullable=True)
    import_run_id = Column(Integer, ForeignKey("sc_import_run.id"), nullable=True, index=True)

    # Contexto DPP / trazabilidad (opcionales)
    product_id = Column(Integer, ForeignKey("sc_product.id"), nullable=True)
    facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)
    process_id = Column(Integer, ForeignKey("sc_process.id"), nullable=True)
    transport_leg_id = Column(Integer, ForeignKey("sc_transport_leg.id"), nullable=True)

    source_system = Column(String(100), nullable=True)
    comment = Column(Text, nullable=True)

    scenario = relationship("Scenario", back_populates="measurements")
    import_run = relationship("ImportRun", back_populates="measurements")
    product = relationship("Product", back_populates="measurements")
    facility = relationship("Facility", back_populates="measurements")
    process = relationship("Process", back_populates="measurements")
    transport_leg = relationship("TransportLeg", back_populates="measurements")


# ---------- KPI results ----------
class KPIResult(Base):
    __tablename__ = "sc_kpi_result"
    id = Column(Integer, primary_key=True)

    kpi_id = Column(Integer, ForeignKey("sc_kpi.id"), nullable=False)
    scenario_id = Column(Integer, ForeignKey("sc_scenario.id"), nullable=True)
    import_run_id = Column(Integer, ForeignKey("sc_import_run.id"), nullable=True, index=True)

    # contexto opcional (para drill-down por DPP/planta)
    product_id = Column(Integer, ForeignKey("sc_product.id"), nullable=True)
    facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)

    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)
    value = Column(Float, nullable=False)

    kpi = relationship("KPI", back_populates="results")
    scenario = relationship("Scenario", back_populates="kpi_results")
    import_run = relationship("ImportRun", back_populates="kpi_results")
    product = relationship("Product", back_populates="kpi_results")
    facility = relationship("Facility", back_populates="kpi_results")


class KPINormalizedResult(Base):
    __tablename__ = "sc_kpi_normalized_result"
    id = Column(Integer, primary_key=True)

    kpi_id = Column(Integer, ForeignKey("sc_kpi.id"), nullable=False)
    scenario_id = Column(Integer, ForeignKey("sc_scenario.id"), nullable=True)
    import_run_id = Column(Integer, ForeignKey("sc_import_run.id"), nullable=True, index=True)

    period_end = Column(DateTime, nullable=True)

    raw_value = Column(Float, nullable=True)
    normalized_value = Column(Float, nullable=True)   # 0..100
    semaforo = Column(String(20), nullable=True)      # Green / Amber / Red / Missing / Need BASE

    lower_ref = Column(Float, nullable=True)
    upper_ref = Column(Float, nullable=True)
    baseline_value = Column(Float, nullable=True)

    normalization_method = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    import_run = relationship("ImportRun", back_populates="normalized_results")


class EmissionFactor(Base):
    __tablename__ = "sc_emission_factor"
    id = Column(Integer, primary_key=True)
    code = Column(String(100), unique=True, nullable=True)
    name = Column(String(255), nullable=False)
    activity_type = Column(String(100), nullable=False)
    unit = Column(String(50), nullable=True)
    value = Column(Float, nullable=False)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    source = Column(String(255), nullable=True)
    analytical_role = Column(String(100), nullable=True)
    factor_set_id = Column(String(100), nullable=True)


class CostFactor(Base):
    __tablename__ = "sc_cost_factor"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    activity_type = Column(String(100), nullable=False)
    unit = Column(String(50), nullable=True)
    value = Column(Float, nullable=False)
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    source = Column(String(255), nullable=True)


# ---------- Minimal DPP-ready MVP extension ----------

class ProductBatch(Base):
    __tablename__ = "sc_product_batch"
    id = Column(Integer, primary_key=True)

    batch_code = Column(String(120), unique=True, nullable=False)

    product_id = Column(Integer, ForeignKey("sc_product.id"), nullable=False)
    scenario_id = Column(Integer, ForeignKey("sc_scenario.id"), nullable=True)
    origin_facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)

    production_date = Column(DateTime, nullable=True)
    quantity = Column(Float, nullable=True)
    unit = Column(String(50), nullable=True)

    status = Column(String(50), nullable=True)   # produced / in_stock / shipped / delivered
    notes = Column(Text, nullable=True)
    source_system = Column(String(100), nullable=True)
    source_reference = Column(String(255), nullable=True)
    import_run_id = Column(Integer, ForeignKey("sc_import_run.id"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    product = relationship("Product")
    scenario = relationship("Scenario")
    origin_facility = relationship("Facility")


class TraceabilityEvent(Base):
    __tablename__ = "sc_traceability_event"
    id = Column(Integer, primary_key=True)

    batch_id = Column(Integer, ForeignKey("sc_product_batch.id"), nullable=False)
    event_code = Column(String(120), nullable=True, index=True)
    event_type = Column(String(80), nullable=False)
    # examples: produced, quality_checked, loaded, shipped, delivered

    timestamp = Column(DateTime, nullable=False)

    facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)
    process_id = Column(Integer, ForeignKey("sc_process.id"), nullable=True)
    transport_leg_id = Column(Integer, ForeignKey("sc_transport_leg.id"), nullable=True)

    quantity = Column(Float, nullable=True)
    unit = Column(String(50), nullable=True)

    source_system = Column(String(100), nullable=True)
    source_reference = Column(String(255), nullable=True)
    comment = Column(Text, nullable=True)
    import_run_id = Column(Integer, ForeignKey("sc_import_run.id"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    batch = relationship("ProductBatch")
    facility = relationship("Facility")
    process = relationship("Process")
    transport_leg = relationship("TransportLeg")
