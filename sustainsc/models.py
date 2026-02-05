# sustainsc/models.py

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from datetime import datetime

from sqlalchemy.orm import declarative_base

Base = declarative_base()



class ProductFamily(Base):
    __tablename__ = "sc_product_family"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)

    products = relationship("Product", back_populates="family")


class Product(Base):
    __tablename__ = "sc_product"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)

    family_id = Column(Integer, ForeignKey("sc_product_family.id"))
    family = relationship("ProductFamily", back_populates="products")

    # For traceability and DPP/ADP integration (generic, any supply chain)
    dpp_id = Column(String)  # external digital product passport identifier


class Facility(Base):
    __tablename__ = "sc_facility"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    facility_type = Column(String)  # e.g. "supplier", "plant", "warehouse", "customer"
    country = Column(String)
    region = Column(String)
    city = Column(String)

    processes = relationship("Process", back_populates="facility")
    outgoing_legs = relationship(
        "TransportLeg",
        back_populates="origin_facility",
        foreign_keys="TransportLeg.origin_facility_id",
    )
    incoming_legs = relationship(
        "TransportLeg",
        back_populates="destination_facility",
        foreign_keys="TransportLeg.destination_facility_id",
    )


class Process(Base):
    __tablename__ = "sc_process"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)

    facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=False)
    facility = relationship("Facility", back_populates="processes")

    # Generic tags for mapping to SCOR or other taxonomies if needed
    process_category = Column(String)  # e.g. "make", "source", "deliver"


class TransportLeg(Base):
    __tablename__ = "sc_transport_leg"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    mode = Column(String)  # e.g. "road", "rail", "sea", "air"

    origin_facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=False)
    destination_facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=False)

    origin_facility = relationship(
        "Facility",
        foreign_keys=[origin_facility_id],
        back_populates="outgoing_legs",
    )
    destination_facility = relationship(
        "Facility",
        foreign_keys=[destination_facility_id],
        back_populates="incoming_legs",
    )

    distance_km = Column(Float)       # nominal distance
    typical_lead_time_h = Column(Float)


class Scenario(Base):
    __tablename__ = "sc_scenario"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)

    # Scenario metadata: could store JSON in a string column or in a separate table
    # to capture assumptions about network configuration, technologies, policies, etc.
    notes = Column(String)


class EmissionFactor(Base):
    __tablename__ = "sc_emission_factor"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    activity_type = Column(String, nullable=False)  # e.g. "electricity_kwh", "diesel_litre", "tkm_road"
    unit = Column(String, nullable=False)           # e.g. "kg CO2e/kWh"
    value = Column(Float, nullable=False)

    valid_from = Column(DateTime)
    valid_to = Column(DateTime)
    source = Column(String)  # e.g. "GHG Protocol", "LCA database", etc.


class CostFactor(Base):
    __tablename__ = "sc_cost_factor"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    activity_type = Column(String, nullable=False)  # e.g. "labour_hour", "electricity_kwh"
    unit = Column(String, nullable=False)           # e.g. "€/kWh"
    value = Column(Float, nullable=False)

    valid_from = Column(DateTime)
    valid_to = Column(DateTime)
    source = Column(String)


class Measurement(Base):
    """
    Generic MRV measurement record:
    - Captures activity data such as production quantity, energy use, transport work, etc.
    - Can be linked to process and/or transport legs, facilities, products and scenarios.
    """

    __tablename__ = "sc_measurement"

    id = Column(Integer, primary_key=True)

    variable_name = Column(String, nullable=False)   # e.g. "electricity_kwh", "diesel_litre", "throughput_tonnes"
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)            # e.g. "kWh", "litre", "tonne", "tkm"

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    product_id = Column(Integer, ForeignKey("sc_product.id"), nullable=True)
    facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)
    process_id = Column(Integer, ForeignKey("sc_process.id"), nullable=True)
    transport_leg_id = Column(Integer, ForeignKey("sc_transport_leg.id"), nullable=True)
    scenario_id = Column(Integer, ForeignKey("sc_scenario.id"), nullable=True)

    source_system = Column(String)   # e.g. "ERP", "MES", "manual", "IoT"
    comment = Column(String)

    product = relationship("Product")
    facility = relationship("Facility")
    process = relationship("Process")
    transport_leg = relationship("TransportLeg")
    scenario = relationship("Scenario")


class KPI(Base):
    __tablename__ = "sc_kpi"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)  # e.g. "E1", "EC3", "S2", "T4"
    name = Column(String, nullable=False)
    description = Column(String)

    dimension = Column(String, nullable=False)  # "environmental", "economic", "social", "technological"
    decision_level = Column(String, nullable=False)  # "strategic", "tactical", "operational"
    flow = Column(String, nullable=False)  # "physical", "informational", "financial"

    unit = Column(String, nullable=False)       # "t CO2e", "€/unit", "%", etc.
    is_benefit = Column(Boolean, default=False) # True if higher is better (for MCDA)

    # For implementation simplicity, store a formula reference (e.g. Python function name or expression).
    formula_id = Column(String)                 # Link to KPI computation logic

    # Optional: JSON or text specifying data sources and protocol details
    protocol_notes = Column(String)

    results = relationship("KPIResult", back_populates="kpi")


class KPIResult(Base):
    __tablename__ = "sc_kpi_result"

    id = Column(Integer, primary_key=True)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    kpi_id = Column(Integer, ForeignKey("sc_kpi.id"), nullable=False)
    scenario_id = Column(Integer, ForeignKey("sc_scenario.id"), nullable=True)

    # Context for which the KPI is computed (generic, any supply chain)
    product_id = Column(Integer, ForeignKey("sc_product.id"), nullable=True)
    facility_id = Column(Integer, ForeignKey("sc_facility.id"), nullable=True)

    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)

    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    kpi = relationship("KPI", back_populates="results")
    scenario = relationship("Scenario")
    product = relationship("Product")
    facility = relationship("Facility")

    __table_args__ = (
        UniqueConstraint(
            "kpi_id",
            "scenario_id",
            "product_id",
            "facility_id",
            "period_start",
            "period_end",
            name="uq_kpi_context",
        ),
    )