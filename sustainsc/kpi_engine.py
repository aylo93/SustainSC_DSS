# sustainsc/kpi_engine.py
# SustainSC DSS - KPI Engine for 26 core KPIs
# Computes KPIResults from MRV measurements (+ emission/cost factors when needed)
# Key improvement: KPIResult.period_end uses last Measurement.timestamp (per scenario) for time-series
# Key improvement: Upsert behavior avoids duplicates / UniqueConstraint conflicts

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, Callable, List

from sqlalchemy.orm import Session
from sqlalchemy import and_

from .config import SessionLocal
from .models import KPI, KPIResult, Measurement, EmissionFactor, CostFactor, Scenario


# -----------------------------
# Helpers
# -----------------------------

def utc_now_naive() -> datetime:
    """Naive UTC datetime (SQLite-friendly)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

def safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return float(n) / float(d)

def clamp_0_100(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return max(0.0, min(100.0, float(x)))

def fetch_values(session: Session, var: str, scenario_id: Optional[int]) -> List[float]:
    q = session.query(Measurement.value).filter(Measurement.variable_name == var)
    if scenario_id is not None:
        q = q.filter(Measurement.scenario_id == scenario_id)
    return [float(v[0]) for v in q.all()]

def fetch_latest(session: Session, var: str, scenario_id: Optional[int]) -> Optional[float]:
    q = session.query(Measurement).filter(Measurement.variable_name == var)
    if scenario_id is not None:
        q = q.filter(Measurement.scenario_id == scenario_id)
    q = q.order_by(Measurement.timestamp.desc())
    m = q.first()
    return float(m.value) if m else None

def sum_var(session: Session, var: str, scenario_id: Optional[int]) -> Optional[float]:
    vals = fetch_values(session, var, scenario_id)
    return float(sum(vals)) if vals else None

def avg_var(session: Session, var: str, scenario_id: Optional[int]) -> Optional[float]:
    vals = fetch_values(session, var, scenario_id)
    return float(sum(vals) / len(vals)) if vals else None

def direct_or_none(session: Session, formula_id: str, scenario_id: Optional[int]) -> Optional[float]:
    """If a measurement exists with variable_name == formula_id, use latest value as direct KPI."""
    return fetch_latest(session, formula_id, scenario_id)

def list_scenarios(session: Session) -> List[Optional[Scenario]]:
    scenarios = session.query(Scenario).all()
    return scenarios if scenarios else [None]

def kpiresult_has_column(colname: str) -> bool:
    return colname in KPIResult.__table__.columns.keys()

def scenario_effective_timestamp(session: Session, scenario_id: Optional[int]) -> datetime:
    """
    Use last MRV measurement timestamp as the KPIResult timestamp for that scenario.
    This makes the dashboard time-series reflect the month of the input data.
    """
    q = session.query(Measurement.timestamp)
    if scenario_id is not None:
        q = q.filter(Measurement.scenario_id == scenario_id)
    max_ts = q.order_by(Measurement.timestamp.desc()).first()
    return max_ts[0] if max_ts and max_ts[0] else utc_now_naive()

def upsert_kpi_result(
    session: Session,
    kpi_id: int,
    scenario_id: Optional[int],
    value: float,
    computed_at: datetime
) -> None:
    """
    Upsert 1 result per KPI per scenario per period_end.
    Avoids duplicates and UniqueConstraint collisions.
    """
    existing = (
        session.query(KPIResult)
        .filter_by(
            kpi_id=kpi_id,
            scenario_id=scenario_id,
            product_id=None,
            facility_id=None,
            period_start=None,
            period_end=computed_at,
        )
        .first()
    )

    if existing:
        existing.value = float(value)
        # optional extra columns
        if kpiresult_has_column("timestamp"):
            setattr(existing, "timestamp", computed_at)
        if kpiresult_has_column("computed_at"):
            setattr(existing, "computed_at", computed_at)
        return

    kwargs = dict(
        kpi_id=kpi_id,
        scenario_id=scenario_id,
        product_id=None,
        facility_id=None,
        period_start=None,
        period_end=computed_at,
        value=float(value),
    )

    if kpiresult_has_column("timestamp"):
        kwargs["timestamp"] = computed_at
    if kpiresult_has_column("computed_at"):
        kwargs["computed_at"] = computed_at

    session.add(KPIResult(**kwargs))


# -----------------------------
# Factors (optional use)
# -----------------------------

def select_valid_emission_factor(session: Session, activity_type: str, ts: datetime) -> Optional[EmissionFactor]:
    q = session.query(EmissionFactor).filter(EmissionFactor.activity_type == activity_type)
    q = q.filter(
        and_(
            (EmissionFactor.valid_from.is_(None) | (EmissionFactor.valid_from <= ts)),
            (EmissionFactor.valid_to.is_(None) | (EmissionFactor.valid_to >= ts)),
        )
    )
    q = q.order_by(EmissionFactor.valid_from.desc().nullslast(), EmissionFactor.id.desc())
    return q.first()

def compute_total_ghg_tco2e_from_factors(session: Session, scenario_id: Optional[int]) -> Optional[float]:
    """
    Sum MRV measurements where emission_factor.activity_type == measurement.variable_name.
    Factor value assumed: kgCO2e per unit of measurement. Return tCO2e.
    """
    q = session.query(Measurement)
    if scenario_id is not None:
        q = q.filter(Measurement.scenario_id == scenario_id)

    total_kg = 0.0
    used_any = False

    for m in q.all():
        ef = select_valid_emission_factor(session, m.variable_name, m.timestamp)
        if ef is None:
            continue
        used_any = True
        total_kg += float(m.value) * float(ef.value)

    if not used_any:
        return None
    return total_kg / 1000.0


# -----------------------------
# KPI Context
# -----------------------------

@dataclass
class Ctx:
    session: Session
    scenario_id: Optional[int]
    cache: Dict[Tuple[Optional[int], str], Optional[float]]

    def get_cached(self, formula_id: str) -> Optional[float]:
        return self.cache.get((self.scenario_id, formula_id))

    def set_cached(self, formula_id: str, value: Optional[float]) -> None:
        self.cache[(self.scenario_id, formula_id)] = value

    def sum(self, var: str) -> Optional[float]:
        return sum_var(self.session, var, self.scenario_id)

    def avg(self, var: str) -> Optional[float]:
        return avg_var(self.session, var, self.scenario_id)

    def latest(self, var: str) -> Optional[float]:
        return fetch_latest(self.session, var, self.scenario_id)

    def direct(self, formula_id: str) -> Optional[float]:
        return direct_or_none(self.session, formula_id, self.scenario_id)

    def pick_sum(self, *names: str) -> Optional[float]:
        for n in names:
            v = self.sum(n)
            if v is not None:
                return v
        return None

    def pick_avg(self, *names: str) -> Optional[float]:
        for n in names:
            v = self.avg(n)
            if v is not None:
                return v
        return None

    def pick_latest(self, *names: str) -> Optional[float]:
        for n in names:
            v = self.latest(n)
            if v is not None:
                return v
        return None


# -----------------------------
# FORMULAS (aligned to your measurements.csv)
# -----------------------------

def ghg_total_s1s2(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("ghg_total_s1s2")
    if v is not None:
        return v
    return compute_total_ghg_tco2e_from_factors(ctx.session, ctx.scenario_id)

def ghg_intensity_fu(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("ghg_intensity_fu")
    if v is not None:
        return v
    e1_t = compute_formula(ctx, "ghg_total_s1s2")
    output = ctx.pick_sum("output_qty_fu")
    if e1_t is None or output is None:
        return None
    return safe_div(e1_t * 1000.0, output)  # tCO2e -> kgCO2e

def energy_intensity_fu(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("energy_intensity_fu")
    if v is not None:
        return v
    electricity = ctx.pick_sum("electricity_kwh") or 0.0
    diesel = ctx.pick_sum("diesel_kwh") or 0.0
    total_energy = electricity + diesel
    output = ctx.pick_sum("output_qty_fu")
    if output is None:
        return None
    return safe_div(total_energy, output)

def renewable_energy_share(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("renewable_energy_share")
    if v is not None:
        return v
    renewable = ctx.pick_sum("renewable_energy_kwh")
    electricity = ctx.pick_sum("electricity_kwh") or 0.0
    diesel = ctx.pick_sum("diesel_kwh") or 0.0
    total_energy = electricity + diesel
    if renewable is None or total_energy == 0:
        return None
    return clamp_0_100(safe_div(renewable, total_energy) * 100.0)

def waste_recovery_rate(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("waste_recovery_rate")
    if v is not None:
        return v
    rec = ctx.pick_sum("waste_recovered_t")
    gen = ctx.pick_sum("waste_generated_t")
    if rec is None or gen is None:
        return None
    return clamp_0_100(safe_div(rec, gen) * 100.0)

def water_intensity_fu(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("water_intensity_fu")
    if v is not None:
        return v
    water = ctx.pick_sum("water_withdrawn_m3")
    output = ctx.pick_sum("output_qty_fu")
    if water is None or output is None:
        return None
    return safe_div(water, output)

def circularity_ratio(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("circularity_ratio")
    if v is not None:
        return v
    circ = ctx.pick_sum("material_circular_t")
    total = ctx.pick_sum("material_total_t")
    if circ is None or total is None:
        return None
    return clamp_0_100(safe_div(circ, total) * 100.0)

def cost_per_fu(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("cost_per_fu")
    if v is not None:
        return v
    op_cost = ctx.pick_sum("operating_cost_eur")
    output = ctx.pick_sum("output_qty_fu")
    if op_cost is None or output is None:
        return None
    return safe_div(op_cost, output)

def energy_cost_share(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("energy_cost_share")
    if v is not None:
        return v
    op_cost = ctx.pick_sum("operating_cost_eur")
    fuel = ctx.pick_sum("fuel_cost_eur") or 0.0
    elec = ctx.pick_sum("electricity_cost_eur") or 0.0
    energy_cost = fuel + elec
    if op_cost is None or op_cost == 0:
        return None
    return clamp_0_100(safe_div(energy_cost, op_cost) * 100.0)

def oee(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("oee")
    if v is not None:
        return v
    a = ctx.pick_avg("oee_availability")
    p = ctx.pick_avg("oee_performance")
    q = ctx.pick_avg("oee_quality")
    if a is None or p is None or q is None:
        return None
    if a > 1.5: a /= 100.0
    if p > 1.5: p /= 100.0
    if q > 1.5: q /= 100.0
    return clamp_0_100(a * p * q * 100.0)

def rosi(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("rosi")
    if v is not None:
        return v
    capex = ctx.pick_sum("investment_cost_eur")
    benefits = ctx.pick_sum("incremental_benefits_eur")
    if capex is None or benefits is None or capex == 0:
        return None
    return safe_div((benefits - capex), capex) * 100.0

def maintenance_cost_per_hour(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("maintenance_cost_per_hour")
    if v is not None:
        return v
    mcost = ctx.pick_sum("maintenance_cost_eur")
    hours = ctx.pick_sum("operating_hours_h")
    if mcost is None or hours is None:
        return None
    return safe_div(mcost, hours)

def logistics_cost_per_tkm(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("logistics_cost_per_tkm")
    if v is not None:
        return v
    lcost = ctx.pick_sum("logistics_cost_eur")
    tkm = ctx.pick_sum("transport_work_tkm")
    if lcost is None or tkm is None:
        return None
    return safe_div(lcost, tkm)

def ltifr(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("ltifr")
    if v is not None:
        return v
    lti = ctx.pick_sum("lti_count")
    hours = ctx.pick_sum("hours_worked_h")
    if lti is None or hours is None:
        return None
    return safe_div(lti * 1_000_000.0, hours)

def training_hours_per_employee(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("training_hours_per_employee")
    if v is not None:
        return v
    th = ctx.pick_sum("training_hours_h")
    emp = ctx.pick_avg("employees_avg") or ctx.pick_sum("employees_total")
    if th is None or emp is None:
        return None
    return safe_div(th, emp)

def suggestions_per_employee(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("suggestions_per_employee")
    if v is not None:
        return v
    sug = ctx.pick_sum("suggestions_count")
    emp = ctx.pick_avg("employees_avg") or ctx.pick_sum("employees_total")
    if sug is None or emp is None:
        return None
    return safe_div(sug, emp)

def community_incidents_total(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("community_incidents_total")
    if v is not None:
        return v
    inc = ctx.pick_sum("community_incidents_count") or 0.0
    comp = ctx.pick_sum("substantiated_complaints_count") or 0.0
    return float(inc + comp)

def health_checks_coverage(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("health_checks_coverage")
    if v is not None:
        return v
    checked = ctx.pick_sum("employees_health_checks")
    emp_total = ctx.pick_sum("employees_total") or ctx.pick_avg("employees_avg")
    if checked is None or emp_total is None:
        return None
    return clamp_0_100(safe_div(checked, emp_total) * 100.0)

def customer_acceptance_index(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("customer_acceptance_index")
    if v is not None:
        return v
    sales = ctx.pick_avg("sustainable_sales_share")
    survey = ctx.pick_avg("customer_survey_score")
    renewal = ctx.pick_avg("contract_renewal_rate")
    if sales is None or survey is None or renewal is None:
        return None
    idx = (0.3 * sales + 0.4 * survey + 0.3 * renewal) * 100.0
    return clamp_0_100(idx)

def digitalization_rate(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("digitalization_rate")
    if v is not None:
        return v
    digital = ctx.pick_sum("digital_processes_count")
    total = ctx.pick_sum("core_processes_total")
    if digital is None or total is None:
        return None
    return clamp_0_100(safe_div(digital, total) * 100.0)

def iot_asset_connectivity(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("iot_asset_connectivity")
    if v is not None:
        return v
    conn = ctx.pick_sum("assets_connected")
    total = ctx.pick_sum("assets_total")
    if conn is None or total is None:
        return None
    return clamp_0_100(safe_div(conn, total) * 100.0)

def mrv_coverage(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("mrv_coverage")
    if v is not None:
        return v
    active = ctx.pick_sum("mrv_points_active_valid")
    req = ctx.pick_sum("mrv_points_required")
    if active is None or req is None:
        return None
    return clamp_0_100(safe_div(active, req) * 100.0)

def dpp_coverage(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("dpp_coverage")
    if v is not None:
        return v
    dpp_vol = ctx.pick_sum("dpp_volume")
    ship_vol = ctx.pick_sum("shipped_volume_total")
    if dpp_vol is None or ship_vol is None:
        return None
    return clamp_0_100(safe_div(dpp_vol, ship_vol) * 100.0)

def data_quality_index(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("data_quality_index")
    if v is not None:
        return v
    cpl = ctx.pick_avg("data_completeness")
    cns = ctx.pick_avg("data_consistency")
    acc = ctx.pick_avg("data_accuracy")
    if cpl is None or cns is None or acc is None:
        return None
    idx = (cpl + cns + acc) / 3.0 * 100.0
    return clamp_0_100(idx)

def ot_it_integration(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("ot_it_integration")
    if v is not None:
        return v
    return ctx.pick_avg("ot_it_integration_score")

def analytics_supported_decisions(ctx: Ctx) -> Optional[float]:
    v = ctx.direct("analytics_supported_decisions")
    if v is not None:
        return v
    supported = ctx.pick_sum("decisions_supported")
    total = ctx.pick_sum("decisions_total")
    if supported is None or total is None:
        return None
    return clamp_0_100(safe_div(supported, total) * 100.0)


FORMULAS: Dict[str, Callable[[Ctx], Optional[float]]] = {
    "ghg_total_s1s2": ghg_total_s1s2,
    "ghg_intensity_fu": ghg_intensity_fu,
    "energy_intensity_fu": energy_intensity_fu,
    "renewable_energy_share": renewable_energy_share,
    "waste_recovery_rate": waste_recovery_rate,
    "water_intensity_fu": water_intensity_fu,
    "circularity_ratio": circularity_ratio,
    "cost_per_fu": cost_per_fu,
    "energy_cost_share": energy_cost_share,
    "oee": oee,
    "rosi": rosi,
    "maintenance_cost_per_hour": maintenance_cost_per_hour,
    "logistics_cost_per_tkm": logistics_cost_per_tkm,
    "ltifr": ltifr,
    "training_hours_per_employee": training_hours_per_employee,
    "suggestions_per_employee": suggestions_per_employee,
    "community_incidents_total": community_incidents_total,
    "health_checks_coverage": health_checks_coverage,
    "customer_acceptance_index": customer_acceptance_index,
    "digitalization_rate": digitalization_rate,
    "iot_asset_connectivity": iot_asset_connectivity,
    "mrv_coverage": mrv_coverage,
    "dpp_coverage": dpp_coverage,
    "data_quality_index": data_quality_index,
    "ot_it_integration": ot_it_integration,
    "analytics_supported_decisions": analytics_supported_decisions,
}

def compute_formula(ctx: Ctx, formula_id: str) -> Optional[float]:
    cached = ctx.get_cached(formula_id)
    if cached is not None or (ctx.scenario_id, formula_id) in ctx.cache:
        return cached

    fn = FORMULAS.get(formula_id)
    if fn is None:
        value = ctx.direct(formula_id)
        ctx.set_cached(formula_id, value)
        return value

    value = fn(ctx)
    ctx.set_cached(formula_id, value)
    return value


def run_engine(debug_missing: bool = False) -> None:
    session = SessionLocal()
    try:
        kpis = session.query(KPI).all()
        scenarios = list_scenarios(session)

        cache: Dict[Tuple[Optional[int], str], Optional[float]] = {}
        total_written = 0

        for sc in scenarios:
            scenario_id = sc.id if sc is not None else None

            # IMPORTANT: compute timestamp per scenario based on last MRV measurement
            computed_at = scenario_effective_timestamp(session, scenario_id)

            ctx = Ctx(session=session, scenario_id=scenario_id, cache=cache)

            for k in kpis:
                fid = (k.formula_id or "").strip()
                if not fid:
                    continue

                value = compute_formula(ctx, fid)
                if value is None:
                    if debug_missing:
                        print(f"SKIP {k.code} ({fid}) -> missing input data")
                    continue

                upsert_kpi_result(session, k.id, scenario_id, float(value), computed_at)
                total_written += 1

        session.commit()
        print("=== KPI engine run completed ===")
        print(f"KPIs registered (sc_kpi): {len(kpis)}")
        print(f"KPI results written (sc_kpi_result): {total_written}")

    finally:
        session.close()


if __name__ == "__main__":
    run_engine()
