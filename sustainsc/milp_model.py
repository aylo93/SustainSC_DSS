# sustainsc/milp_model.py
# SustainSC DSS - MILP (2 plants) solved analytically (LP) for robustness
# Objective: minimize total operating cost subject to demand + capacities + CO2 cap.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Plant:
    code: str
    name: str
    capacity_t: float

    # per-ton intensities / coefficients
    prod_cost_eur_per_t: float          # "production" cost excluding energy/logistics/maintenance
    elec_kwh_per_t: float
    diesel_kwh_per_t: float
    distance_km: float                  # to market (for tkm)


@dataclass(frozen=True)
class Prices:
    electricity_eur_per_kwh: float
    fuel_eur_per_kwh: float
    logistics_eur_per_tkm: float
    maintenance_fixed_eur: float = 0.0  # kept constant (copied from BASE usually)


@dataclass(frozen=True)
class EmissionFactors:
    electricity_kgco2e_per_kwh: float
    diesel_kgco2e_per_kwh: float


@dataclass
class Solution:
    feasible: bool
    reason: str

    demand_t: float
    cap_tco2e: float

    # decision vars
    x_t: Dict[str, float]  # plant_code -> tonnes

    # derived totals (MRV-like)
    electricity_kwh: float
    diesel_kwh: float
    transport_work_tkm: float

    electricity_cost_eur: float
    fuel_cost_eur: float
    logistics_cost_eur: float
    production_cost_eur: float
    maintenance_cost_eur: float
    operating_cost_eur: float

    total_ghg_tco2e: float


def _compute_totals(
    plants: List[Plant],
    prices: Prices,
    efs: EmissionFactors,
    demand_t: float,
    cap_tco2e: float,
    x: Dict[str, float],
) -> Solution:
    elec = 0.0
    diesel = 0.0
    tkm = 0.0
    prod_cost = 0.0

    for p in plants:
        xi = float(x.get(p.code, 0.0))
        elec += xi * p.elec_kwh_per_t
        diesel += xi * p.diesel_kwh_per_t
        tkm += xi * p.distance_km
        prod_cost += xi * p.prod_cost_eur_per_t

    elec_cost = elec * prices.electricity_eur_per_kwh
    fuel_cost = diesel * prices.fuel_eur_per_kwh
    log_cost = tkm * prices.logistics_eur_per_tkm

    # emissions consistent with your KPI engine factor logic (electricity_kwh + diesel_kwh)
    total_kg = elec * efs.electricity_kgco2e_per_kwh + diesel * efs.diesel_kgco2e_per_kwh
    total_t = total_kg / 1000.0

    operating = prod_cost + elec_cost + fuel_cost + log_cost + prices.maintenance_fixed_eur

    return Solution(
        feasible=True,
        reason="OK",
        demand_t=demand_t,
        cap_tco2e=cap_tco2e,
        x_t=x,
        electricity_kwh=elec,
        diesel_kwh=diesel,
        transport_work_tkm=tkm,
        electricity_cost_eur=elec_cost,
        fuel_cost_eur=fuel_cost,
        logistics_cost_eur=log_cost,
        production_cost_eur=prod_cost,
        maintenance_cost_eur=prices.maintenance_fixed_eur,
        operating_cost_eur=operating,
        total_ghg_tco2e=total_t,
    )


def solve_cost_min_two_plants_with_co2_cap(
    demand_t: float,
    cap_tco2e: float,
    plants: List[Plant],
    prices: Prices,
    efs: EmissionFactors,
) -> Solution:
    """
    Two-plant LP:
      minimize c1*x1 + c2*x2 (+ derived components) subject to:
        x1 + x2 = demand
        0 <= x1 <= cap1
        0 <= x2 <= cap2
        emissions(x1,x2) <= cap_tco2e
    We solve analytically by restricting to feasible interval of x1.
    """
    if len(plants) != 2:
        return Solution(
            feasible=False,
            reason="This demo solver supports exactly 2 plants.",
            demand_t=demand_t,
            cap_tco2e=cap_tco2e,
            x_t={},
            electricity_kwh=0, diesel_kwh=0, transport_work_tkm=0,
            electricity_cost_eur=0, fuel_cost_eur=0, logistics_cost_eur=0,
            production_cost_eur=0, maintenance_cost_eur=0, operating_cost_eur=0,
            total_ghg_tco2e=0,
        )

    p1, p2 = plants[0], plants[1]

    if demand_t > (p1.capacity_t + p2.capacity_t):
        return Solution(
            feasible=False,
            reason="Infeasible: demand exceeds total capacity.",
            demand_t=demand_t,
            cap_tco2e=cap_tco2e,
            x_t={},
            electricity_kwh=0, diesel_kwh=0, transport_work_tkm=0,
            electricity_cost_eur=0, fuel_cost_eur=0, logistics_cost_eur=0,
            production_cost_eur=0, maintenance_cost_eur=0, operating_cost_eur=0,
            total_ghg_tco2e=0,
        )

    # x2 = demand - x1
    # capacity bounds -> x1 in [L, U]
    L = max(0.0, demand_t - p2.capacity_t)
    U = min(p1.capacity_t, demand_t)

    # emissions constraint:
    # total_t = (elec(x)*ef_e + diesel(x)*ef_d)/1000
    # elec(x) = x1*e1 + x2*e2
    # diesel(x)= x1*d1 + x2*d2
    # => total_kg = x1*(e1*ef_e + d1*ef_d) + x2*(e2*ef_e + d2*ef_d)
    # => total_kg = x1*A1 + (demand-x1)*A2 = demand*A2 + x1*(A1-A2)
    A1 = p1.elec_kwh_per_t * efs.electricity_kgco2e_per_kwh + p1.diesel_kwh_per_t * efs.diesel_kgco2e_per_kwh
    A2 = p2.elec_kwh_per_t * efs.electricity_kgco2e_per_kwh + p2.diesel_kwh_per_t * efs.diesel_kgco2e_per_kwh

    cap_kg = cap_tco2e * 1000.0
    # demand*A2 + x1*(A1-A2) <= cap_kg
    denom = (A1 - A2)

    if abs(denom) < 1e-9:
        # same emissions per tonne => either feasible or not
        total_kg_at_any = demand_t * A2
        if total_kg_at_any > cap_kg + 1e-9:
            return Solution(
                feasible=False,
                reason="Infeasible: CO2 cap too tight (same intensity for both plants).",
                demand_t=demand_t,
                cap_tco2e=cap_tco2e,
                x_t={},
                electricity_kwh=0, diesel_kwh=0, transport_work_tkm=0,
                electricity_cost_eur=0, fuel_cost_eur=0, logistics_cost_eur=0,
                production_cost_eur=0, maintenance_cost_eur=0, operating_cost_eur=0,
                total_ghg_tco2e=0,
            )
        # emissions don't restrict x1 interval
        x1_min, x1_max = L, U
    else:
        bound = (cap_kg - demand_t * A2) / denom
        # if denom>0 => x1 <= bound; if denom<0 => x1 >= bound
        if denom > 0:
            x1_min, x1_max = L, min(U, bound)
        else:
            x1_min, x1_max = max(L, bound), U

        if x1_min > x1_max + 1e-9:
            return Solution(
                feasible=False,
                reason="Infeasible: CO2 cap too tight given capacities.",
                demand_t=demand_t,
                cap_tco2e=cap_tco2e,
                x_t={},
                electricity_kwh=0, diesel_kwh=0, transport_work_tkm=0,
                electricity_cost_eur=0, fuel_cost_eur=0, logistics_cost_eur=0,
                production_cost_eur=0, maintenance_cost_eur=0, operating_cost_eur=0,
                total_ghg_tco2e=0,
            )

    # objective (cost): production + energy + logistics (+ maintenance constant)
    # since maintenance is constant, choose x1 at an interval end depending on marginal cost.
    def unit_total_cost(p: Plant) -> float:
        return (
            p.prod_cost_eur_per_t
            + p.elec_kwh_per_t * prices.electricity_eur_per_kwh
            + p.diesel_kwh_per_t * prices.fuel_eur_per_kwh
            + p.distance_km * prices.logistics_eur_per_tkm
        )

    c1 = unit_total_cost(p1)
    c2 = unit_total_cost(p2)

    # minimize c1*x1 + c2*(demand-x1) = demand*c2 + x1*(c1-c2)
    slope = (c1 - c2)
    if slope < 0:
        # cheaper to put more on plant1 => maximize x1
        x1 = float(x1_max)
    elif slope > 0:
        # cheaper to put less on plant1 => minimize x1
        x1 = float(x1_min)
    else:
        x1 = float(x1_min)

    x2 = demand_t - x1
    x = {p1.code: x1, p2.code: x2}

    sol = _compute_totals(plants, prices, efs, demand_t, cap_tco2e, x)
    # sanity check cap
    if sol.total_ghg_tco2e > cap_tco2e + 1e-6:
        sol.feasible = False
        sol.reason = "Numerical issue: emissions exceed cap."
    return sol
