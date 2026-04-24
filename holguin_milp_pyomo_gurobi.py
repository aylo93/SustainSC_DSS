from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, NonNegativeIntegers, Binary,
    Objective, Constraint, minimize, SolverFactory, value
)
import csv
from pathlib import Path

# -----------------------------
# Data
# -----------------------------
SITES = ["BARIAY", "LOS", "PILON", "RIO", "BUEN"]
TRUCKS = ["TR7.5", "TR15"]

cap = {
    "BARIAY": 70000,
    "LOS": 65000,
    "PILON": 65000,
    "RIO": 65000,
    "BUEN": 65000,
}

dist_km = {
    "BARIAY": 120,
    "LOS": 160,
    "PILON": 180,
    "RIO": 170,
    "BUEN": 150,
}

elec_kwh_t = {
    "BARIAY": 2.95,
    "LOS": 3.79,
    "PILON": 3.58,
    "RIO": 3.37,
    "BUEN": 3.68,
}

diesel_kwh_t = {
    "BARIAY": 1.44,
    "LOS": 1.81,
    "PILON": 1.83,
    "RIO": 1.79,
    "BUEN": 1.81,
}

residual_prod_cost = {
    "BARIAY": 9.10,
    "LOS": 9.80,
    "PILON": 9.60,
    "RIO": 9.50,
    "BUEN": 9.75,
}

truck_payload_t = {
    "TR7.5": 7.5,
    "TR15": 15.0,
}

# €/vehicle-km
truck_gamma = {
    "TR7.5": 0.18,
    "TR15": 0.30,
}

# Annual total loaded trips available by truck class
truck_availability = {
    "TR7.5": 8000,
    "TR15": 18000,
}

# Big-M upper bound for trip activation logic
M = {(i, k): truck_availability[k] for i in SITES for k in TRUCKS}

# Global parameters
D = 288000
p_e = 0.40
p_f = 1.20
C_maint = 350000.0
EF_e = 0.8170     # kgCO2e/kWh
EF_d = 0.2668     # kgCO2e/kWh
CO2_cap = 940.0   # tCO2e

# -----------------------------
# Model builder
# -----------------------------

def build_model(mode="min_cost", co2_cap=None):
    """
    mode:
      - 'min_cost'
      - 'min_cost_cap'
      - 'min_co2'
    co2_cap: optional cap in tCO2e
    """
    m = ConcreteModel()
    m.I = Set(initialize=SITES)
    m.K = Set(initialize=TRUCKS)

    m.cap = Param(m.I, initialize=cap)
    m.dist = Param(m.I, initialize=dist_km)
    m.e = Param(m.I, initialize=elec_kwh_t)
    m.d = Param(m.I, initialize=diesel_kwh_t)
    m.cprod = Param(m.I, initialize=residual_prod_cost)
    m.q = Param(m.K, initialize=truck_payload_t)
    m.gamma = Param(m.K, initialize=truck_gamma)
    m.avail = Param(m.K, initialize=truck_availability)

    m.x = Var(m.I, domain=NonNegativeReals)
    m.n = Var(m.I, m.K, domain=NonNegativeIntegers)
    m.z = Var(m.I, m.K, domain=Binary)

    def demand_rule(mm):
        return sum(mm.x[i] for i in mm.I) == D
    m.Demand = Constraint(rule=demand_rule)

    def cap_rule(mm, i):
        return mm.x[i] <= mm.cap[i]
    m.Capacity = Constraint(m.I, rule=cap_rule)

    def trip_link_rule(mm, i):
        return mm.x[i] <= sum(mm.q[k] * mm.n[i, k] for k in mm.K)
    m.TripLink = Constraint(m.I, rule=trip_link_rule)

    def activation_rule(mm, i, k):
        return mm.n[i, k] <= M[(i, k)] * mm.z[i, k]
    m.Activation = Constraint(m.I, m.K, rule=activation_rule)

    def fleet_avail_rule(mm, k):
        return sum(mm.n[i, k] for i in mm.I) <= mm.avail[k]
    m.FleetAvailability = Constraint(m.K, rule=fleet_avail_rule)

    def total_co2_t(mm):
        return sum(((mm.e[i] * EF_e + mm.d[i] * EF_d) / 1000.0) * mm.x[i] for i in mm.I)

    m.TotalCO2 = total_co2_t

    if co2_cap is not None:
        def co2_cap_rule(mm):
            return total_co2_t(mm) <= co2_cap
        m.CO2Cap = Constraint(rule=co2_cap_rule)

    if mode in ("min_cost", "min_cost_cap"):
        def obj_rule(mm):
            production_energy = sum((mm.cprod[i] + p_e * mm.e[i] + p_f * mm.d[i]) * mm.x[i] for i in mm.I)
            dispatch_cost = sum(mm.gamma[k] * mm.dist[i] * mm.n[i, k] for i in mm.I for k in mm.K)
            return production_energy + dispatch_cost + C_maint
        m.Obj = Objective(rule=obj_rule, sense=minimize)
    elif mode == "min_co2":
        eps = 1e-6
        def obj_rule(mm):
            dispatch_cost = sum(mm.gamma[k] * mm.dist[i] * mm.n[i, k] for i in mm.I for k in mm.K)
            return total_co2_t(mm) + eps * dispatch_cost
        m.Obj = Objective(rule=obj_rule, sense=minimize)
    else:
        raise ValueError("Unknown mode")

    return m


# -----------------------------
# Solve and export
# -----------------------------

def solve_scenario(scenario_code, mode, co2_cap=None, solver_name="gurobi"):
    model = build_model(mode=mode, co2_cap=co2_cap)
    solver = SolverFactory(solver_name)
    result = solver.solve(model, tee=False)

    x = {i: value(model.x[i]) for i in model.I}
    trips = {(i, k): int(round(value(model.n[i, k]))) for i in model.I for k in model.K}

    electricity = sum(elec_kwh_t[i] * x[i] for i in SITES)
    diesel = sum(diesel_kwh_t[i] * x[i] for i in SITES)
    transport_work = sum(dist_km[i] * x[i] for i in SITES)
    total_co2 = sum(((elec_kwh_t[i] * EF_e + diesel_kwh_t[i] * EF_d) / 1000.0) * x[i] for i in SITES)
    operating_cost = value(model.Obj) if mode != "min_co2" else (
        sum((residual_prod_cost[i] + p_e * elec_kwh_t[i] + p_f * diesel_kwh_t[i]) * x[i] for i in SITES)
        + sum(truck_gamma[k] * dist_km[i] * trips[(i, k)] for i in SITES for k in TRUCKS)
        + C_maint
    )

    return {
        "scenario_code": scenario_code,
        "x": x,
        "trips": trips,
        "electricity_kwh": electricity,
        "diesel_kwh": diesel,
        "transport_work_tkm": transport_work,
        "total_co2_t": total_co2,
        "operating_cost_eur": operating_cost,
        "solver_status": str(result.solver.status),
        "termination_condition": str(result.solver.termination_condition),
    }


def export_results(results, out_dir="milp_outputs"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # summary csv
    with open(out_path / "scenario_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario_code", "operating_cost_eur", "total_co2_t", "electricity_kwh",
            "diesel_kwh", "transport_work_tkm", "solver_status", "termination_condition"
        ])
        for r in results:
            w.writerow([
                r["scenario_code"], round(r["operating_cost_eur"], 3), round(r["total_co2_t"], 3),
                round(r["electricity_kwh"], 3), round(r["diesel_kwh"], 3), round(r["transport_work_tkm"], 3),
                r["solver_status"], r["termination_condition"]
            ])

    # allocation csv
    with open(out_path / "milp_allocations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_code", "plant_code", "x_ton", "truck_class", "trips"])
        for r in results:
            for i in SITES:
                for k in TRUCKS:
                    w.writerow([r["scenario_code"], i, round(r["x"][i], 3), k, r["trips"][(i, k)]])


if __name__ == "__main__":
    scenarios = [
        ("MILP_MIN_COST", "min_cost", None),
        ("MILP_CO2CAP_940", "min_cost_cap", CO2_cap),
        ("MILP_MIN_CO2", "min_co2", None),
    ]

    results = []
    for code, mode, cap_value in scenarios:
        r = solve_scenario(code, mode, co2_cap=cap_value, solver_name="gurobi")
        results.append(r)
        print(f"\n=== {code} ===")
        print("status:", r["solver_status"], "| term:", r["termination_condition"])
        print("operating_cost_eur:", round(r["operating_cost_eur"], 2))
        print("total_co2_t:", round(r["total_co2_t"], 3))
        print("electricity_kwh:", round(r["electricity_kwh"], 1))
        print("diesel_kwh:", round(r["diesel_kwh"], 1))
        print("transport_work_tkm:", round(r["transport_work_tkm"], 1))
        print("allocation:", {i: round(r["x"][i], 1) for i in SITES})

    export_results(results)
    print("\nFiles written to ./milp_outputs/")
