from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Exact Section 8.13 formal factor domain used by the Romanian metamodel package.
FORMAL_BOUNDS: dict[str, tuple[float, float]] = {
    "demand_load": (0.80, 1.20),
    "demand_cv": (0.05, 0.30),
    "oee": (0.55, 0.90),
    "distance_mult": (0.75, 1.50),
    "resource_mult": (0.75, 1.25),
    "renewable_share": (0.00, 0.75),
    "zC": (0.10, 0.90),
    "zD": (0.25, 0.95),
    "zS": (0.40, 0.95),
}

STRATEGIES = ["BASE", "ENERGY", "MAINT", "CIRCULAR", "DIGITAL", "SOCIAL", "INTEGRATED"]
SIZE_CLASSES = ["small", "medium", "large"]
SIZE = {
    "small": {"scale": 0.45, "headcount": 35, "cost_scale": 1.08},
    "medium": {"scale": 1.00, "headcount": 140, "cost_scale": 1.00},
    "large": {"scale": 2.20, "headcount": 620, "cost_scale": 0.93},
}

# Constants copied from the reproducibility package used to generate the trained models.
GRID_EF = 0.22041
RENEW_EF = 0.0134
BASE_CYCLE = np.array([1.00, 1.35, 1.20, 1.55, 1.10, 0.80])
BASE_INFO_W = np.array([0.18, 0.12, 0.12, 0.16, 0.20, 0.22])

ARCHETYPE_ALIASES = {
    "RO-A1": "automotive vehicle auto components supplier forming machining joining finishing assembly",
    "RO-A2": "electrical electromechanical cable assembly cable harness wiring wire connector electronics testing",
    "RO-A3": "fabricated metal metalworking metal products cnc machining cutting deburring SME",
    "RO-A4": "industrial machinery equipment machine manufacturer fabrication subassembly final assembly",
    "RO-A5": "food processing beverage cold storage packaging preparation processing",
    "RO-A6": "wood furniture timber carpentry drying cutting finishing",
    "RO-A7": "rubber plastics polymer moulding molding extrusion injection processor",
    "RO-A8": "non metallic mineral construction material cement concrete aggregates kiln crushing mixing",
}

STRATEGY_PROTOTYPES = {
    "ENERGY": "energy electricity renewable decarbonization carbon emissions energy intensive low renewable electricity dependency",
    "MAINT": "maintenance OEE downtime reliability availability defects rework lean kaizen operational efficiency",
    "CIRCULAR": "circularity waste recycling recovery scrap material efficiency material loss circular economy",
    "DIGITAL": "digital MRV traceability DPP data IoT analytics automation information delay digital maturity",
    "SOCIAL": "social workforce training skills safety ergonomics workers competence health human resources",
    "INTEGRATED": "integrated combined balanced multiple sustainability dimensions simultaneous energy maintenance circular digital social",
}

PROBLEM_PATTERNS = {
    "ENERGY": [r"low renewable", r"limited renewable", r"high energy", r"energy[- ]intensive", r"electricity depend", r"carbon[- ]intensive", r"high emissions"],
    "MAINT": [r"low oee", r"downtime", r"maintenance gap", r"poor reliability", r"high defect", r"rework", r"availability problem"],
    "CIRCULAR": [r"low circular", r"limited circular", r"high waste", r"scrap", r"limited recovery", r"material loss", r"low recycling"],
    "DIGITAL": [r"low digital", r"limited digital", r"low mrv", r"traceability gap", r"poor data", r"information delay", r"low automation"],
    "SOCIAL": [r"low skills", r"training gap", r"safety problem", r"ergonomic", r"workforce gap", r"skill shortage"],
}

QUALITATIVE_RULES = {
    # feature: [(regex, relative position in formal range, trace rule id)]
    "renewable_share": [
        (r"(?:low|limited|very low) renewable", 0.20, "QL_RENEW_LOW"),
        (r"(?:high|strong) renewable", 0.80, "QL_RENEW_HIGH"),
    ],
    "zD": [
        (r"(?:low|limited|weak) digital(?:ization|isation| maturity)?", 0.25, "QL_DIGITAL_LOW"),
        (r"moderate digital(?:ization|isation| maturity)?", 0.50, "QL_DIGITAL_MOD"),
        (r"(?:high|advanced|strong) digital(?:ization|isation| maturity)?", 0.75, "QL_DIGITAL_HIGH"),
    ],
    "zC": [
        (r"(?:low|limited|weak) circular(?:ity| maturity)?", 0.25, "QL_CIRC_LOW"),
        (r"moderate circular(?:ity| maturity)?", 0.50, "QL_CIRC_MOD"),
        (r"(?:high|advanced|strong) circular(?:ity| maturity)?", 0.75, "QL_CIRC_HIGH"),
    ],
    "zS": [
        (r"(?:low|limited|weak) (?:social|skills|workforce)(?: maturity)?", 0.25, "QL_SOCIAL_LOW"),
        (r"moderate (?:social|skills|workforce)(?: maturity)?", 0.50, "QL_SOCIAL_MOD"),
        (r"(?:high|advanced|strong) (?:social|skills|workforce)(?: maturity)?", 0.75, "QL_SOCIAL_HIGH"),
    ],
    "oee": [
        (r"low oee", 0.25, "QL_OEE_LOW"),
        (r"moderate oee", 0.50, "QL_OEE_MOD"),
        (r"high oee", 0.75, "QL_OEE_HIGH"),
    ],
    "distance_mult": [
        (r"(?:high|complex) logistics", 0.75, "QL_LOGISTICS_HIGH"),
        (r"(?:low|simple) logistics", 0.25, "QL_LOGISTICS_LOW"),
    ],
    "demand_cv": [
        (r"(?:high|volatile) demand variability", 0.75, "QL_DEMAND_VAR_HIGH"),
        (r"(?:low|stable) demand variability", 0.25, "QL_DEMAND_VAR_LOW"),
    ],
    "demand_load": [
        (r"(?:high|heavy) demand load", 0.75, "QL_DEMAND_LOAD_HIGH"),
        (r"(?:low|light) demand load", 0.25, "QL_DEMAND_LOAD_LOW"),
    ],
    "resource_mult": [
        (r"(?:energy|resource)[- ]intensive", 0.75, "QL_RESOURCE_HIGH"),
        (r"low resource intensity", 0.25, "QL_RESOURCE_LOW"),
    ],
}


@dataclass
class ASCASuggestion:
    description: str
    archetype: str
    archetype_name: str
    archetype_similarity: float
    size_class: str
    strategy: str
    strategy_similarity: float
    lambda_intensity: float
    intensity_label: str
    parameters: dict[str, float]
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "archetype": self.archetype,
            "archetype_name": self.archetype_name,
            "archetype_similarity": self.archetype_similarity,
            "size_class": self.size_class,
            "strategy": self.strategy,
            "strategy_similarity": self.strategy_similarity,
            "lambda_intensity": self.lambda_intensity,
            "intensity_label": self.intensity_label,
            "parameters": self.parameters,
            "trace": self.trace,
        }


def midpoint_parameters() -> dict[str, float]:
    return {k: (lo + hi) / 2 for k, (lo, hi) in FORMAL_BOUNDS.items()}


def _range_value(feature: str, position: float) -> float:
    lo, hi = FORMAL_BOUNDS[feature]
    return float(lo + position * (hi - lo))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("_", "-")).strip()


def _semantic_choice(text: str, prototypes: dict[str, str]) -> tuple[str, float, dict[str, float]]:
    keys = list(prototypes)
    docs = [prototypes[k] for k in keys]
    vec = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
    mat = vec.fit_transform(docs + [text])
    scores = cosine_similarity(mat[-1], mat[:-1]).ravel()
    ranking = {k: float(scores[i]) for i, k in enumerate(keys)}
    best_i = int(np.argmax(scores))
    return keys[best_i], float(scores[best_i]), ranking


def infer_size(text: str) -> tuple[str, str]:
    t = _norm_text(text)
    if re.search(r"\b(large|large-sized|big|enterprise-sized)\b", t):
        return "large", "SIZE_EXPLICIT_LARGE"
    if re.search(r"\b(small|small-sized|micro)\b", t):
        return "small", "SIZE_EXPLICIT_SMALL"
    if re.search(r"\b(medium|medium-sized|mid-sized|sme)\b", t):
        return "medium", "SIZE_EXPLICIT_MEDIUM"
    return "medium", "SIZE_DEFAULT_MEDIUM"


def infer_intensity(text: str, strategy: str) -> tuple[float, str, str]:
    if strategy == "BASE":
        return 0.0, "BASE", "LAMBDA_BASE_ZERO"
    t = _norm_text(text)
    if re.search(r"\b(high|strong|aggressive) (?:intensity|intervention|strategy)\b", t):
        return 0.75, "HIGH", "LAMBDA_HIGH_075"
    if re.search(r"\b(low|mild|conservative) (?:intensity|intervention|strategy)\b", t):
        return 0.30, "LOW", "LAMBDA_LOW_030"
    return 0.50, "MODERATE", "LAMBDA_MODERATE_050"


def load_archetypes(assets_dir: str | Path) -> pd.DataFrame:
    p = Path(assets_dir) / "01_design" / "archetype_reference_parameters.csv"
    return pd.read_csv(p).sort_values("archetype").reset_index(drop=True)


def suggest_from_description(description: str, assets_dir: str | Path) -> ASCASuggestion:
    text = _norm_text(description)
    archetypes = load_archetypes(assets_dir)
    arch_prototypes: dict[str, str] = {}
    for _, r in archetypes.iterrows():
        arch_prototypes[str(r.archetype)] = " ".join([
            str(r["name"]), str(r["process_topology"]), ARCHETYPE_ALIASES.get(str(r.archetype), "")
        ])
    arch, arch_sim, _ = _semantic_choice(text, arch_prototypes)

    strategy, strategy_sim, strategy_scores = _semantic_choice(text, STRATEGY_PROTOTYPES)
    strong_problems = []
    for s, pats in PROBLEM_PATTERNS.items():
        if any(re.search(p, text) for p in pats):
            strong_problems.append(s)
    if len(strong_problems) >= 2:
        strategy = "INTEGRATED"
        strategy_sim = max(strategy_scores.get(x, 0.0) for x in strong_problems)
    elif len(strong_problems) == 1:
        strategy = strong_problems[0]
        strategy_sim = max(strategy_sim, strategy_scores.get(strategy, 0.0))

    # Explicit scenario family names always win over semantic suggestion.
    for s in STRATEGIES:
        if s != "BASE" and re.search(rf"\b{s.lower()}\b", text):
            strategy = s
            break
    if re.search(r"\bbase(?:line)?\b", text) and not strong_problems:
        strategy = "BASE"

    size_class, size_rule = infer_size(text)
    lam, intensity_label, lambda_rule = infer_intensity(text, strategy)

    params = midpoint_parameters()
    trace = [f"ARCH_SEMANTIC:{arch}:{arch_sim:.3f}", f"STRATEGY_SEMANTIC:{strategy}:{strategy_sim:.3f}", size_rule, lambda_rule]
    for feature, rules in QUALITATIVE_RULES.items():
        for pattern, position, rule_id in rules:
            if re.search(pattern, text):
                params[feature] = _range_value(feature, position)
                trace.append(f"{rule_id}:{feature}={params[feature]:.6g}")
                break

    name = str(archetypes.loc[archetypes.archetype == arch, "name"].iloc[0])
    return ASCASuggestion(
        description=description,
        archetype=arch,
        archetype_name=name,
        archetype_similarity=arch_sim,
        size_class=size_class,
        strategy=strategy,
        strategy_similarity=strategy_sim,
        lambda_intensity=lam,
        intensity_label=intensity_label,
        parameters=params,
        trace=trace,
    )


def _arch_record(archetype: str, assets_dir: str | Path) -> dict[str, Any]:
    df = load_archetypes(assets_dir)
    row = df[df.archetype == archetype]
    if row.empty:
        raise ValueError(f"Unknown archetype: {archetype}")
    return row.iloc[0].to_dict()


def enrich_reference_features(config: dict[str, Any], assets_dir: str | Path) -> dict[str, Any]:
    r = dict(config)
    a = _arch_record(str(r["archetype"]), assets_dir)
    sz = SIZE[str(r["size_class"])]
    ref_out = float(a["ref_output_medium"]) * float(sz["scale"])
    demand = ref_out * float(r["demand_load"])
    nominal_capacity = ref_out * 1.28 * (float(r["oee"]) / 0.725) ** 0.45
    nominal_capacity = max(nominal_capacity, demand * 1.04)
    r.update(
        archetype_name=str(a["name"]),
        ref_output=ref_out,
        demand_fu_y=demand,
        headcount_ref=float(sz["headcount"]) * float(sz["scale"]) ** 0.15,
        ref_energy_kwh_fu=float(a["ref_energy"]),
        ref_material_kg_fu=float(a["ref_material_kg"]),
        ref_cost_eur_fu=float(a["ref_cost"]) * float(sz["cost_scale"]),
        ref_distance_km=float(a["ref_distance"]),
        ref_waste_kg_fu=float(a["ref_waste_kg"]),
        fuel_share_ref=float(a["fuel_share"]),
        complexity_index=float(a["complexity"]),
        safety_reference=float(a["safety"]),
        annual_growth_ref=float(a["growth"]),
        nominal_capacity_fu_y=float(nominal_capacity),
        n_process_steps=float(len(str(a["process_topology"]).split(" -> "))),
    )
    return r


def vsmc_current(config: dict[str, Any], assets_dir: str | Path) -> dict[str, float]:
    """Reproduce the BASE/current-state VSM-C feature bridge used for model training."""
    r = enrich_reference_features(config, assets_dir)
    a = _arch_record(str(r["archetype"]), assets_dir)
    complexity = float(a["complexity"])
    safety = float(a["safety"])
    n = int(r["n_process_steps"])
    cycle_base = BASE_CYCLE[:n].copy() * (1 + 0.30 * complexity)
    oee = float(r["oee"])
    ct = cycle_base * (0.725 / max(oee, 0.40)) ** 0.55
    load_pressure = np.clip((float(r["demand_load"]) - 0.8) / 0.4, 0, 1)
    wait_factor = (
        0.45 + 1.8 * float(r["demand_cv"]) + 1.25 * (0.90 - oee)
        + 0.55 * load_pressure + 0.45 * (1 - float(r["zD"]))
    )
    stage_wait = np.array([0.75, 1.15, 1.30, 1.45, 1.05, 0.70])[:n] * cycle_base * wait_factor
    info_w = BASE_INFO_W[:n].copy(); info_w /= info_w.sum()
    info_delay_total = (
        0.10 + 0.55 * (1 - float(r["zD"])) + 0.20 * float(r["demand_cv"]) + 0.08 * complexity
    ) * float((ct + stage_wait).sum())
    stage_wait += info_delay_total * info_w
    vat = float(ct.sum()); nvat = float(stage_wait.sum()); plt = vat + nvat
    pce = 100 * vat / plt; nvat_r = 100 * nvat / plt
    wip_i = float(np.clip(0.08 + 0.75 * float(r["demand_cv"]) + 0.30 * (1-oee) + 0.12 * float(r["distance_mult"]) + 0.18 * (1-float(r["zD"])) + 0.08 * complexity, 0.05, 1.25))
    dr = float(np.clip((0.006 + 0.018 * complexity) * (1 + 1.4 * (0.90-oee) + 1.2 * float(r["demand_cv"])) * (1 - 0.18 * float(r["zS"])), 0.003, 0.12))
    ei = float(float(a["ref_energy"]) * float(r["resource_mult"]) * (1 - 0.10 * float(r["zD"])))
    fuel_ei = ei * float(a["fuel_share"]); elec_ei = ei - fuel_ei
    renew = float(r["renewable_share"])
    ghgi = float(elec_ei * ((1-renew) * GRID_EF + renew * RENEW_EF) + fuel_ei * 0.2668)
    wi = float(float(a["ref_waste_kg"]) * float(r["resource_mult"]) * (1 - 0.42 * float(r["zC"])) * (1 + 1.8 * dr))
    idr = float(np.clip(0.08 + 0.58 * (1-float(r["zD"])) + 0.18 * float(r["demand_cv"]) + 0.06 * complexity, 0.03, 0.72))
    shr = float(np.clip(safety * (1 - 0.55 * float(r["zS"])) + 0.12 * (0.90-oee) + 0.08 * float(r["demand_cv"]), 0.02, 0.65))
    return {
        "VSM_PCE": pce, "VSM_NVAT_R": nvat_r, "VSM_OEE": oee * 100,
        "VSM_WIP_I": wip_i, "VSM_DR": dr * 100, "VSM_EI": ei,
        "VSM_GHGI": ghgi, "VSM_WI": wi, "VSM_IDR": idr * 100, "VSM_SHR": shr * 100,
    }


def diagnostic_priority(config: dict[str, Any], vsm: dict[str, float], assets_dir: str | Path) -> tuple[str, dict[str, float]]:
    a = _arch_record(str(config["archetype"]), assets_dir)
    ref_energy = float(a["ref_energy"]); ref_waste = float(a["ref_waste_kg"])
    sev = {
        "ENERGY": max(vsm["VSM_EI"] / max(ref_energy, 1e-9) - 1, vsm["VSM_GHGI"] / (ref_energy * GRID_EF + 1e-9) - 1, 0),
        "MAINT": max((70-vsm["VSM_PCE"])/45, (75-float(config["oee"])*100)/30, vsm["VSM_WIP_I"]/0.65-1, 0),
        "CIRCULAR": max(vsm["VSM_WI"] / max(ref_waste, 1e-9) - 0.65, vsm["VSM_DR"]/5-1, 0),
        "DIGITAL": max(vsm["VSM_IDR"]/28-1, (0.60-float(config["zD"]))/0.35, 0),
        "SOCIAL": max(vsm["VSM_SHR"]/25-1, (0.65-float(config["zS"]))/0.30, 0),
    }
    active = [k for k, val in sev.items() if val > 0.18]
    return ("INTEGRATED" if len(active) >= 2 else max(sev, key=sev.get)), {k: float(v) for k, v in sev.items()}


def build_model_row(
    *, archetype: str, size_class: str, strategy: str, lambda_intensity: float,
    parameters: dict[str, float], assets_dir: str | Path,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "archetype": archetype,
        "size_class": size_class,
        "strategy": strategy,
        "lambda_intensity": float(lambda_intensity),
        **{k: float(parameters[k]) for k in FORMAL_BOUNDS},
    }
    if strategy == "BASE":
        config["lambda_intensity"] = 0.0
    config = enrich_reference_features(config, assets_dir)
    config.update(vsmc_current(config, assets_dir))
    priority, severity = diagnostic_priority(config, config, assets_dir)
    config["priority_strategy"] = priority
    config.update({f"severity_{k}": v for k, v in severity.items()})
    canonical = json.dumps({k: config[k] for k in ["archetype", "size_class", "strategy", "lambda_intensity", *FORMAL_BOUNDS]}, sort_keys=True)
    config["scenario_id"] = f"ASCA-{archetype}-{strategy}-L{int(round(config['lambda_intensity']*100)):03d}-{hashlib.sha1(canonical.encode()).hexdigest()[:6].upper()}"
    return config
