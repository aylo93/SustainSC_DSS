from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .core import FORMAL_BOUNDS, SIZE_CLASSES, STRATEGIES

OUTPUT_UNITS = {
    "milp_total_cost": "EUR/y",
    "milp_total_co2_t": "t CO2e/y",
    "milp_energy_kwh": "kWh/y",
    "milp_transport_work_tkm": "t·km/y",
    "milp_capacity_util": "fraction",
    "des_throughput_rate_mean": "FU/min",
    "des_service_pct_mean": "%",
    "des_mean_lead_time_mean": "min",
    "des_mean_wait_mean": "min",
    "des_logistics_cost_mean": "EUR/y",
    "des_transport_co2_t_mean": "t CO2e/y",
    "sd_ghg_2030_t": "t CO2e/y",
    "sd_ghg_2035_t": "t CO2e/y",
    "sd_cum_ghg_t": "t CO2e (2026–2035)",
    "sd_cum_output": "FU (2026–2035)",
    "sd_oee_2035": "%",
    "sd_digital_2035": "%",
}

BOUNDED_OUTPUTS = {
    "milp_capacity_util": (0.0, 1.0),
    "des_service_pct_mean": (0.0, 100.0),
    "sd_oee_2035": (0.0, 100.0),
    "sd_digital_2035": (0.0, 100.0),
}


@dataclass
class DomainCheck:
    status: str
    formal_ok: bool
    finite_envelope_ok: bool
    near_boundary: bool
    violations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def surrogate_allowed(self) -> bool:
        return self.status == "INSIDE_VALIDATED_DOMAIN"


class MetamodelRouter:
    def __init__(self, assets_dir: str | Path):
        self.assets_dir = Path(assets_dir)
        self.meta_dir = self.assets_dir / "03_metamodels"
        self.design_dir = self.assets_dir / "01_design"
        self.registry = json.loads((self.meta_dir / "model_registry.json").read_text(encoding="utf-8"))
        self.validation = pd.read_csv(self.meta_dir / "holdout_validation_metrics.csv")
        self.development = pd.read_csv(self.design_dir / "development_136.csv")
        vsm = pd.read_csv(self.design_dir / "vsmc_baseline_vectors.csv")
        dev_ids = set(self.development["config_id"].astype(str))
        vsm = vsm[vsm["config_id"].astype(str).isin(dev_ids)].copy()
        self.dev_full = self.development.merge(
            vsm.drop(columns=[c for c in ["company_id", "archetype", "strategy"] if c in vsm.columns]),
            on="config_id", how="left"
        )

    def environment_status(self) -> dict[str, Any]:
        stored = next(iter(self.registry.values())).get("software", {})
        return {
            "current_python": platform.python_version(),
            "trained_python": stored.get("python"),
            "current_sklearn": sklearn.__version__,
            "trained_sklearn": stored.get("scikit_learn"),
            "sklearn_exact_match": sklearn.__version__ == stored.get("scikit_learn"),
        }

    def check_domain(self, row: dict[str, Any], boundary_fraction: float = 0.03) -> DomainCheck:
        violations: list[dict[str, Any]] = []
        near = False
        formal_ok = True
        if row.get("archetype") not in set(self.development["archetype"]):
            formal_ok = False; violations.append({"feature": "archetype", "value": row.get("archetype"), "rule": "unknown archetype"})
        if row.get("size_class") not in SIZE_CLASSES:
            formal_ok = False; violations.append({"feature": "size_class", "value": row.get("size_class"), "rule": "unknown size"})
        if row.get("strategy") not in STRATEGIES:
            formal_ok = False; violations.append({"feature": "strategy", "value": row.get("strategy"), "rule": "unknown strategy"})
        lam = float(row.get("lambda_intensity", np.nan))
        if not np.isfinite(lam) or not (0.0 <= lam <= 1.0):
            formal_ok = False; violations.append({"feature": "lambda_intensity", "value": lam, "rule": "formal [0,1]"})
        if row.get("strategy") == "BASE" and abs(lam) > 1e-12:
            formal_ok = False; violations.append({"feature": "lambda_intensity", "value": lam, "rule": "BASE requires 0"})
        for feature, (lo, hi) in FORMAL_BOUNDS.items():
            val = float(row.get(feature, np.nan))
            if not np.isfinite(val) or val < lo or val > hi:
                formal_ok = False
                violations.append({"feature": feature, "value": val, "min": lo, "max": hi, "domain": "formal"})
            elif hi > lo and min((val-lo)/(hi-lo), (hi-val)/(hi-lo)) < boundary_fraction:
                near = True

        finite_ok = True
        arch_dev = self.dev_full[self.dev_full["archetype"] == row.get("archetype")]
        # Check the actual finite training envelope for all primary factors and all current-state VSM-C predictors.
        finite_features = [*FORMAL_BOUNDS, "VSM_PCE", "VSM_NVAT_R", "VSM_OEE", "VSM_WIP_I", "VSM_DR", "VSM_EI", "VSM_GHGI", "VSM_WI", "VSM_IDR", "VSM_SHR"]
        if arch_dev.empty:
            finite_ok = False
        else:
            for feature in finite_features:
                if feature not in arch_dev.columns or feature not in row:
                    continue
                lo = float(arch_dev[feature].min()); hi = float(arch_dev[feature].max()); val = float(row[feature])
                tol = max(abs(hi-lo), 1.0) * 1e-9
                if val < lo-tol or val > hi+tol:
                    finite_ok = False
                    violations.append({"feature": feature, "value": val, "min": lo, "max": hi, "domain": "finite_archetype_training_envelope"})
                elif hi > lo and min((val-lo)/(hi-lo), (hi-val)/(hi-lo)) < boundary_fraction:
                    near = True

        if not formal_ok:
            status = "OUTSIDE_FORMAL_DOMAIN"
        elif not finite_ok:
            status = "OUTSIDE_FINITE_TRAINING_ENVELOPE"
        elif near:
            status = "NEAR_BOUNDARY"
        else:
            status = "INSIDE_VALIDATED_DOMAIN"
        notes = []
        if status == "NEAR_BOUNDARY":
            notes.append("Section 8.15 routing prefers the parent model near applicability boundaries.")
        if status.startswith("OUTSIDE"):
            notes.append("No surrogate accuracy claim is permitted outside the validated applicability domain.")
        return DomainCheck(status, formal_ok, finite_ok, near, violations, notes)

    @lru_cache(maxsize=64)
    def _load_model(self, registry_key: str) -> dict[str, Any]:
        spec = self.registry[registry_key]
        path = self.assets_dir / spec["model_file"].replace("03_metamodels/", "03_metamodels/")
        if not path.exists():
            # Assets bundle stores model_file path under asca_assets/03_metamodels exactly as in the source package.
            path = self.meta_dir / "models" / Path(spec["model_file"]).name
        return joblib.load(path)

    def _validation_row(self, module: str, target: str) -> pd.Series:
        d = self.validation[(self.validation.module == module) & (self.validation.target == target)]
        if d.empty:
            raise KeyError(f"Validation status missing for {module}:{target}")
        return d.iloc[0]

    @staticmethod
    def _postprocess(target: str, value: float) -> tuple[float, bool]:
        changed = False
        if target in BOUNDED_OUTPUTS:
            lo, hi = BOUNDED_OUTPUTS[target]
            clipped = float(np.clip(value, lo, hi)); changed = clipped != value; value = clipped
        elif value < 0:
            value = 0.0; changed = True
        return float(value), changed

    def predict(self, row: dict[str, Any], domain: DomainCheck) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        X = pd.DataFrame([row])
        for key, spec in self.registry.items():
            module, target = key.split(":", 1)
            vr = self._validation_row(module, target)
            validation_status = str(vr.study_acceptance)
            rec = {
                "module": module,
                "target": target,
                "unit": OUTPUT_UNITS.get(target, ""),
                "algorithm": spec["algorithm"],
                "holdout_nrmse": float(vr.nrmse),
                "holdout_spearman": float(vr.spearman),
                "validation_status": validation_status,
                "domain_status": domain.status,
                "prediction": np.nan,
                "route": "PARENT_MODEL_REQUIRED",
                "postprocess_applied": False,
                "message": "",
            }
            if not domain.surrogate_allowed:
                rec["message"] = "Surrogate blocked by applicability-domain gate."
            elif validation_status == "FULL_MODEL_REQUIRED":
                rec["message"] = "Output-level validation does not permit surrogate substitution."
            else:
                try:
                    payload = self._load_model(key)
                    features = payload.get("features", spec["features"])
                    raw = float(payload["bundle"].predict(X[features])[0])
                    value, changed = self._postprocess(target, raw)
                    rec["prediction"] = value
                    rec["postprocess_applied"] = changed
                    if validation_status == "PASS":
                        rec["route"] = "SURROGATE_SCREENING"
                        rec["message"] = "Validated for rapid in-domain screening; confirm high-impact final decisions with parent model."
                    else:
                        rec["route"] = "SURROGATE_EXPLORATORY"
                        rec["message"] = "Conditional surrogate: exploratory screening only; use parent model near decision/ranking boundaries."
                except Exception as exc:
                    rec["route"] = "METAMODEL_LOAD_ERROR"
                    rec["message"] = f"Model could not be loaded/executed: {type(exc).__name__}: {exc}"
            records.append(rec)
        return pd.DataFrame(records)
