"""M6 Explanation: SHAP values for the ML models and for the simulation results.

  exit_model_shap      which features drive the M2 separation hazard (XGBoost)
  uncertainty_shap     which Monte Carlo drivers move a scenario's outcome
                       (surrogate XGBoost fitted on run-level inputs -> result)
  policy_shap          levers vs macro uncertainty across all scenarios pooled
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from . import config as C
from .m2_workforce import features, separation_sample
from .simulate import YEARS


def exit_model_shap(workforce, history: pd.DataFrame, n: int = 4000, seed: int = C.SEED):
    sep = separation_sample(history)
    X = features(sep.sample(min(n, len(sep)), random_state=seed))
    sv = shap.TreeExplainer(workforce.xgb.model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return imp.rename("mean_abs_shap"), sv, X


def run_drivers(paths: dict, until: int) -> pd.DataFrame:
    """Per-run summary of the random inputs up to `until` (the result year)."""
    i = int(np.where(YEARS == until)[0][0]) + 1
    return pd.DataFrame({
        "avg_real_gdp_growth": paths["real_growth"][:, :i].mean(axis=1) * 100,
        "avg_inflation": paths["inflation"][:, :i].mean(axis=1) * 100,
        "avg_revenue_share": paths["revenue_share"][:, :i].mean(axis=1) * 100,
        "avg_fund_real_return": paths["real_return"][:, :i].mean(axis=1) * 100,
        "mortality_level": paths["mortality_mult"],
    })


def _surrogate_shap(X: pd.DataFrame, y: np.ndarray, seed: int = C.SEED):
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=seed, n_jobs=4)
    model.fit(X, y)
    r2 = float(1 - np.var(y - model.predict(X)) / np.var(y))
    sample = X.sample(min(len(X), 3000), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(sample)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return imp, r2, sv, sample


def uncertainty_shap(result: dict, paths: dict, metric: str = "total_cost_gdp", year: int = 2046):
    X = run_drivers(paths, year)
    y = result["series"][metric][:, int(np.where(YEARS == year)[0][0])]
    return _surrogate_shap(X, y)


LEVERS = {
    "pay_indexed_to_inflation": lambda sc: float(sc.salary_rule == "inflation"),
    "pay_targeted": lambda sc: float(sc.salary_rule == "targeted"),
    "retirement_age_target": lambda sc: sc.retirement_age_target,
    "accrual_formula": lambda sc: float(sc.pension_formula == "accrual"),
    "contribution_increase": lambda sc: sc.contribution_cap,
    "hiring_restraint": lambda sc: float(sc.restrain_non_priority),
}


def policy_shap(results: dict, paths: dict, metric: str = "total_cost_gdp", year: int = 2046,
                runs_per_scenario: int = 2000, seed: int = C.SEED):
    drivers = run_drivers(paths, year)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(drivers), min(runs_per_scenario, len(drivers)), replace=False)
    i = int(np.where(YEARS == year)[0][0])
    frames, ys = [], []
    for res in results.values():
        sc = res["scenario"]
        X = drivers.iloc[pick].copy()
        for name, f in LEVERS.items():
            X[name] = f(sc)
        frames.append(X)
        ys.append(res["series"][metric][pick, i])
    X = pd.concat(frames, ignore_index=True)
    return _surrogate_shap(X, np.concatenate(ys))
