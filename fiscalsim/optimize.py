"""AI reform optimiser: surrogate-assisted search for the best cost/adequacy trade-offs.

Each reform package is a set of levers (retirement age, pension formula and accrual rate,
contribution increases, and optionally pay policy and hiring). The simulator (M2-M4) scores a
package on two objectives:

    cost      average total cost 2027-2076, % of GDP (wage bill + government pension cost; median path)
    adequacy  average replacement rate of new retirees 2047-2076

The search is a small Bayesian-optimisation loop:
    1. score a space-filling sample of packages with the simulator;
    2. fit Gaussian-process surrogates for both objectives;
    3. predict every package in the lever grid, pick the ones the surrogates expect to be on (or
       near) the Pareto front, allowing for their uncertainty, and score those with the simulator;
    4. repeat, then report the front of packages actually scored by the simulator.

`full_grid` scores every package so the search can be checked against brute force.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_predict

from . import config as C
from . import simulate as S

YEARS = S.YEARS
RET_AGES = tuple(range(55, 63))
ACCRUAL = (None, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03)  # None = current 80%-of-final rule
CONTRIB = (0.0, 0.01, 0.02, 0.03, 0.04)  # total rise per side, 1 pp every 5 years
PAY = ("recent_average", "inflation", "targeted")


@dataclass(frozen=True)
class Package:
    retirement_age: int = 55
    accrual: float | None = None
    contribution_rise: float = 0.0
    salary_rule: str = "recent_average"
    restrain: bool = False

    def scenario(self, base: C.Scenario = C.SCENARIOS["S0"], code: str = "Optimised") -> C.Scenario:
        return base.with_(
            code=code, name=code,
            retirement_age_target=float(self.retirement_age),
            retirement_phase_years=10 if self.retirement_age > base.retirement_age else 0,
            pension_formula="current" if self.accrual is None else "accrual",
            accrual_rate=base.accrual_rate if self.accrual is None else self.accrual,
            contribution_step=0.01 if self.contribution_rise else 0.0, contribution_step_every=5,
            contribution_cap=self.contribution_rise,
            salary_rule=self.salary_rule, restrain_non_priority=self.restrain,
        )

    def features(self) -> list[float]:
        return [(self.retirement_age - 55) / 7, float(self.accrual is not None),
                0.0 if self.accrual is None else (self.accrual - 0.015) / 0.015,
                self.contribution_rise / 0.04,
                float(self.salary_rule == "inflation"), float(self.salary_rule == "targeted"),
                float(self.restrain)]

    def as_row(self) -> dict:
        return {"retirement_age": self.retirement_age,
                "pension_formula": "current" if self.accrual is None else "accrual",
                "accrual_rate": self.accrual, "contribution_rise_per_side": self.contribution_rise,
                "salary_rule": self.salary_rule, "restrain_non_priority": self.restrain}


def grid(pay_levers: bool = False) -> list[Package]:
    pay = PAY if pay_levers else ("recent_average",)
    restrain = (False, True) if pay_levers else (False,)
    return [Package(r, a, c, p, h) for r, a, c, p, h in itertools.product(RET_AGES, ACCRUAL, CONTRIB, pay, restrain)]


def evaluate(ctx, pkg: Package, base: C.Scenario = C.SCENARIOS["S0"]) -> dict:
    return {**pkg.as_row(), **objectives(ctx, pkg.scenario(base))}


def objectives(ctx, sc: C.Scenario) -> dict:
    res = S.run_scenario(ctx, sc)
    s = S.summarize(res)
    cost = res["series"]["total_cost_gdp"][:, 1:]  # 2027-2076
    return {"cost": float(np.median(cost.mean(axis=1))),
            "adequacy": s["avg_replacement_2047_2076"],
            "total_cost_gdp_2046": s["total_cost_gdp_2046_p50"],
            "share_on_minimum": s["share_on_minimum_2047_2076"],
            "prob_fund_depleted": s["prob_fund_depleted_by_2076"]}


def pareto_mask(cost: np.ndarray, adequacy: np.ndarray) -> np.ndarray:
    """True where no other point is at least as cheap and as adequate, and strictly better in one."""
    c, a = np.asarray(cost), np.asarray(adequacy)
    better = (c[None, :] <= c[:, None]) & (a[None, :] >= a[:, None]) & ((c[None, :] < c[:, None]) | (a[None, :] > a[:, None]))
    return ~better.any(axis=1)


def hypervolume(cost, adequacy, ref_cost: float, ref_adequacy: float) -> float:
    """Area dominated by the front, relative to a reference point (worse than every package)."""
    m = pareto_mask(cost, adequacy)
    c, a = np.asarray(cost)[m], np.asarray(adequacy)[m]
    order = np.argsort(c)
    c, a = c[order], a[order]
    area, best = 0.0, ref_adequacy
    for i in range(len(c)):
        nxt = c[i + 1] if i + 1 < len(c) else ref_cost
        best = max(best, a[i])
        area += (nxt - c[i]) * (best - ref_adequacy)
    return area


def _gp() -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(7), nu=2.5) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=C.SEED)


def search(*args, **kw) -> dict:
    """Surrogate-assisted multi-objective search. Returns scored packages, the front and diagnostics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # GP noise level at its bound on near-noiseless fits
        return _search(*args, **kw)


def _search(ctx, pay_levers: bool = False, n_initial: int = 40, rounds: int = 4, per_round: int = 12,
           base: C.Scenario = C.SCENARIOS["S0"], seed: int = C.SEED, progress=None) -> dict:
    rng = np.random.default_rng(seed)
    space = grid(pay_levers)
    X = np.array([p.features() for p in space])
    total = n_initial + rounds * per_round
    done: dict[int, dict] = {}

    def score(ids):
        for i in ids:
            done[i] = evaluate(ctx, space[i], base)
            if progress:
                progress(len(done) / total)

    # 1. space-filling start: stratify on retirement age and formula, the levers that matter most
    first = [0]  # status-quo package is always scored
    strata = pd.Series(range(len(space))).groupby([X[:, 0], X[:, 1]]).apply(list)
    while len(first) < n_initial:
        for ids in strata:
            if len(first) < n_initial:
                first.append(int(rng.choice(ids)))
    score(dict.fromkeys(first))

    for _ in range(rounds):
        ids = np.array(sorted(done))
        y_c = np.array([done[i]["cost"] for i in ids])
        y_a = np.array([done[i]["adequacy"] for i in ids])
        gp_c, gp_a = _gp().fit(X[ids], y_c), _gp().fit(X[ids], y_a)
        mc, sc = gp_c.predict(X, return_std=True)
        ma, sa = gp_a.predict(X, return_std=True)
        # optimistic front: cheaper by one sd and more adequate by one sd
        opt_c, opt_a = mc - sc, ma + sa
        cand = [i for i in np.where(pareto_mask(opt_c, opt_a))[0] if i not in done]
        if len(cand) < per_round:  # widen to the next layer of near-front packages
            rest = np.array([i for i in range(len(space)) if i not in done and i not in cand])
            if len(rest):
                layer2 = rest[pareto_mask(opt_c[rest], opt_a[rest])]
                cand += list(layer2)
        if not cand:
            break
        cand = np.array(cand)
        if len(cand) > per_round:  # spread picks along the front
            order = cand[np.argsort(opt_c[cand])]
            cand = order[np.linspace(0, len(order) - 1, per_round).round().astype(int)]
        score(dict.fromkeys(int(i) for i in cand))

    scored = pd.DataFrame([done[i] for i in sorted(done)])
    scored["on_front"] = pareto_mask(scored["cost"], scored["adequacy"])

    # surrogate accuracy on the packages it learned from (5-fold cross-validation)
    ids = np.array(sorted(done))
    fit = {}
    for col in ("cost", "adequacy"):
        y = scored[col].to_numpy()
        pred = cross_val_predict(_gp(), X[ids], y, cv=5)
        fit[col] = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    refs = pd.DataFrame([{"scenario": k, **objectives(ctx, sc)} for k, sc in C.SCENARIOS.items()]).set_index("scenario")
    return {"scored": scored, "front": scored[scored.on_front].sort_values("cost").reset_index(drop=True),
            "scenarios": refs, "grid_size": len(space), "evaluations": len(scored), "surrogate_r2": fit,
            "pay_levers": pay_levers, "paths": len(next(iter(ctx.paths.values())))}


def paths_subset(ctx, n: int = 200):
    """Same context on the first `n` Monte Carlo paths (common random numbers across packages)."""
    from types import SimpleNamespace
    return SimpleNamespace(workforce=ctx.workforce, stock0=ctx.stock0, data={"pensioners": ctx.data["pensioners"]},
                           paths={k: v[:n] for k, v in ctx.paths.items()})


def full_grid(ctx, pay_levers: bool = False, base: C.Scenario = C.SCENARIOS["S0"]) -> pd.DataFrame:
    rows = pd.DataFrame([evaluate(ctx, p, base) for p in grid(pay_levers)])
    rows["on_front"] = pareto_mask(rows["cost"], rows["adequacy"])
    return rows


def compare_to_grid(found: pd.DataFrame, allrows: pd.DataFrame) -> dict:
    """How much of the true front's hypervolume the search recovered."""
    ref_c, ref_a = allrows["cost"].max() * 1.01, allrows["adequacy"].min() - 0.01
    hv_true = hypervolume(allrows["cost"], allrows["adequacy"], ref_c, ref_a)
    hv_found = hypervolume(found["cost"], found["adequacy"], ref_c, ref_a)
    key = ["retirement_age", "pension_formula", "accrual_rate", "contribution_rise_per_side", "salary_rule",
           "restrain_non_priority"]
    ident = lambda df: df[key].apply(lambda r: "|".join(map(str, r)), axis=1)
    true_front, got = ident(allrows[allrows.on_front]), ident(found)
    return {"hypervolume_ratio": hv_found / hv_true, "true_front_size": int(allrows.on_front.sum()),
            "true_front_found": int(true_front.isin(set(got)).sum())}


def best_for_target(front: pd.DataFrame, min_adequacy: float | None = None, max_cost: float | None = None):
    """Cheapest front package meeting an adequacy floor, or most adequate one within a cost ceiling."""
    f = front
    if min_adequacy is not None:
        f = f[f.adequacy >= min_adequacy]
    if max_cost is not None:
        f = f[f.cost <= max_cost]
    if f.empty:
        return None
    return f.sort_values("cost").iloc[0] if min_adequacy is not None else f.sort_values("adequacy").iloc[-1]


def package_from_row(row) -> Package:
    acc = row["accrual_rate"]
    return Package(int(row["retirement_age"]), None if pd.isna(acc) else float(acc),
                   float(row["contribution_rise_per_side"]), row["salary_rule"], bool(row["restrain_non_priority"]))
