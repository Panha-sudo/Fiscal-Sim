"""Pension fund risk alert: the chance that the NSSF-C civil servant fund runs out, by year.

For a scenario that has been simulated, the chance is simply the share of Monte Carlo runs in
which the reserve is used up by that year. That is not AI. The AI part learns from many
simulated reform packages and Monte Carlo runs so that it can answer questions the runs cannot
answer directly:

    risk model      gradient-boosted classifier: P(fund runs out by year Y) for any set of policy
                    levers and any assumption about average inflation, fund return, growth and
                    mortality over 2026-2045. Its calibration (do stated chances come true that
                    often?) is checked on held-out packages. An extra isotonic calibration step is
                    scored as well; it did not improve the held-out score, so it is not applied
    year range      the risk curve gives the 5th and 95th percentile run-out years; a conformal
                    correction fitted on held-out packages widens them just enough that 90% of
                    outcomes fall inside on new data
    drivers         SHAP on the run-out-year model: which levers and assumptions move it

Everything is scored on reform packages the models never saw, against simple baselines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from . import config as C
from . import simulate as S

YEARS = S.YEARS
CHECK_YEARS = (2030, 2035, 2040, 2045, 2050, 2060, 2070, 2076)
NEVER = C.END_YEAR + 1  # "not by 2076"
WINDOW = (C.BASE_YEAR, 2045)  # assumptions are averages over this window
GREEN, RED = 0.10, 0.50  # traffic light: green below 10%, red from 50%
ALPHA = 0.10  # 90% ranges

LEVER_COLS = ["retirement_age", "accrual_formula", "accrual_rate", "contribution_rise", "employee_rate",
              "employer_rate", "pay_indexed_to_inflation", "pay_targeted", "fixed_raise", "hiring_growth",
              "hiring_restraint"]
DRIVER_COLS = ["avg_inflation", "avg_fund_real_return", "avg_real_growth", "mortality_level"]
FEATURES = LEVER_COLS + DRIVER_COLS


def lever_features(sc: C.Scenario) -> dict:
    return {"retirement_age": sc.retirement_age_target, "accrual_formula": float(sc.pension_formula == "accrual"),
            "accrual_rate": sc.accrual_rate if sc.pension_formula == "accrual" else 0.0,
            "contribution_rise": sc.contribution_cap if sc.contribution_step else 0.0,
            "employee_rate": sc.employee_contribution, "employer_rate": sc.employer_contribution,
            "pay_indexed_to_inflation": float(sc.salary_rule == "inflation"),
            "pay_targeted": float(sc.salary_rule == "targeted"), "fixed_raise": sc.recent_raise,
            "hiring_growth": sc.hiring_growth, "hiring_restraint": float(sc.restrain_non_priority)}


def driver_features(paths: dict) -> pd.DataFrame:
    i = slice(0, int(np.where(YEARS == WINDOW[1])[0][0]) + 1)
    return pd.DataFrame({"avg_inflation": paths["inflation"][:, i].mean(axis=1) * 100,
                         "avg_fund_real_return": paths["real_return"][:, i].mean(axis=1) * 100,
                         "avg_real_growth": paths["real_growth"][:, i].mean(axis=1) * 100,
                         "mortality_level": paths["mortality_mult"]})


def central_drivers(paths: dict) -> dict:
    return driver_features({k: v[:1] for k, v in paths.items()}).iloc[0].to_dict()


def random_scenario(rng, i: int) -> C.Scenario:
    age = int(rng.integers(55, 63))
    accrual = rng.random() < 0.6
    rise = float(rng.choice([0.0, 0.01, 0.02, 0.03, 0.04]))
    return C.SCENARIOS["S0"].with_(
        code=f"R{i}", name=f"R{i}", retirement_age_target=float(age), retirement_phase_years=10 if age > 55 else 0,
        pension_formula="accrual" if accrual else "current",
        accrual_rate=float(rng.uniform(0.015, 0.03)) if accrual else 0.02,
        contribution_step=0.01 if rise else 0.0, contribution_step_every=5, contribution_cap=rise,
        employee_contribution=float(rng.uniform(0.04, 0.08)), employer_contribution=float(rng.uniform(0.08, 0.16)),
        salary_rule=str(rng.choice(["recent_average", "inflation", "targeted"], p=[0.5, 0.3, 0.2])),
        recent_raise=float(rng.uniform(0.03, 0.08)), hiring_growth=float(rng.uniform(-0.01, 0.03)),
        restrain_non_priority=bool(rng.random() < 0.2))


def depletion(res: dict) -> np.ndarray:
    d = res["depletion_year"].astype(float)
    return np.where(d > 0, d, NEVER)


def training_runs(ctx, n_random: int = 900, paths: int = 40, seed: int = C.SEED, progress=None) -> pd.DataFrame:
    """One row per (reform package, Monte Carlo run): levers, assumptions and the run-out year."""
    rng = np.random.default_rng(seed)
    sub = {k: v[:paths] for k, v in ctx.paths.items()}
    drivers = driver_features(sub)
    settings = list(C.SCENARIOS.values()) + [random_scenario(rng, i) for i in range(n_random)]
    frames = []
    for j, sc in enumerate(settings):
        res = S.run_scenario(ctx, sc, sub)
        frames.append(drivers.assign(setting=sc.code, **lever_features(sc), depletion_year=depletion(res)))
        if progress:
            progress((j + 1) / len(settings))
    return pd.concat(frames, ignore_index=True)


def _by_year(runs: pd.DataFrame, per_run: int | None = None, seed: int = C.SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Stack runs with a year: features + year, label = ran out by that year.

    Scoring uses every check year. Fitting uses `per_run` random years per run, so the model
    sees every year from 2026 to 2076 and gives a smooth yearly risk curve.
    """
    if per_run is None:
        years = np.repeat(np.array(CHECK_YEARS)[:, None], len(runs), axis=1)
    else:
        years = np.random.default_rng(seed).integers(C.BASE_YEAR, C.END_YEAR + 1, (per_run, len(runs)))
    X = pd.concat([runs[FEATURES].assign(year=yr) for yr in years], ignore_index=True)
    y = np.concatenate([(runs["depletion_year"].to_numpy() <= yr) for yr in years]).astype(int)
    return X, y


def _ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[k == b].mean() - p[k == b].mean()) * (k == b).mean() for b in range(bins) if (k == b).any()))


class RiskModel:
    """Calibrated P(fund runs out by year) and a conformal range for the run-out year."""

    GRID = np.arange(C.BASE_YEAR, NEVER + 1)  # yearly, NEVER = not by 2076

    def __init__(self, seed: int = C.SEED):
        mono = tuple([0] * len(FEATURES) + [1])  # more years can only add risk
        self.clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 monotone_constraints=mono, random_state=seed, n_jobs=4, eval_metric="logloss")
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.point = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=4)
        self.q = 0.0

    def fit(self, train: pd.DataFrame, cal: pd.DataFrame) -> "RiskModel":
        X, y = _by_year(train, per_run=12)
        self.clf.fit(X, y)
        Xc, yc = _by_year(cal, per_run=12, seed=C.SEED + 1)
        self.iso.fit(self.prob(Xc), yc)  # scored in evaluate() as an alternative
        self.point.fit(train[FEATURES], train["depletion_year"])  # explained with SHAP
        lo, hi, _ = self.quantiles(cal)
        yv = cal["depletion_year"].to_numpy()
        score = np.maximum(lo - yv, yv - hi)
        n = len(score)
        self.q = float(np.quantile(score, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))
        return self

    def prob(self, X: pd.DataFrame) -> np.ndarray:
        """X has FEATURES + year."""
        return self.clf.predict_proba(X[FEATURES + ["year"]])[:, 1]

    def cdf(self, df: pd.DataFrame) -> np.ndarray:
        """P(run out by each year of GRID), [rows, years], non-decreasing."""
        X = pd.concat([df[FEATURES].assign(year=min(y, C.END_YEAR)) for y in self.GRID], ignore_index=True)
        P = self.prob(X).reshape(len(self.GRID), len(df)).T
        P[:, -1] = 1.0  # by "never" everything has happened
        return np.maximum.accumulate(P, axis=1)

    def quantiles(self, df: pd.DataFrame, a: float = ALPHA):
        P = self.cdf(df)
        first = lambda level: self.GRID[np.argmax(P >= level, axis=1)]
        return first(a / 2), first(1 - a / 2), first(0.5)

    def curve(self, features: dict, years=CHECK_YEARS) -> pd.Series:
        P = self.cdf(pd.DataFrame([features]))[0]
        return pd.Series([P[self.GRID == y][0] for y in years], index=list(years))

    def year_range(self, features: dict) -> tuple[int, int, int]:
        """Most likely run-out year and a 90% range; NEVER means not by 2076."""
        lo, hi, med = self.quantiles(pd.DataFrame([features]))
        clip = lambda v: int(min(max(v, C.BASE_YEAR), NEVER))
        return clip(med[0]), clip(lo[0] - self.q), clip(hi[0] + self.q)

    def shap_global(self, runs: pd.DataFrame, n: int = 3000, seed: int = C.SEED) -> pd.Series:
        X = runs[FEATURES].sample(min(n, len(runs)), random_state=seed)
        sv = shap.TreeExplainer(self.point).shap_values(X)
        return pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    def shap_local(self, features: dict) -> pd.Series:
        """Years each input moves the predicted run-out year away from the average package."""
        sv = shap.TreeExplainer(self.point).shap_values(pd.DataFrame([features])[FEATURES])[0]
        return pd.Series(sv, index=FEATURES)


def evaluate(model: RiskModel, train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Score on reform packages the models never saw, against simple baselines."""
    X, y = _by_year(test)
    Xt, yt = _by_year(train, per_run=12)
    base_rate = pd.Series(yt).groupby(Xt["year"].to_numpy()).mean()
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xt, yt)
    preds = {"ai_risk_model": model.prob(X),
             "ai_plus_isotonic": model.iso.predict(model.prob(X)),
             "logistic_regression": logit.predict_proba(X)[:, 1],
             "year_only_base_rate": base_rate.reindex(X["year"]).to_numpy()}
    rows = []
    for name, p in preds.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        # per package and year: predicted chance vs share of its simulated runs that ran out
        pkg = pd.DataFrame({"s": np.tile(test["setting"].to_numpy(), len(CHECK_YEARS)), "yr": X["year"], "p": p, "y": y})
        g = pkg.groupby(["s", "yr"])[["p", "y"]].mean()
        rows.append({"model": name, "brier": round(brier_score_loss(y, p), 4), "log_loss": round(log_loss(y, p, labels=[0, 1]), 4),
                     "AUC": round(roc_auc_score(y, p), 4) if 0 < y.mean() < 1 else np.nan, "ECE": round(_ece(y, p), 4),
                     "package_error_pp": round(float((g["p"] - g["y"]).abs().mean() * 100), 2)})
    clf_table = pd.DataFrame(rows).set_index("model")

    yv = test["depletion_year"].to_numpy()
    lo, hi, med = model.quantiles(test)
    lo, hi = lo - model.q, hi + model.q
    lin = LinearRegression().fit(train[FEATURES], train["depletion_year"])
    # baseline: the same split-conformal step around a plain linear model (one width for all)
    res = np.abs(cal["depletion_year"] - lin.predict(cal[FEATURES]))
    half = float(np.quantile(res, min(1.0, np.ceil((len(res) + 1) * (1 - ALPHA)) / len(res))))
    lp = lin.predict(test[FEATURES])
    rng_table = pd.DataFrame([
        {"model": "ai_risk_curve_conformal", "coverage": round(float(((yv >= lo) & (yv <= hi)).mean()), 3),
         "avg_width_years": round(float((hi - lo).mean()), 2), "median_error_years": round(float(np.abs(med - yv).mean()), 2)},
        {"model": "linear_conformal", "coverage": round(float(((yv >= lp - half) & (yv <= lp + half)).mean()), 3),
         "avg_width_years": round(float(2 * half), 2), "median_error_years": round(float(np.abs(lp - yv).mean()), 2)},
    ]).set_index("model")
    return {"classifier": clf_table, "year_range": rng_table, "target_coverage": 1 - ALPHA,
            "test_packages": int(test["setting"].nunique()), "train_packages": int(train["setting"].nunique())}


def build(ctx, n_random: int = 900, paths: int = 40, seed: int = C.SEED, progress=None) -> dict:
    """Simulate packages, split them 70/15/15 into train, calibration and test, fit and score."""
    return from_runs(training_runs(ctx, n_random, paths, seed, progress), ctx.paths, seed)


def from_runs(runs: pd.DataFrame, paths: dict, seed: int = C.SEED) -> dict:
    ids = np.array(list(runs["setting"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    part = {s: ("train" if i < 0.7 * n else "cal" if i < 0.85 * n else "test") for i, s in enumerate(ids)}
    split = runs["setting"].map(part)
    train, cal, test = runs[split == "train"], runs[split == "cal"], runs[split == "test"]
    model = RiskModel(seed).fit(train, cal)
    return {"model": model, "evaluation": evaluate(model, train, cal, test), "shap": model.shap_global(runs),
            "central": central_drivers(paths), "driver_ranges": driver_features(paths).quantile([0.01, 0.99])}


def traffic_light(p: float) -> str:
    return "red" if p >= RED else "amber" if p >= GREEN else "green"


def simulated_curve(res: dict, years=CHECK_YEARS) -> pd.Series:
    d = depletion(res)
    return pd.Series([(d <= y).mean() for y in years], index=list(years))
