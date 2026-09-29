"""M2 Workforce projection: exits, promotions and hires by cohort.

Models (proposal Table 4):
  - Kaplan-Meier survival curves of time to separation (descriptive)
  - discrete-time survival model (logistic hazard on person-years)
  - XGBoost classifier on person-years
  - Markov transition matrix for promotions between frameworks
  - baseline: cohort-ratio method (historical exit rate by age band and sector)

Mandatory retirement and deaths are not learned: retirement follows the
scenario's retirement-age rule and deaths follow the mortality table, so the
learned hazard covers separations (resignations, dismissals, early retirement).

The fitted hazard is turned into a lookup over the projection grid
[sector, framework, eligible60, age, service], which `project()` rolls forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from . import config as C
from .synthetic import mortality_table

AGES = np.arange(18, 66)
SERVICE = np.arange(0, 48)
TRAIN_END = 2021  # backtest: train <= 2021, test 2022-2025


# ---------- feature engineering ----------
def features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "age": df["age"].astype(float),
        "service": df["service"].astype(float),
        "year": df["year"].astype(float),
        "eligible60": df["eligible60"].astype(int),
    }, index=df.index)
    for s in C.SECTORS:
        X[f"sector_{s}"] = (df["sector"] == s).astype(int)
    for f in C.FRAMEWORKS:
        X[f"fw_{f}"] = (df["framework"] == f).astype(int)
    return X


def separation_sample(history: pd.DataFrame) -> pd.DataFrame:
    """Person-years at risk of separation: drop mandatory-retirement and death exits."""
    h = history[~history["exit_type"].isin(["retirement", "death"])].copy()
    h["y"] = (h["exit_type"] == "separation").astype(int)
    return h


# ---------- models ----------
def kaplan_meier(history: pd.DataFrame) -> pd.DataFrame:
    """KM survival of service duration until separation, by sector (entry cohorts observed in panel)."""
    first = history.groupby("person_id").agg(sector=("sector", "first"), start=("service", "first"))
    last = history.groupby("person_id").agg(end=("service", "last"), exit_type=("exit_type", "last"))
    d = first.join(last)
    d = d[d["start"] == 0]  # hires observed from entry
    d["duration"] = d["end"] + 1
    d["event"] = d["exit_type"] == "separation"
    rows = []
    for sector, g in d.groupby("sector"):
        s = 1.0
        for t in range(1, int(g["duration"].max()) + 1):
            at_risk = (g["duration"] >= t).sum()
            events = ((g["duration"] == t) & g["event"]).sum()
            if at_risk:
                s *= 1 - events / at_risk
            rows.append({"sector": sector, "service_years": t, "survival": s, "at_risk": at_risk})
    return pd.DataFrame(rows)


LOGIT_COLS = ["age", "service", "year", "eligible60", "sector", "framework"]


class Hazard:
    """Wraps a fitted classifier so every model exposes predict_hazard(person_years)."""

    def __init__(self, model, prepare):
        self.model, self.prepare = model, prepare

    def predict_hazard(self, df):
        return self.model.predict_proba(self.prepare(df))[:, 1]


def _logit_frame(df):
    return df[LOGIT_COLS].astype({"eligible60": int})


def fit_logit(train: pd.DataFrame) -> Hazard:
    """Discrete-time hazard model: logit with splines on age and service."""
    ct = ColumnTransformer([
        ("spl", SplineTransformer(n_knots=6, degree=3), ["age", "service"]),
        ("lin", "passthrough", ["year", "eligible60"]),
        ("cat", OneHotEncoder(drop="first"), ["sector", "framework"]),
    ])
    model = make_pipeline(ct, LogisticRegression(max_iter=2000, C=10))
    model.fit(_logit_frame(train), train["y"])
    return Hazard(model, _logit_frame)


def fit_xgb(train: pd.DataFrame, seed: int = C.SEED) -> Hazard:
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.9, random_state=seed, n_jobs=4, eval_metric="logloss")
    model.fit(features(train), train["y"])
    return Hazard(model, features)


class CohortRatio:
    """Baseline: average separation rate by 5-year age band and sector over training years."""

    def __init__(self, train: pd.DataFrame):
        self.rates = train.assign(band=train["age"] // 5 * 5).groupby(["sector", "band"])["y"].mean()
        self.overall = train["y"].mean()

    def predict_hazard(self, df):
        idx = pd.MultiIndex.from_arrays([df["sector"], df["age"] // 5 * 5])
        return self.rates.reindex(idx).fillna(self.overall).to_numpy()


def promotion_matrix(history: pd.DataFrame) -> pd.DataFrame:
    """Annual Markov transition probabilities between frameworks for staff who stay."""
    stay = history[~history["exit"]]
    p = stay.groupby("framework")["promoted"].mean()
    order = list(C.FRAMEWORKS)
    M = pd.DataFrame(0.0, index=order, columns=order)
    for i, f in enumerate(order):
        up = p.get(f, 0.0) if i > 0 else 0.0
        M.loc[f, f] = 1 - up
        if i > 0:
            M.loc[f, order[i - 1]] = up
    return M


# ---------- backtest (H1) ----------
def _mape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(pred - actual) / np.abs(actual)) * 100)


def diebold_mariano(e1, e2, h: int = 1):
    """DM test on squared-error loss; positive stat means model 1 has larger loss."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    if n < 3 or np.var(d) == 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(np.var(d, ddof=1) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def backtest(history: pd.DataFrame, sample_frac: float = 0.5, seed: int = C.SEED) -> dict:
    data = separation_sample(history)
    train = data[data["year"] <= TRAIN_END].sample(frac=sample_frac, random_state=seed)
    test = data[data["year"] > TRAIN_END].copy()
    models = {"cohort_ratio": CohortRatio(train), "logit_hazard": fit_logit(train), "xgboost": fit_xgb(train)}

    # retirements and deaths in the test years come from rules both approaches share
    other_exits = history[(history["year"] > TRAIN_END) & history["exit_type"].isin(["retirement", "death"])]
    other = other_exits.groupby(["year", "sector"]).size()

    rows, auc, err = [], {}, {}
    actual = test.groupby(["year", "sector"])["y"].sum()
    for name, m in models.items():
        test[name] = m.predict_hazard(test)
        pred = test.groupby(["year", "sector"])[name].sum()
        auc[name] = float(roc_auc_score(test["y"], test[name]))
        err[name] = (pred - actual).to_numpy()
        tot_a = actual + other.reindex(actual.index).fillna(0)
        tot_p = pred + other.reindex(actual.index).fillna(0)
        rows.append({"model": name, "separations_MAPE_%": round(_mape(actual, pred), 2),
                     "separations_RMSE": round(float(np.sqrt(np.mean(err[name] ** 2))), 1),
                     "all_exits_MAPE_%": round(_mape(tot_a, tot_p), 2), "AUC": round(auc[name], 4)})
    table = pd.DataFrame(rows).set_index("model")
    for name in ("logit_hazard", "xgboost"):
        dm, p = diebold_mariano(err["cohort_ratio"], err[name])
        table.loc[name, "DM_vs_baseline"] = round(dm, 2)
        table.loc[name, "DM_p_value"] = round(p, 4)
    by_cell = test.groupby(["year", "sector"])[["y", *models]].sum().rename(columns={"y": "actual"})
    return {"table": table, "by_year_sector": by_cell.round(1)}


# ---------- projection ----------
@dataclass
class WorkforceModel:
    hazard: np.ndarray  # [sector, fw, elig, age, service]
    promo: np.ndarray  # annual promotion probability by fw (to the next framework up)
    hire_fw: np.ndarray  # share of hires by fw
    hire_age: np.ndarray  # [fw, age] entry-age distribution
    q: np.ndarray  # mortality by age index in AGES
    xgb: Hazard


def fit_projection_model(history: pd.DataFrame, sample_frac: float = 0.5, seed: int = C.SEED) -> WorkforceModel:
    data = separation_sample(history).sample(frac=sample_frac, random_state=seed)
    xgb = fit_xgb(data)
    grid = pd.MultiIndex.from_product(
        [C.SECTORS, C.FRAMEWORKS, [False, True], AGES, SERVICE],
        names=["sector", "framework", "eligible60", "age", "service"]).to_frame(index=False)
    grid["year"] = history["year"].max()  # trend held at the latest observed year
    haz = xgb.predict_hazard(grid)
    haz = np.where(grid["age"] - grid["service"] < 18, 0.0, haz)
    hazard = haz.reshape(len(C.SECTORS), len(C.FRAMEWORKS), 2, len(AGES), len(SERVICE))

    M = promotion_matrix(history)
    promo = np.array([0.0] + [M.iloc[i, i - 1] for i in range(1, 4)])
    hires = history[(history["service"] == 0) & (history["year"] >= history["year"].max() - 4)]
    hire_fw = hires["framework"].value_counts(normalize=True).reindex(C.FRAMEWORKS).fillna(0).to_numpy()
    hire_age = np.zeros((4, len(AGES)))
    for i, f in enumerate(C.FRAMEWORKS):
        a = hires.loc[hires["framework"] == f, "age"].value_counts(normalize=True)
        hire_age[i] = a.reindex(AGES).fillna(0).to_numpy()
    q = mortality_table().reindex(AGES).to_numpy()
    return WorkforceModel(hazard, promo, hire_fw, hire_age, q, xgb)


def base_stock(clean: pd.DataFrame) -> np.ndarray:
    """Cleaned base-year records -> headcount grid [sector, fw, elig, age, service]."""
    stock = np.zeros((len(C.SECTORS), len(C.FRAMEWORKS), 2, len(AGES), len(SERVICE)))
    d = clean[(clean["age"].between(AGES[0], AGES[-1])) & (clean["service"] <= SERVICE[-1])]
    s = d["sector"].map({k: i for i, k in enumerate(C.SECTORS)}).to_numpy()
    f = d["framework"].map({k: i for i, k in enumerate(C.FRAMEWORKS)}).to_numpy()
    e = d["eligible60"].astype(int).to_numpy()
    np.add.at(stock, (s, f, e, d["age"].to_numpy() - AGES[0], d["service"].to_numpy()), 1)
    return stock


def retirement_age_path(sc: C.Scenario, years: np.ndarray) -> np.ndarray:
    """Integer retirement age for non-eligible staff each year (phased schedule for S3/S5)."""
    k = years - C.BASE_YEAR
    if sc.retirement_phase_years and sc.retirement_age_target > sc.retirement_age:
        frac = np.clip(k / sc.retirement_phase_years, 0, 1)
        return np.floor(sc.retirement_age + frac * (sc.retirement_age_target - sc.retirement_age)).astype(int)
    return np.full(len(years), int(sc.retirement_age))


def project(wm: WorkforceModel, stock0: np.ndarray, sc: C.Scenario, years: np.ndarray) -> dict:
    """Roll the cohort grid forward. stock[t] is headcount at the start of year t.

    Returns stocks by year and the retirees leaving at the end of each year,
    split by [sector, fw, elig, age, service].
    """
    T = len(years)
    stocks = np.zeros((T, *stock0.shape))
    retirees = np.zeros((T, *stock0.shape))
    seps = np.zeros(T)
    deaths = np.zeros(T)
    ret_age = retirement_age_path(sc, years)
    stock = stock0.copy()
    a_idx = AGES[None, None, None, :, None]
    growth = np.array([0.0 if (sc.restrain_non_priority and s not in C.PRIORITY_SECTORS) else sc.hiring_growth
                       for s in C.SECTORS])
    for t, year in enumerate(years):
        stocks[t] = stock
        ra = np.array([ret_age[t], max(60, ret_age[t])])[None, None, :, None, None]
        retire = (a_idx >= ra) * stock
        retirees[t] = retire
        remain = stock - retire
        dead = remain * wm.q[None, None, None, :, None]
        remain -= dead
        sep = remain * wm.hazard
        remain -= sep
        seps[t], deaths[t] = sep.sum(), dead.sum()
        # promotions: fw index 1..3 move up to fw-1
        moved = remain * wm.promo[None, :, None, None, None]
        remain = remain - moved
        remain[:, :-1] += moved[:, 1:]
        # age and service advance by one year
        nxt = np.zeros_like(remain)
        nxt[..., 1:, 1:] = remain[..., :-1, :-1]
        # hires to reach target headcount by sector
        target = stock.sum(axis=(1, 2, 3, 4)) * (1 + growth)
        hires = np.clip(target - nxt.sum(axis=(1, 2, 3, 4)), 0, None)
        for s in range(len(C.SECTORS)):
            for f in range(len(C.FRAMEWORKS)):
                for e, share_e in ((0, 1 - C.BASE.share_eligible_age60), (1, C.BASE.share_eligible_age60)):
                    nxt[s, f, e, :, 0] += hires[s] * wm.hire_fw[f] * share_e * wm.hire_age[f]
        stock = nxt
    return {"years": years, "stocks": stocks, "retirees": retirees, "separations": seps,
            "deaths": deaths, "retirement_age": ret_age}


def headcount_table(proj: dict) -> pd.DataFrame:
    by_sector = proj["stocks"].sum(axis=(2, 3, 4, 5))
    df = pd.DataFrame(by_sector, index=proj["years"], columns=C.SECTORS)
    df["total"] = df.sum(axis=1)
    df["retirements"] = proj["retirees"].sum(axis=(1, 2, 3, 4, 5))
    df["separations"] = proj["separations"]
    ages = AGES[None, None, None, None, :, None]
    df["mean_age"] = (proj["stocks"] * ages).sum(axis=(1, 2, 3, 4, 5)) / df["total"]
    df["retirement_age"] = proj["retirement_age"]
    return df
