"""Retirement wave early warning: where and when today's staff will leave, by province, ministry and sector.

For every group of base-year staff (same province, ministry, sector, framework, age, service and
retirement eligibility) the engine rolls forward year by year:

    retirement   by the scenario's retirement-age rule (60 for staff eligible to work to 60)
    death        mortality table
    separation   the M2 XGBoost hazard (resignations, dismissals, early exits)

Summing the expected exits over the groups in a unit gives its expected exit wave. A unit gets a
warning when the share of its current staff expected to leave within the horizon is well above
the national share. SHAP on a surrogate of the per-group exit probability says why a unit
differs from the national average (age mix, years of service, grade, sector).

`backtest` checks the approach on the staff history: from the 2021 stock it predicts exits by
unit in 2022-2025 and compares with what happened, against two simple baselines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy import stats
from xgboost import XGBRegressor

from . import config as C
from . import m2_workforce as m2

KEYS = ["province", "ministry", "sector", "framework", "eligible60", "age", "service"]
HORIZON = 20  # years projected
HIGH, WATCH = 1.25, 1.1  # unit exit share / national exit share
WAVE_FACTOR = 1.5  # a wave year: unit exit rate at least this multiple of the national average rate
MIN_STAFF = 30  # smaller units are shown but not rated


def cells(records: pd.DataFrame) -> pd.DataFrame:
    """Staff records -> counts by unit and cohort. Keeps no names or IDs."""
    r = records.copy()
    for k, v in (("province", "Unknown"), ("ministry", "Unknown"), ("eligible60", False)):
        if k not in r:
            r[k] = v
    r = r[r["age"].between(m2.AGES[0], m2.AGES[-1]) & r["service"].between(0, m2.SERVICE[-1])]
    r["eligible60"] = r["eligible60"].astype(bool)
    c = r.groupby(KEYS, observed=True).size().rename("n").reset_index()
    for k in ("province", "ministry", "sector", "framework"):
        c[k] = c[k].astype(str).astype("category")
    return c.astype({"age": "int16", "service": "int16", "n": "int32"})


def _index(c: pd.DataFrame):
    s = c["sector"].map({k: i for i, k in enumerate(C.SECTORS)}).to_numpy()
    f = c["framework"].map({k: i for i, k in enumerate(C.FRAMEWORKS)}).to_numpy()
    return s, f, c["eligible60"].astype(int).to_numpy()


def exit_paths(c: pd.DataFrame, wm: m2.WorkforceModel, sc: C.Scenario, years: np.ndarray,
               sep_hazard=None) -> np.ndarray:
    """Expected exits of each cell's staff in each year, [cells, years] (people, not shares).

    `sep_hazard(age, service)` overrides the XGBoost separation hazard (used by the baselines).
    """
    s, f, e = _index(c)
    age0, sv0, n = c["age"].to_numpy(), c["service"].to_numpy(), c["n"].to_numpy(float)
    ra = m2.retirement_age_path(sc, years)
    present = n.copy()
    out = np.zeros((len(c), len(years)))
    for t in range(len(years)):
        age, sv = age0 + t, sv0 + t
        limit = np.where(e == 1, max(60, ra[t]), ra[t])
        a_i = np.clip(age - m2.AGES[0], 0, len(m2.AGES) - 1)
        q = wm.q[a_i]
        h = (wm.hazard[s, f, e, a_i, np.clip(sv, 0, m2.SERVICE[-1])] if sep_hazard is None
             else sep_hazard(age, sv))
        p = np.where(age >= limit, 1.0, q + (1 - q) * h)
        out[:, t] = present * p
        present = present * (1 - p)
    return out


def unit_table(c: pd.DataFrame, exits: np.ndarray, years: np.ndarray, level: str | list,
               horizon: int = 10) -> tuple[pd.DataFrame, dict]:
    """One row per unit: staff, age mix, share leaving within 5 and `horizon` years, peak and wave years."""
    level = [level] if isinstance(level, str) else list(level)
    ex = pd.DataFrame(exits, columns=years)
    ex[level] = c[level].to_numpy()
    by_year = ex.groupby(level)[list(years)].sum()
    base = c.assign(age_x=c["age"] * c["n"], old=(c["age"] >= 50) * c["n"]).groupby(level).agg(
        staff=("n", "sum"), age_x=("age_x", "sum"), old=("old", "sum"))
    rate = by_year.div(base["staff"], axis=0)  # share of today's staff leaving each year
    nat_rate = by_year.sum() / base["staff"].sum()
    nat = {"staff": float(base["staff"].sum()),
           "share_5": float(nat_rate.iloc[:5].sum()), "share_h": float(nat_rate.iloc[:horizon].sum()),
           "avg_rate": float(nat_rate.iloc[:horizon].mean()), "rate": nat_rate}
    t = pd.DataFrame({
        "staff": base["staff"],
        "mean_age": base["age_x"] / base["staff"],
        "aged_50_plus": base["old"] / base["staff"],
        "leave_5y": rate.iloc[:, :5].sum(axis=1),
        "leave_h": rate.iloc[:, :horizon].sum(axis=1),
        "peak_year": rate.iloc[:, :horizon].idxmax(axis=1).astype(int),
        "peak_rate": rate.iloc[:, :horizon].max(axis=1),
    })
    wave = rate.iloc[:, :horizon] >= WAVE_FACTOR * nat["avg_rate"]
    t["wave_start"] = [int(r.index[r.to_numpy()][0]) if r.any() else None for _, r in wave.iterrows()]
    t["wave_years"] = wave.sum(axis=1)
    t["vs_national"] = t["leave_h"] / nat["share_h"]
    t["tier"] = np.select([t["staff"] < MIN_STAFF, t["vs_national"] >= HIGH, t["vs_national"] >= WATCH],
                          ["few_staff", "high", "watch"], "normal")
    return t.sort_values("vs_national", ascending=False), {**nat, "by_year": by_year}


# ---------- explanation ----------
FEATURE_GROUPS = {"age": "age", "service": "service", "eligible60": "eligible60",
                  **{f"sector_{s}": "sector" for s in C.SECTORS}, **{f"fw_{f}": "framework" for f in C.FRAMEWORKS}}


def _cell_features(c: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({"age": c["age"].astype(float), "service": c["service"].astype(float),
                      "eligible60": c["eligible60"].astype(int)})
    for s in C.SECTORS:
        X[f"sector_{s}"] = (c["sector"] == s).astype(int).to_numpy()
    for f in C.FRAMEWORKS:
        X[f"fw_{f}"] = (c["framework"] == f).astype(int).to_numpy()
    return X


def explain(c: pd.DataFrame, exits: np.ndarray, level: str | list, horizon: int = 10, seed: int = C.SEED):
    """SHAP: why each unit's expected exit share differs from the national one, in percentage points.

    A gradient-boosted surrogate learns each cohort's probability of leaving within `horizon`
    years from its features; SHAP splits the gap between a unit's average and the national
    average into age, service, grade, sector and eligibility. Exit probabilities depend only on
    those features, so the surrogate and SHAP run once per distinct cohort and units are
    staff-weighted averages. Returns (table, surrogate R²).
    """
    level = [level] if isinstance(level, str) else list(level)
    feats = ["sector", "framework", "eligible60", "age", "service"]
    d = c[list(dict.fromkeys(feats + level + ["n"]))].assign(y=exits[:, :horizon].sum(axis=1) / c["n"].to_numpy())
    combo = d.groupby(feats, observed=True).agg(y=("y", "first"), n=("n", "sum")).reset_index()
    X, y, w = _cell_features(combo), combo["y"].to_numpy(), combo["n"].to_numpy(float)
    model = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=4)
    model.fit(X, y, sample_weight=w)
    pred = model.predict(X)
    r2 = float(1 - np.average((y - pred) ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    sv = pd.DataFrame(shap.TreeExplainer(model).shap_values(X), columns=X.columns)
    sv = sv.T.groupby(FEATURE_GROUPS).sum().T  # one column per feature group
    groups = list(sv.columns)
    sv = sv.add_prefix("shap_")
    cell_sv = d.merge(pd.concat([combo[feats], sv], axis=1), on=feats, how="left")
    wsum = cell_sv[list(sv.columns)].mul(cell_sv["n"], axis=0)
    unit = wsum.groupby([cell_sv[k] for k in level]).sum().div(cell_sv.groupby(level)["n"].sum(), axis=0)
    national = wsum.sum() / cell_sv["n"].sum()
    return ((unit - national) * 100).set_axis(groups, axis=1), r2


# ---------- backtest ----------
def _wape(actual, pred):
    return float(np.abs(pred - actual).sum() / actual.sum() * 100)


def backtest(history: pd.DataFrame, origin: int | None = None, level=("province", "sector"),
             top: float = 0.2, sample_frac: float = 0.5) -> dict:
    """Predict exits by unit and year after `origin` from the staff in post then, and score them.

    model        age rule + mortality + XGBoost separation hazard trained on years <= origin
    past_rate    each unit's average exit rate over the five years to origin, carried forward
    rule_plus_avg   age rule + mortality + each unit's past separation rate (no machine learning)
    Scores: WAPE of exits by unit and year, rank correlation of 4-year exit shares, and the
    warning hit rate (share of the units flagged in the top `top` that really were in the top).
    """
    origin = m2.backtest_origin(history) if origin is None else origin
    level = list(level)
    past = history[history["year"] <= origin]
    wm = m2.fit_projection_model(past, sample_frac=sample_frac)
    stay = past[(past["year"] == origin) & ~past["exit"]].copy()
    stay["age"] += 1
    stay["service"] += 1
    years = np.arange(origin + 1, history["year"].max() + 1)
    later = history[history["year"].isin(years) & history["exit"] & history["person_id"].isin(stay["person_id"])]
    actual = later.groupby([*level, "year"]).size().unstack("year").reindex(columns=years).fillna(0)

    c = cells(stay)
    window = past[past["year"] > origin - 5]
    exit_rate = window.groupby(level)["exit"].mean()
    at_risk = window[~window["exit_type"].isin(["retirement", "death"])]
    sep_rate = at_risk.groupby(level).apply(lambda g: (g["exit_type"] == "separation").mean(), include_groups=False)
    unit_of = pd.MultiIndex.from_frame(c[level])
    s0 = C.SCENARIOS["S0"]

    preds = {"model": exit_paths(c, wm, s0, years)}
    r = exit_rate.reindex(unit_of).to_numpy()
    preds["past_rate"] = c["n"].to_numpy()[:, None] * r[:, None] * (1 - r[:, None]) ** np.arange(len(years))[None, :]
    h_unit = sep_rate.reindex(unit_of).to_numpy()
    preds["rule_plus_avg"] = exit_paths(c, wm, s0, years, sep_hazard=lambda age, sv: h_unit)

    staff = c.groupby(level)["n"].sum()
    act = actual.reindex(staff.index).fillna(0)
    act_share = act.sum(axis=1) / staff
    k = max(1, int(round(top * len(staff))))
    true_top = set(act_share.nlargest(k).index)
    rows, errs = [], {}
    for name, p in preds.items():
        pu = pd.DataFrame(p, columns=years).assign(**{lv: c[lv].to_numpy() for lv in level}).groupby(level)[list(years)].sum()
        pu = pu.reindex(staff.index)
        errs[name] = (pu - act).to_numpy().ravel()
        share = pu.sum(axis=1) / staff
        rows.append({"model": name, "WAPE_%": round(_wape(act.to_numpy(), pu.to_numpy()), 2),
                     "rank_corr": round(float(stats.spearmanr(share, act_share).statistic), 3),
                     f"top_{int(top * 100)}pct_hit_rate": round(len(true_top & set(share.nlargest(k).index)) / k, 3)})
    table = pd.DataFrame(rows).set_index("model")
    for name in ("past_rate", "rule_plus_avg"):
        dm, p = m2.diebold_mariano(errs[name], errs["model"])
        table.loc[name, "DM_vs_model"] = round(dm, 2)
        table.loc[name, "DM_p_value"] = round(p, 4)
    return {"table": table, "units": len(staff), "unit_years": int(act.size), "origin": origin,
            "years": [int(y) for y in years]}
