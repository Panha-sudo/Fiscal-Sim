"""Hierarchical wage bill forecasting by province and ministry.

Every province x ministry cell gets its own forecast, and so does every total above it:
each sector, ministry and province, and the national wage bill. Forecast separately,
these do not add up (the provinces' forecasts sum to a different national total than the
national forecast). Reconciliation adjusts all of them together so that every total is
exactly the sum of its parts, using what each forecast tells about the others.

Methods compared on held-out years (rolling origins 2018-2024 for the 2012-2025 synthetic panel,
one to three years ahead):
  base         each series forecast on its own (not coherent; for reference)
  bottom_up    forecast the cells, add them up
  top_down     forecast the national total, split it by each cell's average historical share
  ols, wls_struct, wls_var, mint_shrink
               optimal reconciliation (Hyndman et al. 2011; Wickramasuriya et al. 2019),
               differing in how much weight each series' forecast gets

Series: the annual wage bill at the 2026 pay scale (basic salary by framework and service
step, plus sector allowances) from the staff history 2012-2025. Holding the pay scale fixed
keeps what a unit forecast can know (headcount and grade mix) apart from pay-scale rises,
which are policy choices that M3 applies on top.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config as C
from .synthetic import basic_salary

LEVELS = ("national", "sector", "ministry", "province", "cell")
METHODS = ("base", "bottom_up", "top_down", "ols", "wls_struct", "wls_var", "mint_shrink")
COHERENT = METHODS[1:]
FIRST_ORIGIN, MAX_H, FUTURE_H = 2018, 3, 5


def first_origin(years) -> int:
    """Seven rolling origins (2018 for 2012-2025), each with at least five training years."""
    return max(int(years[0]) + 4, int(years[-1]) - 7)


# ---------- series and structure ----------
def cell_series(history: pd.DataFrame) -> pd.DataFrame:
    """Wage bill in bn riel per (sector, ministry, province) cell (rows) and year (columns)."""
    allowance = history["sector"].map(C.BASE.allowance_rate).to_numpy()
    pay = basic_salary(history["framework"], history["service"]) * (1 + allowance) * 12 / 1e9
    g = history.assign(pay=pay).groupby(["sector", "ministry", "province", "year"])["pay"].sum()
    return g.unstack("year").fillna(0.0).sort_index()


@dataclass
class Hierarchy:
    ids: list          # series id per row of S, e.g. "province:Kep"
    level: np.ndarray  # level name per row
    label: list        # display name per row
    S: np.ndarray      # [n_series, n_cells] summing matrix; cells are the last n_cells rows


def structure(cells: pd.Index) -> Hierarchy:
    """National, sector, ministry and province totals over the cells, then the cells themselves."""
    idx = cells.to_frame(index=False)
    rows, ids, level, label = [np.ones(len(idx))], ["national"], ["national"], ["Cambodia"]
    for lv in ("sector", "ministry", "province"):
        for v in sorted(idx[lv].unique()):
            rows.append((idx[lv] == v).to_numpy(float))
            ids.append(f"{lv}:{v}")
            level.append(lv)
            label.append(v)
    rows.extend(np.eye(len(idx)))
    for s, m, p in cells:
        ids.append(f"cell:{s}|{m}|{p}")
        level.append("cell")
        label.append(f"{p} · {m}")
    return Hierarchy(ids, np.array(level), label, np.vstack(rows))


def all_series(cells: pd.DataFrame, H: Hierarchy | None = None) -> pd.DataFrame:
    """Every series in the hierarchy (rows = ids, columns = years)."""
    H = H or structure(cells.index)
    return pd.DataFrame(H.S @ cells.to_numpy(), index=H.ids, columns=cells.columns)


# ---------- base forecasts ----------
def ets(y: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Damped-trend exponential smoothing: forecasts [h] and in-sample one-step residuals [len(y)]."""
    if np.allclose(y, y[0]):
        return np.repeat(y[-1], h), np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
            return np.maximum(fit.forecast(h), 0), y - fit.fittedvalues
        except Exception:  # too short or degenerate: random walk with drift
            d = (y[-1] - y[0]) / max(len(y) - 1, 1)
            return y[-1] + d * np.arange(1, h + 1), np.r_[0, np.diff(y) - d]


def ets_base(Y: np.ndarray, h: int, origin: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Base forecasts [n, h] and residuals [n, T] for every series (rows of Y)."""
    out = [ets(y, h) for y in Y]
    return np.array([f for f, _ in out]), np.array([r for _, r in out])


def ets_grid_base(Y: np.ndarray, h: int, origin: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The same damped-trend smoothing for every series at once, with its three parameters picked from a
    grid by in-sample squared error instead of numerical optimisation. Used when the dashboard reruns the
    forecast on an uploaded history: it takes a second where `ets_base` takes minutes."""
    Y = np.asarray(Y, float)
    a, b, p = np.meshgrid(np.linspace(0.05, 1, 20), np.linspace(0, 1, 11), (0.8, 0.85, 0.9, 0.95, 0.98), indexing="ij")
    a, b, p = (v.reshape(-1, 1) for v in (a, b, p))  # [grid, 1], broadcast over series
    level = np.broadcast_to(Y[:, 0], (len(a), len(Y))).copy()
    trend = np.broadcast_to(Y[:, 1] - Y[:, 0] if Y.shape[1] > 1 else 0 * Y[:, 0], level.shape).copy()
    res = np.zeros((len(a), *Y.shape))
    for t in range(1, Y.shape[1]):
        fit = level + p * trend
        res[:, :, t] = Y[:, t] - fit
        new = a * Y[:, t] + (1 - a) * fit
        trend = b * (new - level) + (1 - b) * p * trend
        level = new
    best = (res ** 2).sum(axis=2).argmin(axis=0)  # per series
    i = np.arange(len(Y))
    damp = np.cumsum(p[best] ** np.arange(1, h + 1), axis=1)  # phi + ... + phi^k
    f = level[best, i][:, None] + damp * trend[best, i][:, None]
    return np.maximum(f, 0), res[best, i]


# ---------- reconciliation ----------
def shrink_cov(res: np.ndarray) -> np.ndarray:
    """Schafer-Strimmer shrinkage of the residual covariance towards its diagonal (MinT-shrink)."""
    x = res.T  # [T, n]
    T = x.shape[0]
    cov = x.T @ x / T
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    xs = x / sd
    cor = cov / np.outer(sd, sd)
    v = (xs ** 2).T @ (xs ** 2) / (T * (T - 1)) - (xs.T @ xs) ** 2 / (T ** 2 * (T - 1))
    np.fill_diagonal(v, 0)
    d = cor ** 2
    np.fill_diagonal(d, 0)
    lam = float(np.clip(v.sum() / d.sum(), 0, 1)) if d.sum() > 0 else 1.0
    return lam * np.diag(np.diag(cov)) + (1 - lam) * cov


def weights(method: str, S: np.ndarray, res: np.ndarray | None) -> np.ndarray:
    n = S.shape[0]
    if method == "ols":
        return np.eye(n)
    if method == "wls_struct":
        return np.diag(S.sum(axis=1))
    var = np.clip(np.mean(res ** 2, axis=1), 1e-9, None)
    if method == "wls_var":
        return np.diag(var)
    W = shrink_cov(res)
    return W + np.eye(n) * 1e-9 * var.mean()


def reconcile(method: str, S: np.ndarray, yhat: np.ndarray, res: np.ndarray | None = None,
              shares: np.ndarray | None = None) -> np.ndarray:
    """Coherent forecasts [n_series, h] from base forecasts `yhat` [n_series, h]."""
    m = S.shape[1]
    if method == "base":
        return yhat
    if method == "bottom_up":
        return S @ yhat[-m:]
    if method == "top_down":
        return S @ (shares[:, None] * yhat[:1])
    W = weights(method, S, res)
    Wi_S = np.linalg.solve(W, S)                  # W^-1 S
    G = np.linalg.solve(S.T @ Wi_S, Wi_S.T)       # (S' W^-1 S)^-1 S' W^-1
    return S @ (G @ yhat)


def average_shares(cells_train: np.ndarray) -> np.ndarray:
    """Each cell's share of the national total, averaged over the training years."""
    tot = cells_train.sum(axis=0)
    return (cells_train / tot).mean(axis=1)


# ---------- evaluation ----------
def backtest(cells: pd.DataFrame, H: Hierarchy | None = None, base_fn=ets_base, origin0: int | None = None,
             max_h: int = MAX_H, methods=METHODS) -> pd.DataFrame:
    """Errors of every method on every series, origin and horizon (long table)."""
    H = H or structure(cells.index)
    years = np.array(cells.columns)
    X = cells.to_numpy()
    Y = H.S @ X
    rows = []
    for origin in range(first_origin(years) if origin0 is None else origin0, years[-1]):
        tr = years <= origin
        h = min(max_h, int(years[-1] - origin))
        yhat, res = base_fn(Y[:, tr], h, origin)
        if yhat is None:
            continue
        shares = average_shares(X[:, tr])
        actual = Y[:, (years > origin) & (years <= origin + h)]
        for method in methods:
            if method in ("wls_var", "mint_shrink") and (res is None or res.shape[1] < 3):
                continue
            f = reconcile(method, H.S, yhat, res, shares)
            for k in range(h):
                rows.append(pd.DataFrame({"method": method, "origin": origin, "h": k + 1, "series": H.ids,
                                          "level": H.level, "actual": actual[:, k], "forecast": f[:, k]}))
    return pd.concat(rows, ignore_index=True)


def wape(errors: pd.DataFrame, by=("method", "level")) -> pd.DataFrame:
    """Weighted absolute percentage error: sum |forecast - actual| / sum actual, in %."""
    e = errors.assign(abs_err=(errors["forecast"] - errors["actual"]).abs())
    g = e.groupby(list(by))[["abs_err", "actual"]].sum()
    return (100 * g["abs_err"] / g["actual"]).rename("WAPE_%")


def summary(errors: pd.DataFrame) -> pd.DataFrame:
    """WAPE by method (rows) and level (columns), plus the average over levels."""
    t = wape(errors).unstack("level").reindex(columns=list(LEVELS))
    t["average"] = t.mean(axis=1)
    return t.reindex([m for m in METHODS if m in t.index])


def coherence_gap(errors: pd.DataFrame, H: Hierarchy) -> dict:
    """How far the separate (base) forecasts are from adding up, as % of the national forecast."""
    b = errors[errors["method"] == "base"]
    out = {}
    for lv in ("sector", "ministry", "province", "cell"):
        parts = b[b["level"] == lv].groupby(["origin", "h"])["forecast"].sum()
        nat = b[b["level"] == "national"].set_index(["origin", "h"])["forecast"]
        out[lv] = float((100 * (parts - nat).abs() / nat).mean())
    return out


def best(tables: dict) -> tuple[str, str]:
    """(base forecaster, reconciliation method) with the lowest average WAPE over the levels."""
    pairs = [(b, m) for b, t in tables.items() for m in COHERENT if m in t.index]
    return min(pairs, key=lambda bm: tables[bm[0]].loc[bm[1], "average"])


def forecast(cells: pd.DataFrame, method: str, H: Hierarchy | None = None, h: int = FUTURE_H,
             base_fn=ets_base) -> pd.DataFrame:
    """Reconciled forecasts for the years after the history, with the history (long table)."""
    H = H or structure(cells.index)
    years = np.array(cells.columns)
    X = cells.to_numpy()
    Y = H.S @ X
    yhat, res = base_fn(Y, h, int(years[-1]))
    f = reconcile(method, H.S, yhat, res, average_shares(X))
    fut = years[-1] + np.arange(1, h + 1)
    meta = pd.DataFrame({"level": H.level, "label": H.label}, index=H.ids)
    out = []
    for kind, v, cols in (("history", Y, years), ("base", yhat, fut), ("forecast", f, fut)):
        d = pd.DataFrame(v, index=H.ids, columns=cols).rename_axis("series").reset_index()
        out.append(d.melt(id_vars="series", var_name="year", value_name="wage_bill_bn").assign(kind=kind))
    return pd.concat(out, ignore_index=True).join(meta, on="series")


def build(history: pd.DataFrame, fm_base: pd.DataFrame | None = None, ets_fn=ets_base) -> dict:
    """Everything the report and dashboard show: backtest tables, the chosen approach, forecasts.

    `fm_base`: optional foundation-model base forecasts for the same series (columns model,
    series, origin, h, value; see `foundation.py`). They are reconciled and scored exactly like
    the exponential-smoothing ones, and the best base forecaster and method make the forecast.
    `ets_fn`: how the smoothing is fitted (`ets_grid_base` for the quick rerun in the dashboard).
    """
    cells = cell_series(history)
    H = structure(cells.index)
    base_fns, errors = {"ets": ets_fn}, {"ets": backtest(cells, H, base_fn=ets_fn)}
    if fm_base is not None and len(fm_base):
        for model, d in fm_base.groupby("model"):
            base_fns[model] = fm_base_fn(d, H)
            errors[model] = backtest(cells, H, base_fn=base_fns[model])
    tables = {b: summary(e) for b, e in errors.items()}
    base_model, method = best(tables)
    e = errors[base_model]
    by_h = wape(e[e["level"].isin(["national", "province", "ministry"])], by=("method", "level", "h")).unstack(["level", "h"])
    return {"cells": cells, "tables": tables, "base_model": base_model, "method": method, "by_h": by_h,
            "gap": coherence_gap(errors["ets"], H), "forecast": forecast(cells, method, H, base_fn=base_fns[base_model]),
            "origins": sorted(e["origin"].unique().tolist()), "max_h": MAX_H, "wins": wins(e, method)}


def wins(errors: pd.DataFrame, method: str, against=("base", "bottom_up", "top_down")) -> dict:
    """Share of (origin, horizon, series) forecasts where `method` is closer to the actual than each alternative."""
    e = errors.assign(abs_err=(errors["forecast"] - errors["actual"]).abs())
    p = e.pivot_table(index=["origin", "h", "series"], columns="method", values="abs_err")
    return {a: float((p[method] < p[a]).mean()) for a in against if a in p}


def fm_base_fn(d: pd.DataFrame, H: Hierarchy):
    """Base forecasts and residuals from saved foundation-model forecasts, for `backtest`.

    Residuals for origin T are the model's one-year-ahead errors for the years up to T,
    each forecast from the years before it (the same role ETS's in-sample errors play).
    """
    piv = d.pivot_table(index=["origin", "h"], columns="series", values="value").reindex(columns=H.ids)

    def fn(Y_train, h, origin):
        if (origin, 1) not in piv.index:
            return None, None
        yhat = np.stack([piv.loc[(origin, k)].to_numpy() for k in range(1, h + 1)], axis=1)
        years = np.arange(origin - Y_train.shape[1] + 1, origin + 1)
        res = [Y_train[:, i] - piv.loc[(y - 1, 1)].to_numpy() for i, y in enumerate(years) if (y - 1, 1) in piv.index]
        return yhat, (np.array(res).T if res else None)
    return fn
