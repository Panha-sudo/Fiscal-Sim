"""M5 Macro-fiscal forecasting: real GDP growth, inflation and revenue share.

Candidates (proposal Table 4): naive, linear trend, ARIMA, Prophet, LSTM.
Backtest: train 1995-2017, forecast 2018-2025 in one multi-step run, scored on
the implied real-GDP index, price index and revenue share (MAPE, RMSE,
Diebold-Mariano against naive). The best model on the full series gives the
central path for the first years; it then converges to long-run anchors
(`config.MC`) because no annual model is credible 50 years out.

Monte Carlo paths add AR(1) shocks around the central path.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from . import config as C
from .m2_workforce import diebold_mariano

TARGETS = ("real_growth", "inflation", "revenue_share_gdp")
TRAIN_END = 2017


# ---------- forecasters: fit(y: pd.Series indexed by year) -> forecast(h) ----------
def f_naive(y, h):
    return np.repeat(y.iloc[-1], h)


def f_mean(y, h):
    return np.repeat(y.iloc[-8:].mean(), h)


def f_trend(y, h):
    x = np.arange(len(y))
    b, a = np.polyfit(x, y.to_numpy(), 1)
    return a + b * (len(y) + np.arange(h))


def f_arima(y, h):
    best, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in [(1, 0, 0), (2, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]:
            try:
                m = ARIMA(y.to_numpy(), order=order, trend="c" if order[1] == 0 else "n").fit()
                if m.aic < best_aic:
                    best, best_aic = m, m.aic
            except Exception:
                continue
    return best.forecast(h)


def f_prophet(y, h):
    try:
        from prophet import Prophet
    except ImportError:
        return None
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = pd.DataFrame({"ds": pd.to_datetime(y.index.astype(str) + "-01-01"), "y": y.to_numpy()})
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=0.05)
    m.fit(df)
    fut = m.make_future_dataframe(periods=h, freq="YS")
    return m.predict(fut)["yhat"].to_numpy()[-h:]


def f_lstm(y, h, window: int = 4, epochs: int = 400, seed: int = C.SEED):
    try:
        import torch
    except ImportError:
        return None
    torch.manual_seed(seed)
    v = y.to_numpy(float)
    mu, sd = v.mean(), v.std() + 1e-9
    z = (v - mu) / sd
    X = np.stack([z[i:i + window] for i in range(len(z) - window)])[..., None]
    Y = z[window:]
    X_t, Y_t = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, 8, batch_first=True)
            self.out = torch.nn.Linear(8, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.out(o[:, -1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(X_t) - Y_t) ** 2)
        loss.backward()
        opt.step()
    hist = list(z[-window:])
    out = []
    with torch.no_grad():
        for _ in range(h):
            p = float(net(torch.tensor(np.array(hist[-window:])[None, :, None], dtype=torch.float32)))
            out.append(p)
            hist.append(p)
    return np.array(out) * sd + mu


MODELS = {"naive": f_naive, "mean_8y": f_mean, "linear_trend": f_trend, "arima": f_arima,
          "prophet": f_prophet, "lstm": f_lstm}


def _to_level(target, rates):
    """Score growth and inflation on the implied index level; revenue share as is."""
    return np.cumprod(1 + np.asarray(rates)) if target != "revenue_share_gdp" else np.asarray(rates)


def backtest(macro: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        y = macro.set_index("year")[target]
        train, test = y[y.index <= TRAIN_END], y[y.index > TRAIN_END]
        actual = _to_level(target, test)
        preds = {}
        for name, fn in MODELS.items():
            p = fn(train, len(test))
            if p is not None:
                preds[name] = _to_level(target, p)
        for name, p in preds.items():
            e, e0 = p - actual, preds["naive"] - actual
            dm, pv = diebold_mariano(e0, e) if name != "naive" else (np.nan, np.nan)
            rows.append({"target": target, "model": name,
                         "MAPE_%": round(float(np.mean(np.abs(e / actual)) * 100), 3),
                         "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 5),
                         "DM_vs_naive": round(dm, 2) if dm == dm else np.nan,
                         "DM_p_value": round(pv, 4) if pv == pv else np.nan})
    return pd.DataFrame(rows)


def best_models(bt: pd.DataFrame) -> dict:
    return bt.loc[bt.groupby("target")["MAPE_%"].idxmin()].set_index("target")["model"].to_dict()


def central_path(macro: pd.DataFrame, years: np.ndarray, choice: dict, mc: C.MonteCarlo = C.MC) -> pd.DataFrame:
    """Best model forecast, blended linearly into long-run anchors over `convergence_years`."""
    anchors = {"real_growth": mc.long_run_real_growth, "inflation": mc.long_run_inflation,
               "revenue_share_gdp": mc.long_run_revenue_share}
    out = {}
    h = len(years)
    w = np.clip(np.arange(1, h + 1) / mc.convergence_years, 0, 1)
    for target in TARGETS:
        y = macro.set_index("year")[target]
        f = MODELS[choice[target]](y, h)
        out[target] = (1 - w) * f + w * anchors[target]
    return pd.DataFrame(out, index=years)


def _ar1(rng, runs, T, sd, rho=0.5):
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), (runs, T))
    x = np.zeros((runs, T))
    x[:, 0] = rng.normal(0, sd, runs)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def simulate_paths(central: pd.DataFrame, runs: int, base_gdp: float, seed: int = C.SEED,
                   mc: C.MonteCarlo = C.MC) -> dict:
    """Stochastic macro paths [runs, years]. Run 0 is the deterministic central path."""
    rng = np.random.default_rng(seed)
    T = len(central)
    g = central["real_growth"].to_numpy() + _ar1(rng, runs, T, mc.growth_sd)
    pi = central["inflation"].to_numpy() + _ar1(rng, runs, T, mc.inflation_sd, rho=0.6)
    rev = central["revenue_share_gdp"].to_numpy() + _ar1(rng, runs, T, mc.revenue_share_sd, rho=0.8)
    ret = C.PENSION.fund_return_real + _ar1(rng, runs, T, mc.return_sd, rho=0.3)
    mort = np.exp(rng.normal(0, mc.mortality_sd, runs) - mc.mortality_sd ** 2 / 2)
    pi = np.maximum(pi, -0.02)
    g[0], pi[0], rev[0], ret[0], mort[0] = (central["real_growth"], central["inflation"],
                                            central["revenue_share_gdp"], C.PENSION.fund_return_real, 1.0)
    cpi = np.cumprod(1 + pi, axis=1) / (1 + pi[:, :1])  # base year = 1
    gdp = base_gdp * np.cumprod((1 + g) * (1 + pi), axis=1) / ((1 + g[:, :1]) * (1 + pi[:, :1]))
    return {"real_growth": g, "inflation": pi, "cpi": cpi, "gdp": gdp, "revenue_share": rev,
            "revenue": gdp * rev, "real_return": ret, "mortality_mult": mort}
