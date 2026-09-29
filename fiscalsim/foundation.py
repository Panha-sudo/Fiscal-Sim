"""Time-series foundation models against M5's forecasters, and as base forecasts for the
hierarchical wage bill.

Chronos-Bolt, Chronos-2 (Amazon) and TimesFM 2.5 (Google) are pretrained on large
collections of time series and forecast zero-shot: they are not trained on Cambodian data,
they only read the history they are given. TimeGPT is left out because it needs a paid key.

The models need PyTorch and a download of their weights, which the dashboard does not have,
so they run offline and their forecasts are saved:

    python -m fiscalsim.foundation inputs     # write benchmarks/*_series.csv* from fresh synthetic data
    python -m fiscalsim.foundation run        # forecast (needs torch, chronos-forecasting, timesfm)

The GitHub Actions workflow `.github/workflows/foundation.yml` runs the second step and saves
benchmarks/foundation_forecasts.csv.gz to the `foundation-results` branch. Everything is
scored here, the same way for every model (`score_macro`, and `hierarchy.build` for the wage bill).

Two tests on the macro series (real growth, inflation, revenue share of GDP):
  same as M5   train 1995-2017, forecast 2018-2025 in one run, MAPE of the implied level
  rolling      forecast 1 to 5 years ahead from every year 2008-2020 (13 origins): error in
               percentage points; for models that give a range, 80% interval coverage and
               quantile loss
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import m5_macro as m5
from .m2_workforce import diebold_mariano

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
MACRO_FILE, WAGE_FILE = "macro_series.csv", "wage_bill_series.csv.gz"
FORECAST_FILE, RUN_FILE = "foundation_forecasts.csv.gz", "foundation_run.json"

FOUNDATION = {  # name -> (Hugging Face model, parameters)
    "chronos_bolt": ("amazon/chronos-bolt-base", "205M"),
    "chronos_2": ("amazon/chronos-2", "120M"),
    "timesfm_2_5": ("google/timesfm-2.5-200m-pytorch", "200M"),
}
NAMES = {"naive": "Naive (last value)", "mean_8y": "8-year mean", "linear_trend": "Linear trend", "arima": "ARIMA",
         "prophet": "Prophet", "lstm": "LSTM", "chronos_bolt": "Chronos-Bolt", "chronos_2": "Chronos-2",
         "timesfm_2_5": "TimesFM 2.5", "ets": "Exponential smoothing"}
QUANTILES = (0.1, 0.5, 0.9)
ROLL_ORIGINS, ROLL_H = range(2008, 2021), 5
FUTURE_H = 10
WAGE_ORIGINS, WAGE_H = range(2014, 2026), 5
BASELINES = ("naive", "mean_8y", "linear_trend", "arima", "prophet", "lstm")


# ---------- the pretrained models (only imported by `run`) ----------
def _chronos(repo):
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", dtype=torch.float32)

    def predict(contexts, h):
        out = []
        for i in range(0, len(contexts), 256):
            batch = [torch.tensor(np.asarray(c, float), dtype=torch.float32) for c in contexts[i:i + 256]]
            q, _ = pipe.predict_quantiles(batch, prediction_length=h, quantile_levels=list(QUANTILES))
            if isinstance(q, (list, tuple)):  # Chronos-2: one [variates, h, q] tensor per series
                q = torch.stack([x.reshape(-1, h, len(QUANTILES))[0] for x in q])
            out.append(q.float().cpu().numpy())
        return np.concatenate(out)
    return predict


def _timesfm(repo):
    import timesfm
    import torch

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, torch_compile=False)
    model.compile(timesfm.ForecastConfig(max_context=64, max_horizon=128, normalize_inputs=True, per_core_batch_size=64,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=False, fix_quantile_crossing=True))

    def predict(contexts, h):
        _, q = model.forecast(horizon=h, inputs=[np.asarray(c, float) for c in contexts])
        return np.asarray(q)[:, :h, [1, 5, 9]]  # column 0 is the mean, then deciles 0.1 ... 0.9
    return predict


LOADERS = {"chronos_bolt": _chronos, "chronos_2": _chronos, "timesfm_2_5": _timesfm}


# ---------- tasks: which histories to forecast ----------
def macro_tasks(macro: pd.DataFrame) -> list[dict]:
    """Every (series, origin, horizon) the macro tests need: rolling origins, M5's origin, the future."""
    y = macro.set_index("year")
    last = int(y.index.max())
    plan = {o: ROLL_H for o in ROLL_ORIGINS}
    for o, h in ((m5.TRAIN_END, last - m5.TRAIN_END), (last, FUTURE_H)):
        plan[o] = max(plan.get(o, 0), h)  # a longer run from the same origin also covers the shorter one
    tasks = []
    for target in m5.TARGETS:
        for origin, h in plan.items():
            tasks.append({"task": "macro", "series": target, "origin": origin, "h": h,
                          "context": y.loc[:origin, target].to_numpy(float)})
    return tasks


def wage_tasks(series: pd.DataFrame) -> list[dict]:
    """Wage bill series (rows = hierarchy ids, columns = years): one-step and multi-step forecasts from each year."""
    years = np.array(series.columns, int)
    tasks = []
    for origin in WAGE_ORIGINS:
        cols = years <= origin
        for sid, row in zip(series.index, series.to_numpy()[:, cols]):
            tasks.append({"task": "wage_bill", "series": sid, "origin": origin, "h": WAGE_H, "context": row})
    return tasks


def forecast_all(tasks: list[dict], predict) -> pd.DataFrame:
    rows = []
    for h in sorted({t["h"] for t in tasks}):
        group = [t for t in tasks if t["h"] == h]
        q = predict([t["context"] for t in group], h)  # [n, h, 3]
        for t, qi in zip(group, q):
            for k in range(h):
                rows.append((t["task"], t["series"], t["origin"], k + 1, t["origin"] + k + 1, *qi[k]))
    return pd.DataFrame(rows, columns=["task", "series", "origin", "h", "year", "q10", "q50", "q90"])


def run(inputs: Path = BENCH, out: Path = BENCH, models=tuple(FOUNDATION)) -> dict:
    """Forecast every task with every foundation model that loads; save forecasts and run notes."""
    macro = pd.read_csv(inputs / MACRO_FILE)
    wage = pd.read_csv(inputs / WAGE_FILE, index_col=0)
    wage.columns = wage.columns.astype(int)
    tasks = macro_tasks(macro) + wage_tasks(wage)
    frames, info = [], {"models": {}, "python": platform.python_version(), "date": time.strftime("%Y-%m-%d")}
    if os.environ.get("GITHUB_RUN_ID"):  # where the forecasts were made, when run by the workflow
        info["run"] = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        info["code"] = os.environ.get("GITHUB_SHA")
    for name in models:
        repo, size = FOUNDATION[name]
        t0 = time.time()
        try:
            predict = LOADERS[name](repo)
            loaded = time.time()
            f = forecast_all(tasks, predict).assign(model=name)
            frames.append(f)
            info["models"][name] = {"repo": repo, "parameters": size, "status": "ok", "rows": len(f),
                                    "load_seconds": round(loaded - t0, 1),
                                    "forecast_seconds": round(time.time() - loaded, 1)}
        except Exception as e:  # keep going with the other models, and say why this one failed
            traceback.print_exc()
            info["models"][name] = {"repo": repo, "parameters": size, "status": f"failed: {type(e).__name__}: {e}"[:500]}
        print(name, info["models"][name], flush=True)
    for pkg in ("torch", "chronos-forecasting", "timesfm", "transformers"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[pkg] = None
    out.mkdir(parents=True, exist_ok=True)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df[["q10", "q50", "q90"]] = df[["q10", "q50", "q90"]].round(7)
        df.to_csv(out / FORECAST_FILE, index=False)
    (out / RUN_FILE).write_text(json.dumps(info, indent=2))
    return info


def export_inputs(macro: pd.DataFrame, wage_series: pd.DataFrame, out: Path = BENCH) -> None:
    out.mkdir(parents=True, exist_ok=True)
    macro[["year", *m5.TARGETS]].to_csv(out / MACRO_FILE, index=False, float_format="%.7g")
    wage_series.round(4).to_csv(out / WAGE_FILE)


def synthetic_inputs(out: Path = BENCH) -> None:
    """Write the benchmark inputs from freshly generated synthetic data (never from data/, which may be real)."""
    from . import hierarchy as HI
    from . import synthetic
    with tempfile.TemporaryDirectory() as tmp:
        d = synthetic.generate(Path(tmp))
    export_inputs(d["macro"], HI.all_series(HI.cell_series(d["history"])), out)


def load_saved(bench: Path = BENCH) -> tuple[pd.DataFrame | None, dict]:
    f, r = bench / FORECAST_FILE, bench / RUN_FILE
    info = json.loads(r.read_text()) if r.exists() else {}
    return (pd.read_csv(f) if f.exists() else None), info


def inputs_match(macro: pd.DataFrame, wage_series: pd.DataFrame | None, bench: Path = BENCH) -> bool:
    """True when the saved forecasts were made from these same series (so they can be scored against them)."""
    try:
        saved = pd.read_csv(bench / MACRO_FILE)
        ok = np.allclose(saved[list(m5.TARGETS)].to_numpy(), macro[list(m5.TARGETS)].to_numpy(), atol=1e-6)
        if wage_series is not None:
            w = pd.read_csv(bench / WAGE_FILE, index_col=0)
            w.columns = w.columns.astype(int)
            ok = ok and w.shape == wage_series.shape and np.allclose(w.to_numpy(), wage_series.to_numpy(), atol=1e-3)
        return bool(ok)
    except (FileNotFoundError, ValueError, KeyError):
        return False


# ---------- baselines on the same tasks ----------
def arima_quantiles(y: pd.Series, h: int) -> np.ndarray:
    fit = m5.arima_fit(y)
    fc = fit.get_forecast(h)
    lo, hi = np.asarray(fc.conf_int(alpha=0.2)).T
    return np.stack([lo, np.asarray(fc.predicted_mean), hi], axis=1)


def baseline_forecasts(macro: pd.DataFrame, models=BASELINES) -> pd.DataFrame:
    """M5's own forecasters on the same macro tasks (ARIMA with its 80% interval)."""
    y_all = macro.set_index("year")
    rows = []
    for t in macro_tasks(macro):
        y = y_all.loc[:t["origin"], t["series"]]
        for name in models:
            if name == "arima":
                q = arima_quantiles(y, t["h"])
            else:
                p = m5.MODELS[name](y, t["h"])
                if p is None:
                    continue
                q = np.stack([np.full(t["h"], np.nan), np.asarray(p, float), np.full(t["h"], np.nan)], axis=1)
            for k in range(t["h"]):
                rows.append(("macro", t["series"], t["origin"], k + 1, t["origin"] + k + 1, *q[k], name))
    return pd.DataFrame(rows, columns=["task", "series", "origin", "h", "year", "q10", "q50", "q90", "model"])


# ---------- scoring ----------
def dm_hln(loss_a, loss_b, h: int = 1):
    """Diebold-Mariano on loss differences with the Harvey-Leybourne-Newbold small-sample
    correction and h-1 autocovariance lags. Positive stat: model a has the larger loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if n < 4 or np.allclose(d, 0):
        return np.nan, np.nan
    dc = d - d.mean()
    gam = [dc[k:] @ dc[:n - k] / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        var = gam[0] / n
    stat = d.mean() / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def pinball(q_pred, y, tau):
    e = y - q_pred
    return np.maximum(tau * e, (tau - 1) * e)


def score_macro(macro: pd.DataFrame, fm: pd.DataFrame | None, base: pd.DataFrame | None = None) -> dict:
    """Both macro tests for the baselines and the foundation models. `fm`: saved forecasts (task == macro)."""
    base = baseline_forecasts(macro) if base is None else base
    frames = [base] + ([fm[fm["task"] == "macro"]] if fm is not None else [])
    f = pd.concat(frames, ignore_index=True)
    actual = macro.set_index("year")[list(m5.TARGETS)].stack().rename("actual")
    actual.index.names = ["year", "series"]
    f = f.join(actual, on=["year", "series"])
    models = [m for m in [*BASELINES, *FOUNDATION] if m in set(f["model"])]

    # 1. same design as M5: origin 2017, the eight years 2018-2025 in one run, scored on the implied level
    same = []
    for target in m5.TARGETS:
        d = f[(f["series"] == target) & (f["origin"] == m5.TRAIN_END)].sort_values("h")
        preds = {m: m5._to_level(target, g.sort_values("h")["q50"]) for m, g in d.groupby("model")}
        act = m5._to_level(target, d[d["model"] == "naive"]["actual"])
        for m in models:
            if m not in preds:
                continue
            e, e0 = preds[m] - act, preds["naive"] - act
            dm, pv = diebold_mariano(e0, e) if m != "naive" else (np.nan, np.nan)
            same.append({"target": target, "model": m, "MAPE_%": round(float(np.mean(np.abs(e / act)) * 100), 3),
                         "RMSE": round(float(np.sqrt(np.mean(e ** 2))), 5),
                         "DM_vs_naive": round(dm, 2) if dm == dm else np.nan,
                         "DM_p_value": round(pv, 4) if pv == pv else np.nan})
    same = pd.DataFrame(same)

    # 2. rolling origins 2008-2020, 1-5 years ahead, error in percentage points of the annual rate
    r = f[f["origin"].isin(ROLL_ORIGINS) & (f["h"] <= ROLL_H)].copy()
    r["abs_err_pp"] = (r["q50"] - r["actual"]).abs() * 100
    r["covered"] = np.where(r["q10"].notna(), (r["actual"] >= r["q10"]) & (r["actual"] <= r["q90"]), np.nan)
    ql = sum(pinball(r[f"q{int(q * 100)}"], r["actual"], q) for q in QUANTILES) * 2 / len(QUANTILES) * 100
    r["qloss_pp"] = np.where(r["q10"].notna(), ql, np.nan)
    per_origin = r.groupby(["series", "model", "origin"])["abs_err_pp"].mean()
    rows = []
    for target in m5.TARGETS:
        naive_mae = r[(r["series"] == target) & (r["model"] == "naive")]["abs_err_pp"].mean()
        for m in models:
            d = r[(r["series"] == target) & (r["model"] == m)]
            if d.empty:
                continue
            row = {"target": target, "model": m, "MAE_pp": d["abs_err_pp"].mean(),
                   "rel_MAE_vs_naive": d["abs_err_pp"].mean() / naive_mae}
            row |= {f"MAE_h{h}": d[d["h"] == h]["abs_err_pp"].mean() for h in range(1, ROLL_H + 1)}
            row |= {"coverage_80_%": d["covered"].mean() * 100, "quantile_loss_pp": d["qloss_pp"].mean()}
            for ref in ("arima", "lstm"):
                if m in FOUNDATION and (target, ref, ROLL_ORIGINS[0]) in per_origin.index:
                    a, b = per_origin.loc[(target, m)], per_origin.loc[(target, ref)]
                    stat, pv = dm_hln(a.to_numpy(), b.reindex(a.index).to_numpy(), h=ROLL_H)
                    row[f"DM_vs_{ref}"], row[f"p_vs_{ref}"] = stat, pv
            rows.append(row)
    rolling = pd.DataFrame(rows)
    overall = (rolling.assign(log_rel=np.log(rolling["rel_MAE_vs_naive"]))
               .groupby("model").agg(rel_MAE_vs_naive=("log_rel", lambda v: float(np.exp(v.mean()))),
                                     coverage_80_pct=("coverage_80_%", "mean"),
                                     quantile_loss_pp=("quantile_loss_pp", "mean"))
               .reindex(models))
    overall["mean_rank"] = rolling.groupby("target")["MAE_pp"].rank().groupby(rolling["model"]).mean().reindex(models)
    future = f[f["origin"] == int(macro["year"].max())][["series", "model", "year", "q10", "q50", "q90"]]
    return {"same_as_m5": same, "rolling": rolling, "overall": overall, "future": future.reset_index(drop=True),
            "models": models, "origins": list(ROLL_ORIGINS), "h": ROLL_H}


def wage_fm_base(fm: pd.DataFrame | None) -> pd.DataFrame | None:
    """Saved wage-bill forecasts in the shape `hierarchy.build` takes (median as the point forecast)."""
    if fm is None:
        return None
    w = fm[fm["task"] == "wage_bill"]
    return w.rename(columns={"q50": "value"})[["model", "series", "origin", "h", "value"]] if len(w) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["inputs", "run"])
    ap.add_argument("--inputs", type=Path, default=BENCH)
    ap.add_argument("--out", type=Path, default=BENCH)
    ap.add_argument("--models", nargs="*", default=list(FOUNDATION), choices=list(FOUNDATION))
    a = ap.parse_args(argv)
    if a.step == "inputs":
        synthetic_inputs(a.out)
        print(f"Wrote {a.out / MACRO_FILE} and {a.out / WAGE_FILE}")
        return
    info = run(a.inputs, a.out, a.models)
    print(json.dumps(info, indent=2))
    if not any(m.get("status") == "ok" for m in info["models"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
