"""Refit the parts of the model that depend on an uploaded staff history or macro series.

The published results are built offline with `python -m fiscalsim.run`. When someone uploads
their own files in the dashboard, these functions rebuild what depends on each file, in a
lighter form that fits in a Streamlit Community Cloud session (a minute or two):

  staff history  M2 exit model (XGBoost on at most MAX_ROWS person-years; the full build uses
                 half of all of them), its backtest, Kaplan-Meier curves, exit-model SHAP, the
                 retirement-wave backtest and the hierarchical wage bill forecast (its smoothing
                 fitted on a parameter grid, `hierarchy.ets_grid_base`, instead of by optimisation)
  macro series   M5 backtest and model choice (among the models installed: the app has no
                 Prophet or LSTM), the central path, RUNS Monte Carlo paths, and the forecast
                 benchmark for M5's models (the pretrained models only run offline)

What these files do not change: the AI pension fund risk model and the policy and uncertainty
SHAP charts, which are trained on the published runs.
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import foundation as FD
from . import hierarchy as HI
from . import m2_workforce as m2
from . import m5_macro as m5
from . import m6_explain as m6
from . import wave as W
from .simulate import YEARS

MAX_ROWS = 300_000
RUNS = 1000


def sample_frac(history: pd.DataFrame, max_rows: int = MAX_ROWS) -> float:
    """Share of person-years the exit model trains on: half, or fewer so at most `max_rows`."""
    return min(0.5, max_rows / max(len(m2.separation_sample(history)), 1))


def history_models(history: pd.DataFrame, log=lambda msg: None, max_rows: int = MAX_ROWS) -> dict:
    frac = sample_frac(history, max_rows)
    log("exit_model")
    wm = m2.fit_projection_model(history, sample_frac=frac)
    log("exit_test")
    bt2 = m2.backtest(history, sample_frac=frac)
    km = m2.kaplan_meier(history)
    shap_exit, _, _ = m6.exit_model_shap(wm, history)
    log("wave_test")
    wave_bt = W.backtest(history, sample_frac=frac)
    log("wage_bill")
    hier = HI.build(history, ets_fn=HI.ets_grid_base)
    return {"workforce": wm, "m2": bt2["table"], "km": km, "shap_exit": shap_exit, "wave_backtest": wave_bt,
            "hierarchy": hier, "rows_used": int(round(frac * len(m2.separation_sample(history))))}


def macro_models(macro: pd.DataFrame, log=lambda msg: None, runs: int = RUNS) -> dict:
    log("macro_test")
    bt5 = m5.backtest(macro)
    choice = m5.best_models(bt5)
    central = m5.central_path(macro, YEARS, choice)
    paths = m5.simulate_paths(central, runs, C.BASE.gdp)
    log("macro_benchmark")
    fm = FD.score_macro(macro, None)
    fm["info"] = {}
    return {"m5": bt5, "m5_choice": choice, "central": central, "paths": paths, "macro": macro, "foundation": fm}
