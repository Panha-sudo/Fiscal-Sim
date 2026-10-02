"""Hierarchical wage bill forecasting and the foundation-model benchmark."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fiscalsim import foundation as FD
from fiscalsim import hierarchy as HI
from fiscalsim import m5_macro as m5
from fiscalsim import synthetic

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cells():
    rng = np.random.default_rng(3)
    history, stock = synthetic.simulate_history(rng, n0=4000)
    history, _ = synthetic.assign_units(history, stock)
    c = HI.cell_series(history)
    return c[c.index.get_level_values("province").isin(["Phnom Penh", "Kandal", "Kep"])]


def test_reconciled_forecasts_add_up(cells):
    H = HI.structure(cells.index)
    n_cells = len(cells)
    assert H.S.shape == (1 + 3 + cells.index.get_level_values("ministry").nunique() + 3 + n_cells, n_cells)
    rng = np.random.default_rng(0)
    Y = H.S @ cells.to_numpy()
    yhat = Y[:, -3:] * rng.normal(1, 0.05, (len(Y), 3))  # incoherent base forecasts
    res = Y[:, 1:] - Y[:, :-1] - (Y[:, 1:] - Y[:, :-1]).mean(axis=1, keepdims=True)
    shares = HI.average_shares(cells.to_numpy())
    assert shares.sum() == pytest.approx(1)
    for method in HI.COHERENT:
        f = HI.reconcile(method, H.S, yhat, res, shares)
        np.testing.assert_allclose(f, H.S @ f[-n_cells:], rtol=1e-9, atol=1e-9)  # every total is the sum of its cells
    coherent = H.S @ yhat[-n_cells:]
    for method in ("ols", "wls_struct", "wls_var", "mint_shrink"):  # already coherent forecasts are left alone
        np.testing.assert_allclose(HI.reconcile(method, H.S, coherent, res), coherent, rtol=1e-6, atol=1e-9)


def test_hierarchy_backtest_and_forecast(cells):
    H = HI.structure(cells.index)
    e = HI.backtest(cells, H)
    table = HI.summary(e)
    assert set(table.index) == set(HI.METHODS) and list(table.columns) == [*HI.LEVELS, "average"]
    assert (table.to_numpy() >= 0).all()
    assert HI.coherence_gap(e, H)["province"] > 0  # separate forecasts do not add up
    method = HI.best(({"ets": table}))[1]
    f = HI.forecast(cells, method, H)
    fc = f[f.kind == "forecast"]
    prov = fc[fc.level == "province"].groupby("year")["wage_bill_bn"].sum()
    nat = fc[fc.level == "national"].set_index("year")["wage_bill_bn"]
    np.testing.assert_allclose(prov.to_numpy(), nat.reindex(prov.index).to_numpy(), rtol=1e-9)
    assert sorted(fc.year.unique()) == list(range(int(cells.columns[-1]) + 1, int(cells.columns[-1]) + 1 + HI.FUTURE_H))


def test_foundation_pipeline_with_a_stand_in_model(tmp_path, monkeypatch):
    """The run/score path, with every pretrained model replaced by a last-value forecaster."""
    macro = pd.read_csv(ROOT / "benchmarks" / FD.MACRO_FILE)
    wage = pd.read_csv(ROOT / "benchmarks" / FD.WAGE_FILE, index_col=0).head(5)
    FD.export_inputs(macro, wage, tmp_path)

    def last_value(repo):
        return lambda ctx, h: np.array([[[c[-1] - 0.01, c[-1], c[-1] + 0.01]] * h for c in ctx])
    monkeypatch.setattr(FD, "LOADERS", {k: last_value for k in FD.FOUNDATION})
    info = FD.run(tmp_path, tmp_path)
    assert all(m["status"] == "ok" for m in info["models"].values())
    fm, _ = FD.load_saved(tmp_path)
    n_macro = sum(t["h"] for t in FD.macro_tasks(macro))
    assert (fm[fm.task == "macro"].groupby("model").size() == n_macro).all()
    base = FD.baseline_forecasts(macro, models=("naive", "mean_8y", "arima"))
    r = FD.score_macro(macro, fm, base)
    roll = r["rolling"].set_index(["target", "model"])
    for target in m5.TARGETS:  # a last-value forecaster scores exactly like naive
        assert roll.loc[(target, "chronos_2"), "MAE_pp"] == pytest.approx(roll.loc[(target, "naive"), "MAE_pp"])
    assert r["overall"].loc["naive", "rel_MAE_vs_naive"] == pytest.approx(1)
    assert 0 <= r["overall"].loc["arima", "coverage_80_pct"] <= 100
    from fiscalsim import report  # the report also works without the LSTM and Prophet baselines
    assert any("Diebold-Mariano" in line for line in report.foundation_markdown(r))
    same = r["same_as_m5"].set_index(["target", "model"])["MAPE_%"]
    ref = m5.backtest(macro).set_index(["target", "model"])["MAPE_%"]
    for key in [k for k in same.index if k in ref.index]:  # same test, same numbers as M5's own backtest
        assert same[key] == pytest.approx(ref[key])


def test_saved_forecasts_cover_every_task_and_bundle_uses_them():
    fm, info = FD.load_saved()
    assert fm is not None and {m for m, v in info["models"].items() if v["status"] == "ok"} == set(fm.model)
    macro = pd.read_csv(ROOT / "benchmarks" / FD.MACRO_FILE)
    wage = pd.read_csv(ROOT / "benchmarks" / FD.WAGE_FILE, index_col=0)
    n = sum(t["h"] for t in FD.macro_tasks(macro)) + len(wage) * len(FD.WAGE_ORIGINS) * FD.WAGE_H
    assert (fm.groupby("model").size() == n).all()
    assert (fm.q10 <= fm.q50 + 1e-9).all() and (fm.q50 <= fm.q90 + 1e-9).all()
    with open(ROOT / "outputs" / "dashboard_bundle.pkl", "rb") as f:
        B = pickle.load(f)
    assert set(fm.model) <= set(B["foundation"]["models"])
    assert set(B["hierarchy"]["tables"]) == {"ets", *fm.model}
    np.testing.assert_allclose(B["macro"][list(m5.TARGETS)].to_numpy(), macro[list(m5.TARGETS)].to_numpy(), atol=1e-6)
