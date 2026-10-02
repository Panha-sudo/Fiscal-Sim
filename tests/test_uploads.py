"""Uploading your own staff history and macro series, and rerunning the models that use them."""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from fiscalsim import config as C
from fiscalsim import m1_data_quality as m1
from fiscalsim import m2_workforce as m2
from fiscalsim import m5_macro as m5
from fiscalsim import refit, synthetic, upload
from fiscalsim import simulate as S


@pytest.fixture(scope="module")
def history():
    return upload.example_history(n0=1500, seed=4)


@pytest.fixture(scope="module")
def macro():
    return synthetic.macro_history(np.random.default_rng(2))


def test_history_upload_checks(history):
    h, errors, notes = upload.prepare_history(history)
    assert not errors and len(h) == len(history)
    assert h["exit"].sum() == (history["exit_type"] != "").sum()
    # without exit and promotion columns they are worked out from who appears the next year
    h2, errors, notes = upload.prepare_history(history.drop(columns=["exit_type", "promoted", "province"]))
    assert not errors and any("exit_type" in n for n in notes) and any("province" in n for n in notes)
    before_last = h2["year"] < h2["year"].max()
    truth = h.set_index(["person_id", "year"])
    got = h2.set_index(["person_id", "year"])
    assert (got.loc[before_last.to_numpy(), "exit"] == truth.loc[before_last.to_numpy(), "exit"]).all()
    assert (got["promoted"] == truth["promoted"]).mean() > 0.99
    assert (h2["province"] == "Unknown").all()


@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns=["person_id"]), "Missing required columns"),
    (lambda d: pd.concat([d, d.head(3)]), "repeat a person_id"),
    (lambda d: d[d["year"] >= d["year"].max() - 3], "at least"),
    (lambda d: d[d["year"] != d["year"].min() + 2], "missing between"),
    (lambda d: d.assign(exit_type=d["exit_type"].replace({"death": "died"})), "Unknown exit_type"),
    (lambda d: d.assign(sector="army"), "sector"),
])
def test_history_upload_rejects(history, change, message):
    h, errors, _ = upload.prepare_history(change(history))
    assert h is None and errors and message in " ".join(errors)


def test_macro_upload_checks(macro):
    m, errors, notes = upload.prepare_macro(macro)
    assert not errors and list(m["year"]) == list(macro["year"])
    pct = macro.assign(**{c: macro[c] * 100 for c in m5.TARGETS})
    m2_, errors, notes = upload.prepare_macro(pct)
    assert not errors and sum("percentages" in n for n in notes) == 3
    np.testing.assert_allclose(m2_[list(m5.TARGETS)], m[list(m5.TARGETS)])
    for bad, message in [(macro.head(10), "at least"), (macro.drop(index=5), "consecutive"),
                         (macro.assign(revenue_share_gdp=0.9), "revenue_share_gdp"),
                         (macro.drop(columns=["inflation"]), "Missing required columns")]:
        out, errors, _ = upload.prepare_macro(bad)
        assert out is None and message in " ".join(errors)


def test_example_files_pass_their_own_checks(history, macro):
    assert not upload.prepare_history(upload.example_history())[1]
    rng = np.random.default_rng(5)
    _, stock = synthetic.simulate_history(rng, n0=800)
    hrmis, _ = synthetic.hrmis_extract(rng, stock)
    sample = upload.read_table("hrmis.csv", upload.template_csv(hrmis))
    assert not upload.prepare_hrmis(sample)[1]
    assert not upload.prepare_pensioners(synthetic.pensioner_register(rng).head(50))[1]
    assert not upload.prepare_macro(upload.read_table("macro.csv", macro.to_csv(index=False).encode()))[1]
    for req, opt in [(upload.HISTORY_REQUIRED, upload.HISTORY_OPTIONAL), (upload.MACRO_REQUIRED, upload.MACRO_OPTIONAL)]:
        assert len(upload.column_table(req, opt)) == len(req) + len(opt)


def test_history_refit(history):
    h, _, _ = upload.prepare_history(history)
    steps = []
    r = refit.history_models(h, steps.append, max_rows=20_000)
    assert steps == ["exit_model", "exit_test", "wave_test", "wage_bill"]
    assert r["rows_used"] <= 20_000
    assert r["wave_backtest"]["origin"] == m2.backtest_origin(h) == h["year"].max() - m2.TEST_YEARS
    assert len(r["m2"]) > 0 and len(r["shap_exit"]) > 0
    assert r["hierarchy"]["forecast"]["year"].max() > h["year"].max()
    # the engine runs on the refitted exit model
    rng = np.random.default_rng(6)
    _, stock = synthetic.simulate_history(rng, n0=800)
    hrmis, _ = synthetic.hrmis_extract(rng, stock)
    central = m5.central_path(synthetic.macro_history(rng), S.YEARS, {t: "mean_8y" for t in m5.TARGETS})
    ctx = SimpleNamespace(workforce=r["workforce"], stock0=m2.base_stock(m1.run(hrmis)["clean"]),
                          data={"pensioners": synthetic.pensioner_register(rng)},
                          paths=m5.simulate_paths(central, 20, C.BASE.gdp))
    res = S.run_scenario(ctx, C.SCENARIOS["S0"])
    assert np.isfinite(S.results_frames({"S0": res})["table"].select_dtypes("number").to_numpy()).all()


def test_macro_refit(macro, monkeypatch):
    # as on Streamlit Community Cloud, without Prophet and LSTM
    for name in ("prophet", "lstm"):
        monkeypatch.setitem(m5.MODELS, name, lambda y, h: None)
    later = macro.assign(year=macro["year"] - 3)  # a series ending three years earlier
    m, _, notes = upload.prepare_macro(later)
    assert any("end in" in n for n in notes)
    steps = []
    r = refit.macro_models(m, steps.append, runs=40)
    assert steps == ["macro_test", "macro_benchmark"]
    assert set(r["m5"]["model"]) <= {"naive", "mean_8y", "linear_trend", "arima"}
    assert set(r["m5_choice"]) == set(m5.TARGETS)
    assert r["paths"]["cpi"].shape[0] == 40
    fm = r["foundation"]
    assert fm["train_end"] == m5.train_end(m) == m["year"].max() - m5.TEST_YEARS
    assert fm["origins"][-1] == m["year"].max() - 5


def test_quick_wage_bill_smoothing_is_close_to_the_full_fit(history):
    from fiscalsim import hierarchy as HI
    cells = HI.cell_series(upload.prepare_history(history)[0])
    Y = HI.structure(cells.index).S @ cells.to_numpy()
    fast, res = HI.ets_grid_base(Y[:, :-3], 3)
    full, _ = HI.ets_base(Y[:, :-3], 3)
    assert fast.shape == full.shape and res.shape == Y[:, :-3].shape and (fast >= 0).all()
    actual = Y[:, -3:]
    wape = lambda f: np.abs(f - actual).sum() / actual.sum()
    assert wape(fast) < 1.5 * wape(full) + 0.01
    flat = np.tile([5.0], (2, 8))
    assert np.allclose(HI.ets_grid_base(flat, 2)[0], 5.0)


def test_payroll_without_truth_file():
    rng = np.random.default_rng(7)
    _, stock = synthetic.simulate_history(rng, n0=800)
    hrmis, _ = synthetic.hrmis_extract(rng, stock)
    r = m1.run(hrmis)
    assert "evaluation" not in r and r["records"]["flag"].any()
    assert len(m2.base_stock(r["clean"])) > 0
