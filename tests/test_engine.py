"""Fast checks on a small synthetic workforce (run: pytest -q)."""
import numpy as np
import pandas as pd
import pytest

from fiscalsim import config as C
from fiscalsim import m1_data_quality as m1
from fiscalsim import m2_workforce as m2
from fiscalsim import m3_salary as m3
from fiscalsim import m4_pension as m4
from fiscalsim import m5_macro as m5
from fiscalsim import simulate as S
from fiscalsim import synthetic
from types import SimpleNamespace


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(1)
    history, stock = synthetic.simulate_history(rng, n0=6000)
    hrmis, truth = synthetic.hrmis_extract(rng, stock)
    macro = synthetic.macro_history(rng)
    wm = m2.fit_projection_model(history, sample_frac=1.0)
    clean = m1.run(hrmis)["clean"]
    central = m5.central_path(macro, S.YEARS, {t: "mean_8y" for t in m5.TARGETS})
    paths = m5.simulate_paths(central, 50, C.BASE.gdp)
    ctx = SimpleNamespace(workforce=wm, stock0=m2.base_stock(clean),
                          data={"pensioners": synthetic.pensioner_register(rng)}, paths=paths)
    return dict(history=history, hrmis=hrmis, truth=truth, ctx=ctx)


def test_m1_beats_rules(small):
    ev = m1.run(small["hrmis"], small["truth"])["evaluation"]
    assert ev.loc["matching_plus_iforest", "recall"] > ev.loc["rule_based_only", "recall"]


def test_survival_is_decreasing():
    s = m4.survival(55, np.arange(40), np.array([1.0, 1.2]))
    assert np.all(np.diff(s, axis=1) <= 0)
    assert np.all(s[1] <= s[0])  # heavier mortality -> fewer survivors


def test_inflation_rule_tracks_cpi():
    cpi = np.cumprod(np.full((3, 10), 1.03), axis=1)
    idx = m3.salary_index(C.SCENARIOS["S1"], cpi)
    assert np.allclose(idx[:, :, 0], cpi)


def test_projection_hits_headcount_target(small):
    sc = C.SCENARIOS["S0"]
    p = m2.project(small["ctx"].workforce, small["ctx"].stock0, sc, S.YEARS[:6])
    tot = p["stocks"].sum(axis=(1, 2, 3, 4, 5))
    assert np.allclose(tot[1:] / tot[:-1], 1 + sc.hiring_growth, rtol=1e-6)


def test_retirement_phase_reaches_target():
    ra = m2.retirement_age_path(C.SCENARIOS["S3"], S.YEARS)
    assert ra[0] == 55 and ra[-1] == 60 and np.all(np.diff(ra) >= 0)


def test_scenarios_run_and_s3_delays_depletion(small):
    r0 = S.run_scenario(small["ctx"], C.SCENARIOS["S0"])
    r3 = S.run_scenario(small["ctx"], C.SCENARIOS["S3"])
    for r in (r0, r3):
        for m in S.METRICS:
            assert np.isfinite(r["series"][m]).all()
    assert r3["series"]["pension_spending_gdp"][:, 10].mean() < r0["series"]["pension_spending_gdp"][:, 10].mean()


def test_accrual_lowers_replacement_vs_current(small):
    r0 = S.summarize(S.run_scenario(small["ctx"], C.SCENARIOS["S0"]))
    r4 = S.summarize(S.run_scenario(small["ctx"], C.SCENARIOS["S4"]))
    assert r4["avg_replacement_2047_2076"] < r0["avg_replacement_2047_2076"]


def test_upload_breakdown_export(small):
    from fiscalsim import breakdown, export, upload
    raw = small["hrmis"].drop(columns=["full_name", "bank_account", "record_id", "person_id"])
    recs, errors, notes = upload.prepare_hrmis(raw)
    assert not errors and any("full_name" in n for n in notes)
    r1 = m1.run(recs)
    base = breakdown.base_table(r1["records"])
    assert base["headcount"].sum() == (~r1["records"]["flag"]).sum()
    res = {k: S.run_scenario(small["ctx"], C.SCENARIOS[k]) for k in ("S0", "S3")}
    frames = S.results_frames(res)
    proj = breakdown.projected(base, frames["wage_bill_sector"].loc["S0"], "province", C.BASE_YEAR)
    assert np.isclose(proj.sum(), frames["wage_bill_sector"].loc["S0"].loc[C.BASE_YEAR].sum())
    xlsx = export.workbook(frames, C.SCENARIOS, 50, "test", breakdown.by(base, "ministry"))
    assert xlsx[:2] == b"PK"


def test_upload_rejects_bad_file():
    from fiscalsim import upload
    _, errors, _ = upload.prepare_hrmis(pd.DataFrame({"age": [30], "sector": ["army"], "framework": ["A"],
                                                       "service": [5], "basic_salary": [1e6]}))
    assert errors and "sector" in errors[0]
    _, errors, _ = upload.prepare_hrmis(pd.DataFrame({"age": [30]}))
    assert "Missing required columns" in errors[0]


def test_khmer_labels_cover_english():
    from fiscalsim import i18n
    assert set(i18n.EN) == set(i18n.KM)
    assert set(i18n.SCENARIO_NAMES["km"]) == set(C.SCENARIOS)
