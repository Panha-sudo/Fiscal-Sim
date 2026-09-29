"""Glue: build the shared context once, then run any scenario through M2 -> M3 -> M4.

All scenarios share the same Monte Carlo macro paths (common random numbers),
so differences between scenarios reflect policy, not sampling noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import m1_data_quality as m1
from . import m2_workforce as m2
from . import m3_salary as m3
from . import m4_pension as m4
from . import m5_macro as m5
from . import synthetic

YEARS = np.arange(C.BASE_YEAR, C.END_YEAR + 1)
METRICS = ("wage_bill_gdp", "wage_bill_revenue", "pension_spending_gdp", "gov_pension_cost_gdp",
           "total_cost_gdp", "fund_balance_gdp")


@dataclass
class Context:
    data: dict
    m1: dict
    m2_backtest: dict
    km: pd.DataFrame
    workforce: m2.WorkforceModel
    stock0: np.ndarray
    m5_backtest: pd.DataFrame
    m5_choice: dict
    central: pd.DataFrame
    paths: dict


def build_context(data_dir: Path, runs: int = C.MC.runs, regenerate: bool = False, log=print) -> Context:
    if regenerate or not (data_dir / "hrmis_2026.csv").exists():
        log("Generating synthetic inputs")
        synthetic.generate(data_dir)
    data = {
        "history": pd.read_parquet(data_dir / "staff_history.parquet"),
        "hrmis": pd.read_csv(data_dir / "hrmis_2026.csv"),
        "truth": pd.read_csv(data_dir / "hrmis_2026_truth.csv"),
        "pensioners": pd.read_csv(data_dir / "pensioners_2026.csv"),
        "macro": pd.read_csv(data_dir / "macro_history.csv"),
    }
    log("M1 data quality")
    r1 = m1.run(data["hrmis"], data["truth"])
    log("M2 workforce: backtest and fit")
    bt2 = m2.backtest(data["history"])
    km = m2.kaplan_meier(data["history"])
    wm = m2.fit_projection_model(data["history"])
    stock0 = m2.base_stock(r1["clean"])
    log("M5 macro: backtest and forecast")
    bt5 = m5.backtest(data["macro"])
    choice = m5.best_models(bt5)
    central = m5.central_path(data["macro"], YEARS, choice)
    paths = m5.simulate_paths(central, runs, C.BASE.gdp)
    return Context(data, r1, bt2, km, wm, stock0, bt5, choice, central, paths)


def run_scenario(ctx: Context, sc: C.Scenario, paths: dict | None = None) -> dict:
    paths = paths or ctx.paths
    proj = m2.project(ctx.workforce, ctx.stock0, sc, YEARS)
    units = m3.wage_units(proj["stocks"])
    idx = m3.salary_index(sc, paths["cpi"])
    wb = m3.wage_bill(units, idx)
    pen = m4.run(sc, proj, idx, paths, ctx.data["pensioners"], wb["basic_payroll"])
    gdp, rev = paths["gdp"], paths["revenue"]
    gov_pension = pen["contrib_employer"] + pen["gov_topup"]
    series = {
        "wage_bill": wb["wage_bill"],
        "wage_bill_gdp": wb["wage_bill"] / gdp * 100,
        "wage_bill_revenue": wb["wage_bill"] / rev * 100,
        "pension_spending_gdp": pen["pension_spending"] / gdp * 100,
        "gov_pension_cost_gdp": gov_pension / gdp * 100,
        "total_cost_gdp": (wb["wage_bill"] + gov_pension) / gdp * 100,
        "fund_balance_gdp": pen["fund_balance"] / gdp * 100,
    }
    sector_wb = np.median(units["total"][None, :, :] * idx, axis=0)  # [T, sector], median path
    return {"scenario": sc, "workforce": m2.headcount_table(proj), "series": series,
            "wage_bill_sector": pd.DataFrame(sector_wb, index=YEARS, columns=C.SECTORS),
            "depletion_year": pen["depletion_year"], "adequacy": pen["adequacy"],
            "new_pensioners": pen["new_pensioners"]}


def fan(series: np.ndarray, q=(5, 50, 95)) -> pd.DataFrame:
    """Percentile bands by year (90% band = 5th to 95th)."""
    p = np.percentile(series, q, axis=0)
    df = pd.DataFrame(p.T, index=YEARS, columns=[f"p{x}" for x in q])
    df["central"] = series[0]
    return df


def summarize(res: dict, horizons=C.HORIZONS) -> dict:
    sc = res["scenario"]
    row = {"scenario": sc.code, "name": sc.name}
    for m in METRICS:
        s = res["series"][m]
        for h in horizons:
            i = int(np.where(YEARS == h)[0][0])
            row[f"{m}_{h}_p50"] = float(np.median(s[:, i]))
            row[f"{m}_{h}_p5"] = float(np.percentile(s[:, i], 5))
            row[f"{m}_{h}_p95"] = float(np.percentile(s[:, i], 95))
            row[f"{m}_{h}_central"] = float(s[0, i])
    dep = res["depletion_year"]
    row["prob_fund_depleted_by_2076"] = float((dep > 0).mean())
    row["median_depletion_year"] = float(np.median(dep[dep > 0])) if (dep > 0).any() else np.nan
    row["central_depletion_year"] = float(dep[0]) if dep[0] else np.nan
    ad = res["adequacy"]
    for a, b in ((2027, 2036), (2037, 2046), (2047, 2076)):
        d = ad[(ad.year >= a) & (ad.year <= b)]
        w = d["retirees"]
        row[f"avg_replacement_{a}_{b}"] = float(np.average(d["avg_replacement"], weights=w)) if w.sum() else np.nan
        row[f"min_replacement_{a}_{b}"] = float(d["min_replacement"].min()) if len(d) else np.nan
        row[f"share_on_minimum_{a}_{b}"] = float(np.average(d["share_on_minimum"], weights=w)) if w.sum() else np.nan
    wf = res["workforce"]
    for h in horizons:
        row[f"headcount_{h}"] = float(wf.loc[h, "total"])
    return row


def rebuild_base(ctx, hrmis: pd.DataFrame, pensioners: pd.DataFrame | None = None) -> dict:
    """Swap in a new base-year payroll (and optionally pensioner register): M1 cleaning, then a new base stock."""
    r1 = m1.run(hrmis)
    ctx.stock0 = m2.base_stock(r1["clean"])
    if pensioners is not None:
        ctx.data["pensioners"] = pensioners
    return r1


def results_frames(results: dict) -> dict:
    """Tables the dashboard and the Excel export use, from a dict of scenario results."""
    from .report import fan_long
    table = pd.DataFrame([summarize(r) for r in results.values()]).set_index("scenario")
    wf = pd.concat({k: r["workforce"] for k, r in results.items()}, names=["scenario", "year"])
    ad = pd.concat({k: r["adequacy"] for k, r in results.items()}, names=["scenario", "i"])
    ad = ad.reset_index(level=1, drop=True)
    sector_wb = pd.concat({k: r["wage_bill_sector"] for k, r in results.items()}, names=["scenario", "year"])
    return {"table": table, "fans": fan_long(results), "workforce": wf, "adequacy": ad, "wage_bill_sector": sector_wb}


def run_all(ctx: Context, scenarios=None) -> tuple[dict, pd.DataFrame]:
    scenarios = scenarios or C.SCENARIOS
    results = {k: run_scenario(ctx, sc) for k, sc in scenarios.items()}
    table = pd.DataFrame([summarize(r) for r in results.values()]).set_index("scenario")
    return results, table
