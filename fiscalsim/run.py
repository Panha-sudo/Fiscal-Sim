"""Run the full pipeline: synthetic data -> M1..M5 -> scenarios S0-S6 -> M6 -> outputs.

    python -m fiscalsim.run --runs 10000 --out outputs
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import m6_explain as m6
from . import breakdown, report
from . import optimize as O
from . import simulate as S


def central_only(paths: dict) -> dict:
    return {k: v[:1] for k, v in paths.items()}


def breakeven_accrual(ctx: S.Context, target: float, lo=0.015, hi=0.05, tol=0.0005) -> float | None:
    """Accrual rate at which retirement-at-60 + accrual formula matches `target` replacement (central path).

    None when no rate up to `hi` reaches it (the replacement cap binds first).
    """
    p0 = central_only(ctx.paths)
    base = C.SCENARIOS["S0"].with_(code="S3S4", retirement_age_target=60.0, retirement_phase_years=10,
                                   pension_formula="accrual")

    def rr(rate):
        res = S.run_scenario(ctx, base.with_(accrual_rate=rate), p0)
        return S.summarize(res)["avg_replacement_2047_2076"]

    if rr(hi) < target:
        return None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if rr(mid) < target else (lo, mid)
    return (lo + hi) / 2


def engine_bundle(ctx: S.Context, runs: int = 1000) -> dict:
    """What the dashboard needs to run custom scenarios live, without the training data."""
    return {"workforce": ctx.workforce, "stock0": ctx.stock0, "pensioners": ctx.data["pensioners"],
            "paths": {k: v[:runs] for k, v in ctx.paths.items()}, "central": ctx.central}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=C.MC.runs)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--regenerate", action="store_true", help="rebuild synthetic data")
    ap.add_argument("--grid-check", action="store_true",
                    help="also score every reform package to check the optimiser against brute force (slow)")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    out, charts = args.out, args.out / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    ctx = S.build_context(args.data, runs=args.runs, regenerate=args.regenerate, log=log)
    log(f"M2-M4 scenarios S0-S6 x {args.runs:,} runs")
    results, table = S.run_all(ctx)

    log("M6 SHAP")
    imp_exit, _, _ = m6.exit_model_shap(ctx.workforce, ctx.data["history"])
    imp_unc, r2_unc, _, _ = m6.uncertainty_shap(results["S0"], ctx.paths)
    imp_pol, r2_pol, _, _ = m6.policy_shap(results, ctx.paths)
    log("Break-even accrual rate for H3")
    be = breakeven_accrual(ctx, table.loc["S0", "avg_replacement_2047_2076"])
    shap_out = {"exit": imp_exit, "uncertainty": imp_unc, "uncertainty_r2": r2_unc,
                "policy": imp_pol, "policy_r2": r2_pol, "breakeven_accrual": be}

    log("AI reform optimiser (pension levers, then pension + pay levers)")
    sub = O.paths_subset(ctx, 200)
    opt = {"pension": O.search(sub, pay_levers=False), "pension_pay": O.search(sub, pay_levers=True)}
    if args.grid_check:
        for k, r in opt.items():
            log(f"Brute-force grid check: {k} ({r['grid_size']} packages)")
            allrows = O.full_grid(sub, r["pay_levers"])
            allrows.to_csv(out / f"optimiser_grid_{k}.csv", index=False)
            r["grid_check"] = O.compare_to_grid(r["front"], allrows)

    log("Writing outputs")
    for k, r in opt.items():
        r["front"].to_csv(out / f"optimiser_front_{k}.csv", index=False)
        r["scored"].to_csv(out / f"optimiser_scored_{k}.csv", index=False)
    ctx.m1["evaluation"].to_csv(out / "m1_evaluation.csv")
    ctx.m1["recall_by_type"].to_csv(out / "m1_recall_by_type.csv")
    flagged = ctx.m1["records"]
    flagged[flagged["flag"]].to_csv(out / "m1_flagged_records.csv", index=False)
    ctx.m2_backtest["table"].to_csv(out / "m2_backtest.csv")
    ctx.m2_backtest["by_year_sector"].to_csv(out / "m2_backtest_by_year_sector.csv")
    ctx.km.to_csv(out / "m2_kaplan_meier.csv", index=False)
    ctx.m5_backtest.to_csv(out / "m5_backtest.csv", index=False)
    ctx.central.rename_axis("year").to_csv(out / "m5_central_path.csv")
    table.to_csv(out / "scenario_summary.csv")
    frames = S.results_frames(results)
    fans, wf, ad = frames["fans"], frames["workforce"], frames["adequacy"]
    fans.to_csv(out / "scenario_fans.csv", index=False)
    wf.to_csv(out / "workforce_projection.csv")
    ad.to_csv(out / "adequacy.csv")
    frames["wage_bill_sector"].to_csv(out / "wage_bill_by_sector.csv")
    base = breakdown.base_table(ctx.m1["records"])
    breakdown.by(base, "ministry").to_csv(out / "base_year_by_ministry.csv")
    breakdown.by(base, "province").to_csv(out / "base_year_by_province.csv")
    for name in ("exit", "uncertainty", "policy"):
        shap_out[name].to_csv(out / f"shap_{name}.csv")

    for m in ("total_cost_gdp", "wage_bill_gdp", "pension_spending_gdp", "fund_balance_gdp"):
        report.chart_scenarios(results, m, charts / f"{m}_scenarios.png")
    report.chart_fans(results, "total_cost_gdp", charts / "total_cost_gdp_fans.png")
    report.chart_bar(imp_exit.head(10), "M2 exit model: mean |SHAP| (log-odds)", charts / "shap_exit_model.png")
    report.chart_bar(imp_pol, "Total cost 2046: mean |SHAP| (pp of GDP)", charts / "shap_policy.png")
    report.markdown(ctx, results, table, shap_out, out, args.runs, opt)

    with open(out / "dashboard_bundle.pkl", "wb") as f:
        pickle.dump({"table": table, "fans": fans, "workforce": wf, "adequacy": ad, "shap": shap_out,
                     "wage_bill_sector": frames["wage_bill_sector"], "base_breakdown": base,
                     "hrmis_sample": ctx.data["hrmis"].head(200),
                     "m1": ctx.m1["evaluation"], "m1_types": ctx.m1["recall_by_type"],
                     "m2": ctx.m2_backtest["table"], "km": ctx.km, "m5": ctx.m5_backtest,
                     "m5_choice": ctx.m5_choice, "macro": ctx.data["macro"], "central": ctx.central,
                     "engine": engine_bundle(ctx), "runs": args.runs, "optimiser": opt}, f)
    log(f"Done. Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
