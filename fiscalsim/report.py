"""Static outputs: CSV tables, PNG charts and a Markdown report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .simulate import YEARS, fan

# Reference categorical palette, fixed order (one slot per scenario S0..S6)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {
    "wage_bill_gdp": "Wage bill, % of GDP",
    "wage_bill_revenue": "Wage bill, % of revenue",
    "pension_spending_gdp": "Pension spending, % of GDP",
    "gov_pension_cost_gdp": "Government pension cost, % of GDP",
    "total_cost_gdp": "Wage bill + government pension cost, % of GDP",
    "fund_balance_gdp": "Pension fund balance, % of GDP",
}


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def chart_scenarios(results: dict, metric: str, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for i, (k, r) in enumerate(results.items()):
        s = r["series"][metric]
        med = np.median(s, axis=0)
        ax.plot(YEARS, med, color=PALETTE[i], linewidth=2, label=f"{k} {r['scenario'].name}")
        ax.annotate(k, (YEARS[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    _style(ax, f"{LABELS[metric]} (median of Monte Carlo runs)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_fans(results: dict, metric: str, path: Path):
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows), sharey=True, facecolor=SURFACE)
    axes = axes.ravel()
    for i, (k, r) in enumerate(results.items()):
        f = fan(r["series"][metric])
        ax = axes[i]
        ax.fill_between(YEARS, f["p5"], f["p95"], color=PALETTE[i], alpha=0.18, linewidth=0)
        ax.plot(YEARS, f["p50"], color=PALETTE[i], linewidth=2)
        _style(ax, f"{k} {r['scenario'].name}")
        ax.title.set_fontsize(9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{LABELS[metric]}: median and 90% band", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_bar(series: pd.Series, title: str, path: Path, color: str = PALETTE[0]):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(s) + 1.2), facecolor=SURFACE)
    ax.barh(s.index, s.values, color=color, height=0.6)
    for y, v in enumerate(s.values):
        ax.annotate(f"{v:.3g}", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    _style(ax, title)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fan_long(results: dict) -> pd.DataFrame:
    rows = []
    for k, r in results.items():
        for m, s in r["series"].items():
            if m == "wage_bill":
                continue
            f = fan(s)
            f["scenario"], f["metric"] = k, m
            rows.append(f.rename_axis("year").reset_index())
    return pd.concat(rows, ignore_index=True)


def _fmt(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def markdown(ctx, results, table, shap_out, out: Path, runs: int, opt: dict | None = None,
             wave: dict | None = None, risk: dict | None = None) -> str:
    ev = ctx.m1["evaluation"]
    bt2 = ctx.m2_backtest["table"]
    bt5 = ctx.m5_backtest
    lines = [
        "# AI fiscal simulation prototype: results on synthetic data",
        "",
        f"Base year {C.BASE_YEAR}, horizon {C.END_YEAR}, {runs:,} Monte Carlo runs, seed {C.SEED}.",
        "",
        "> All inputs are synthetic, shaped like HRMIS, payroll, Budget Law, NSSF-C and macro data and",
        "> calibrated to placeholder totals in `fiscalsim/config.py`. Numbers illustrate how the model",
        "> works; they are not estimates for Cambodia until the placeholders are replaced with official data.",
        "",
        "## Scenario comparison",
        "",
        "Median of Monte Carlo runs, with the 90% band in brackets. Total cost = wage bill + government",
        "pension cost (employer contribution + top-up once the fund is exhausted).",
        "",
        "| Scenario | Total cost % GDP 2036 | 2046 | 2076 | Wage bill % revenue 2046 | Pension spending % GDP 2046 | Fund depleted by 2076 | Avg replacement 2037-46 | On minimum pension 2037-46 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, r in table.iterrows():
        def band(m, h):
            return f"{r[f'{m}_{h}_p50']:.2f} [{r[f'{m}_{h}_p5']:.2f}-{r[f'{m}_{h}_p95']:.2f}]"
        dep = f"{r['prob_fund_depleted_by_2076']:.0%}"
        if not pd.isna(r["median_depletion_year"]):
            dep += f" (median {int(r['median_depletion_year'])})"
        lines.append(f"| {k} {r['name']} | {band('total_cost_gdp', 2036)} | {band('total_cost_gdp', 2046)} | "
                     f"{band('total_cost_gdp', 2076)} | {r['wage_bill_revenue_2046_p50']:.1f} | "
                     f"{r['pension_spending_gdp_2046_p50']:.2f} | {dep} | "
                     f"{r['avg_replacement_2037_2046']:.0%} | {r['share_on_minimum_2037_2046']:.0%} |")
    s0 = table.loc["S0"]
    lines += [
        "",
        "Replacement rate = first pension / final total pay (basic + allowances), central path.",
        "",
        "![Total cost](charts/total_cost_gdp_scenarios.png)",
        "",
        "![Fans](charts/total_cost_gdp_fans.png)",
        "",
        "## Hypotheses (on synthetic data)",
        "",
    ]
    h1_ml = bt2.loc["xgboost", "separations_MAPE_%"]
    h1_base = bt2.loc["cohort_ratio", "separations_MAPE_%"]
    lines.append(f"- **H1** (ML beats cohort-ratio on exits): XGBoost separations MAPE {h1_ml:.2f}% vs "
                 f"cohort-ratio {h1_base:.2f}%; all-exits MAPE {bt2.loc['xgboost', 'all_exits_MAPE_%']:.2f}% vs "
                 f"{bt2.loc['cohort_ratio', 'all_exits_MAPE_%']:.2f}%; Diebold-Mariano p = "
                 f"{bt2.loc['xgboost', 'DM_p_value']:.3f}. "
                 + ("Supported." if h1_ml < h1_base and bt2.loc['xgboost', 'DM_p_value'] < 0.05 else
                    "Not supported at the 5% level on this data."))
    t26 = float(np.median(results["S0"]["series"]["total_cost_gdp"][:, 0]))
    t46 = s0["total_cost_gdp_2046_p50"]
    lines.append(f"- **H2** (status-quo cost share rises over 20 years): {t26:.2f}% of GDP in {C.BASE_YEAR} to "
                 f"{t46:.2f}% in 2046. " + ("Supported." if t46 > t26 else "Not supported."))
    s5 = table.loc["S5"]
    lower_cost = s5["pension_spending_gdp_2076_p50"] < s0["pension_spending_gdp_2076_p50"]
    keep_rr = s5["avg_replacement_2047_2076"] >= s0["avg_replacement_2047_2076"]
    lines.append(f"- **H3** (retirement age + accrual formula cuts long-run pension cost without lowering "
                 f"replacement): combined reform S5 pension spending {s5['pension_spending_gdp_2076_p50']:.2f}% vs "
                 f"{s0['pension_spending_gdp_2076_p50']:.2f}% of GDP in 2076; average replacement "
                 f"{s5['avg_replacement_2047_2076']:.0%} vs {s0['avg_replacement_2047_2076']:.0%}. "
                 + ("Supported." if lower_cost and keep_rr else
                    "Cost falls but replacement drops, so not supported at the default 2% accrual rate."
                    if lower_cost else "Not supported."))
    be = shap_out.get("breakeven_accrual")
    if be is None:
        lines.append(f"  No accrual rate restores the status-quo replacement rate for S3+S4: the "
                     f"{C.PENSION.cap_replacement:.0%} cap on average (not final) salary binds first, so H3 also "
                     f"needs a higher cap or final-salary base.")
    else:
        lines.append(f"  Accrual rate at which S3+S4 matches the status-quo replacement rate: "
                     f"about {be:.2%} per year of service.")
    lines += [
        "",
        "## M1 Data quality",
        "",
        ev.to_markdown(),
        "",
        f"Payroll on records needing verification (synthetic truth): {ctx.m1['payroll_leakage_riel'] / 1e9:,.0f} billion riel a year.",
        "A flag is a prompt to check a record against source documents, not a finding of wrongdoing.",
        "",
        "Recall by check type:",
        "",
        ctx.m1["recall_by_type"].to_markdown(),
        "",
        "## M2 Workforce projection (backtest: train to 2021, test 2022-2025)",
        "",
        bt2.to_markdown(),
        "",
        "![Exit model SHAP](charts/shap_exit_model.png)",
        "",
        "## M5 Macro forecast (backtest: train to 2017, test 2018-2025)",
        "",
        bt5.pivot(index="model", columns="target", values="MAPE_%").to_markdown(),
        "",
        f"Chosen models: {ctx.m5_choice}. Forecasts blend into long-run anchors over "
        f"{C.MC.convergence_years} years (real growth {C.MC.long_run_real_growth:.1%}, inflation "
        f"{C.MC.long_run_inflation:.1%}, revenue {C.MC.long_run_revenue_share:.0%} of GDP).",
        "",
        "## M6 What drives the results (SHAP)",
        "",
        f"Surrogate model across all scenarios for total cost % of GDP in 2046 (R² {shap_out['policy_r2']:.3f}):",
        "",
        shap_out["policy"].rename("mean |SHAP|, pp of GDP").round(3).to_frame().to_markdown(),
        "",
        "![Policy SHAP](charts/shap_policy.png)",
        "",
        f"Within the status quo, uncertainty drivers of 2046 total cost (R² {shap_out['uncertainty_r2']:.3f}):",
        "",
        shap_out["uncertainty"].rename("mean |SHAP|, pp of GDP").round(3).to_frame().to_markdown(),
        "",
    ]
    if opt:
        lines += optimiser_markdown(opt)
    if wave:
        lines += wave_markdown(wave)
    if risk:
        lines += risk_markdown(risk)
    lines += [
        "## Caveats",
        "",
        "- Synthetic data: every level is illustrative; relative scenario differences are the useful part.",
        "- S1 and S5 hold real pay flat for 50 years (raises equal inflation while GDP grows in real terms),",
        "  which is why their wage bill share falls steeply. Read long horizons as stress tests.",
        "- Only mandatory retirees draw pensions; survivor benefits and disability pensions are not modelled.",
        "- The XGBoost hazard holds the separation trend at its last observed year.",
        "- M5 models are fitted on about 30 annual points and scored on an 8-year test window, so the",
        "  Diebold-Mariano tests have little power and the model ranking can change with a new vintage of data.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    return text


def optimiser_markdown(opt: dict) -> list[str]:
    cols = ["retirement_age", "pension_formula", "accrual_rate", "contribution_rise_per_side", "salary_rule",
            "restrain_non_priority", "cost", "adequacy", "prob_fund_depleted"]
    lines = ["## AI reform optimiser", "",
             "Surrogate-assisted search (Gaussian-process surrogates, optimistic Pareto infill) over reform",
             "packages. Cost = average total cost 2027-2076, % of GDP (median path); adequacy = average",
             "replacement rate of new retirees 2047-2076. Scored on the first 200 Monte Carlo paths.", ""]
    for key, title in (("pension", "Pension levers only (pay rule held at the status quo)"),
                       ("pension_pay", "Pension, pay and hiring levers")):
        r = opt[key]
        s0 = r["scenarios"].loc["S0"]
        lines += [f"### {title}", "",
                  f"{r['evaluations']} simulator runs out of {r['grid_size']} possible packages. Surrogate "
                  f"cross-validated R²: cost {r['surrogate_r2']['cost']:.3f}, adequacy {r['surrogate_r2']['adequacy']:.3f}."]
        if "grid_check" in r:
            g = r["grid_check"]
            lines.append(f"Against brute force: {g['hypervolume_ratio']:.1%} of the true front's hypervolume, "
                         f"{g['true_front_found']} of {g['true_front_size']} front packages found exactly.")
        keep = r["front"][r["front"].adequacy >= s0["adequacy"] - 0.005]
        if len(keep):
            b = keep.sort_values("cost").iloc[0]
            lines.append(f"Cheapest package keeping status-quo adequacy ({s0['adequacy']:.0%}): cost "
                         f"{b['cost']:.2f}% vs {s0['cost']:.2f}% of GDP, retirement age {int(b['retirement_age'])}, "
                         f"{b['pension_formula']} formula, contributions +{b['contribution_rise_per_side'] * 100:.0f} pp per side, "
                         f"pay rule {b['salary_rule']}.")
        lines += ["", r["front"][cols].round(4).to_markdown(index=False), ""]
    return lines


def wave_markdown(wave: dict) -> list[str]:
    from .wave import HIGH
    bt = wave["backtest"]
    nat = wave["national"]["province"]
    tab = wave["tables"]["province"]
    cols = ["staff", "mean_age", "aged_50_plus", "leave_5y", "leave_h", "peak_year", "wave_start", "tier", "age", "service"]
    return ["## Retirement wave early warning", "",
            "Expected exits of today's staff by unit: retirement by the age rule, deaths by the mortality",
            "table and separations by the M2 XGBoost hazard. Status quo retirement ages.", "",
            f"Backtest: staff in post in {bt['origin']}, exits in {bt['years'][0]}-{bt['years'][-1]} by province and "
            f"sector ({bt['units']} units, {bt['unit_years']} unit-years). DM tests compare each baseline's "
            "squared errors with the model's (positive = baseline worse).", "",
            bt["table"].to_markdown(), "",
            f"Nationally {nat['share_5']:.0%} of today's staff leave within 5 years and {nat['share_h']:.0%} within 10. "
            f"Units are rated high when their 10-year share is at least {HIGH:.2f} times the national share. "
            f"SHAP columns (age, service) are percentage points against the national share (surrogate R² "
            f"{wave['surrogate_r2']:.3f}).", "",
            tab[cols].head(10).round(3).astype({"wave_start": "Int64"}).astype({"wave_start": "string"}).fillna("").to_markdown(), ""]


def risk_markdown(risk: dict) -> list[str]:
    ev = risk["evaluation"]
    sc = risk["scenarios"].copy()
    sc.columns = [f"by {c}" for c in sc.columns]
    return ["## Pension fund risk alert", "",
            "Chance the fund's reserve is used up by each year, from the Monte Carlo runs:", "",
            (sc * 100).round(0).to_markdown(), "",
            f"AI risk model trained on {ev['train_packages']} simulated reform packages, scored on "
            f"{ev['test_packages']} it never saw (Brier score and log loss: lower is better; ECE = average gap "
            "between stated chance and outcome; package error = average gap, in points, between the predicted "
            "chance and the share of that package's runs that ran out).", "",
            ev["classifier"].to_markdown(), "",
            f"Run-out year ranges, target coverage {ev['target_coverage']:.0%}:", "",
            ev["year_range"].to_markdown(), "",
            "What moves the run-out year (mean |SHAP|, years):", "",
            risk["shap"].round(2).rename("years").to_frame().to_markdown(), ""]
