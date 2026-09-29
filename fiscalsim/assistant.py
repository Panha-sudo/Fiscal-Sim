"""Scenario assistant: plain Khmer or English questions answered by running the simulator.

Two steps, each a single call to Gemma through the Gemini API:
    1. route  Gemma reads the question and calls one tool (function calling) with the settings it states.
    2. write  The tool runs the simulator, optimiser or wave model in Python. Gemma then writes a short answer
              from the tool's facts, writing numbers as placeholders and citing knowledge-base ids.
Every number in an answer must come from the model: placeholders are filled from the tool's facts, and an
answer with any other number is replaced by a fixed template. Without an API key the same tools run from a
keyword parser (`parse_rules`), which is also the baseline the AI is tested against.

Only the question, the tool settings and summary results are sent to the API, never staff records.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import knowledge as K
from . import simulate as S
from . import wave as W

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemma-4-31b-it"
CODES = list(C.SCENARIOS)
PAY_RULES = ("recent_average", "inflation", "targeted")
LABEL_YEARS = {C.BASE_YEAR, *C.HORIZONS, 2027, 2047}
KM_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")


def is_khmer(text: str) -> bool:
    return bool(re.search(r"[ក-៿]", text))


# ---------------------------------------------------------------- tools (function declarations)

TOOLS = [
    {"name": "run_reform",
     "description": "Run the simulator for a policy package and compare it with the status quo (S0). Use for what-if "
                    "questions about retirement age, pension formula, accrual rate, contribution rates, pay raises or "
                    "hiring, including questions about cost, pension levels or when the pension fund runs out.",
     "parameters": {"type": "OBJECT", "properties": {
         "base_scenario": {"type": "STRING", "enum": CODES,
                           "description": "Preset scenario to start from, when the question modifies one. Default S0."},
         "retirement_age": {"type": "INTEGER", "description": "Retirement age, 55 to 65. Rises are phased in over 10 years."},
         "pension_formula": {"type": "STRING", "enum": ["current", "accrual"],
                             "description": "current = 80% of final basic salary; accrual = a rate per year of service."},
         "accrual_rate_percent": {"type": "NUMBER", "description": "Accrual rate in percent per year of service, e.g. 2.5."},
         "contribution_rise_points": {"type": "NUMBER",
                                      "description": "Total rise in contribution rates, in percentage points on each "
                                                     "side (staff and government), 0 to 4."},
         "pay_rule": {"type": "STRING", "enum": list(PAY_RULES),
                      "description": "recent_average = 6% raises a year; inflation = raises follow inflation; "
                                     "targeted = higher raises for education and health."},
         "restrain_hiring": {"type": "BOOLEAN",
                             "description": "Only replace leavers outside education and health."}}}},
    {"name": "compare_scenarios",
     "description": "Compare preset scenarios S0-S6 on cost, pension level and fund run-out year. Use when the "
                    "question names preset scenarios or asks which scenario is best.",
     "parameters": {"type": "OBJECT", "properties": {
         "codes": {"type": "ARRAY", "items": {"type": "STRING", "enum": CODES},
                   "description": "Scenarios to compare. Leave empty for all."}}}},
    {"name": "find_best_reform",
     "description": "Look up the AI optimiser's results: the cheapest reform package that keeps a target average "
                    "replacement rate, or the most generous one within a cost limit.",
     "parameters": {"type": "OBJECT", "properties": {
         "goal": {"type": "STRING", "enum": ["cheapest_for_replacement", "most_adequate_within_cost"]},
         "target": {"type": "NUMBER", "description": "Replacement rate in percent (e.g. 60) for "
                                                     "cheapest_for_replacement, or cost in % of GDP (e.g. 3.5)."},
         "include_pay_levers": {"type": "BOOLEAN", "description": "Also allow changing pay raises and hiring."}},
         "required": ["goal", "target"]}},
    {"name": "staff_exits",
     "description": "Expected staff leaving (retirement, death, resignation) by province or ministry, and which units "
                    "face a retirement wave.",
     "parameters": {"type": "OBJECT", "properties": {
         "level": {"type": "STRING", "enum": ["province", "ministry"]},
         "unit": {"type": "STRING", "description": "A province or ministry name, in English, if one is named."},
         "years_ahead": {"type": "INTEGER", "description": "Years ahead, 5 to 15. Default 10."}}}},
    {"name": "explain_assumption",
     "description": "Look up the model's rules, assumptions, data sources and definitions.",
     "parameters": {"type": "OBJECT", "properties": {
         "topic": {"type": "STRING", "description": "What to look up, in English keywords."}},
         "required": ["topic"]}},
]
TOOL_NAMES = [t["name"] for t in TOOLS] + ["none"]


# ---------------------------------------------------------------- environment and results

@dataclass
class Env:
    ctx: object                 # simulator context (workforce model, base stock, pensioners, Monte Carlo paths)
    table: pd.DataFrame         # summary of the preset scenarios, indexed by code
    runs: int                   # Monte Carlo runs behind `table`
    synthetic: bool = True
    optimiser: dict = field(default_factory=dict)   # {"pension": search result, "pension_pay": ...}
    cells: pd.DataFrame | None = None               # staff counts by unit and cohort (wave.cells)
    cache: dict = field(default_factory=dict)


@dataclass
class Fact:
    value: object
    kind: str        # pct_gdp | pct | pp | year | count | age | ratio | text
    en: str
    km: str


@dataclass
class Result:
    tool: str
    settings: dict
    facts: dict
    sources: list
    table: pd.DataFrame | None = None


@dataclass
class Answer:
    question: str
    lang: str
    mode: str                 # "gemma" | "rules"
    tool: str
    settings: dict
    result: Result
    text: str
    cited: list
    grounded_raw: bool | None = None   # the model's own text used only model numbers (None in rules mode)
    fallback: str = ""               # why Gemma's text failed the checks, if it did
    error: str = ""                  # why Gemma could not be used, if it could not
    seconds: float = 0.0


def fmt(f: Fact, lang: str = "en") -> str:
    v, k = f.value, f.kind
    if k == "text":
        return str(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return {"en": "no run-out by 2076", "km": "មិនអស់ទុនមុនឆ្នាំ ២០៧៦"}[lang] if k == "year" else "–"
    return {"pct_gdp": f"{v:.2f}%", "pct": f"{v * 100:.1f}%".replace(".0%", "%"), "pp": f"{v:+.2f}", "year": f"{int(round(v))}",
            "count": f"{v:,.0f}", "age": f"{int(round(v))}", "ratio": f"{v:.2f}"}[k]


# ---------------------------------------------------------------- argument cleaning

def _num(x):
    try:
        return float(str(x).translate(KM_DIGITS).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def clean_args(tool: str, args: dict) -> dict:
    """Keep known settings, coerce types and clamp to the ranges the model supports."""
    a = dict(args or {})
    out = {}
    if tool == "run_reform":
        if str(a.get("base_scenario", "")).upper() in CODES:
            out["base_scenario"] = str(a["base_scenario"]).upper()
        if (x := _num(a.get("retirement_age"))) is not None:
            out["retirement_age"] = int(min(max(round(x), 55), 65))
        if (x := _num(a.get("accrual_rate_percent"))) is not None and x > 0:
            x = x * 100 if x < 0.1 else x  # accept 0.025 as 2.5%
            out["accrual_rate_percent"] = round(min(max(x, 1.0), 3.0), 2)
            out["pension_formula"] = "accrual"
        if a.get("pension_formula") in ("current", "accrual") and "pension_formula" not in out:
            out["pension_formula"] = a["pension_formula"]
        if (x := _num(a.get("contribution_rise_points"))) is not None:
            x = x * 100 if 0 < x < 0.1 else x
            out["contribution_rise_points"] = round(min(max(x, 0.0), 4.0), 1)
        if a.get("pay_rule") in PAY_RULES:
            out["pay_rule"] = a["pay_rule"]
        if isinstance(a.get("restrain_hiring"), bool):
            out["restrain_hiring"] = a["restrain_hiring"]
    elif tool == "compare_scenarios":
        codes = [str(c).upper() for c in (a.get("codes") or [])]
        out["codes"] = [c for c in CODES if c in codes] or list(CODES)
    elif tool == "find_best_reform":
        out["goal"] = a.get("goal") if a.get("goal") in ("cheapest_for_replacement", "most_adequate_within_cost") \
            else "cheapest_for_replacement"
        x = _num(a.get("target"))
        if out["goal"] == "cheapest_for_replacement":
            x = 60.0 if x is None else (x * 100 if x <= 1 else x)
        else:
            x = 3.5 if x is None else x
        out["target"] = round(x, 2)
        out["include_pay_levers"] = bool(a.get("include_pay_levers", False))
    elif tool == "staff_exits":
        out["level"] = a.get("level") if a.get("level") in ("province", "ministry") else "province"
        if a.get("unit"):
            out["unit"] = str(a["unit"]).strip()
        x = _num(a.get("years_ahead"))
        out["years_ahead"] = int(min(max(round(x), 5), 15)) if x is not None else 10
    elif tool == "explain_assumption":
        out["topic"] = str(a.get("topic") or "").strip()
    return out


def scenario_from(settings: dict) -> C.Scenario:
    base = C.SCENARIOS[settings.get("base_scenario", "S0")]
    kw = {}
    if "retirement_age" in settings:
        ra = float(settings["retirement_age"])
        kw.update(retirement_age_target=ra, retirement_phase_years=10 if ra > base.retirement_age else 0)
    if "pension_formula" in settings:
        kw["pension_formula"] = settings["pension_formula"]
    if "accrual_rate_percent" in settings:
        kw["accrual_rate"] = settings["accrual_rate_percent"] / 100
    if "contribution_rise_points" in settings:
        c = settings["contribution_rise_points"] / 100
        kw.update(contribution_step=0.01 if c else 0.0, contribution_step_every=5, contribution_cap=c)
    if "pay_rule" in settings:
        kw["salary_rule"] = settings["pay_rule"]
    if "restrain_hiring" in settings:
        kw["restrain_non_priority"] = settings["restrain_hiring"]
    return base.with_(code="Q", name="Question", **kw)


SCENARIO_KEY = ("retirement_age_target", "pension_formula", "accrual_rate", "contribution_cap", "salary_rule",
                "restrain_non_priority")


def scenario_signature(sc: C.Scenario) -> tuple:
    return tuple(None if (k == "accrual_rate" and sc.pension_formula == "current") else round(getattr(sc, k), 4)
                 if isinstance(getattr(sc, k), float) else getattr(sc, k) for k in SCENARIO_KEY)


# ---------------------------------------------------------------- tools (execution)

def _run(env: Env, sc: C.Scenario) -> tuple[dict, np.ndarray]:
    key = ("run", scenario_signature(sc))
    if key not in env.cache:
        res = S.run_scenario(env.ctx, sc)
        env.cache[key] = (S.summarize(res), res["depletion_year"])
    return env.cache[key]


def _chance_by(dep: np.ndarray, year: int) -> float:
    return float(((dep > 0) & (dep <= year)).mean())


def _runout_year(dep: np.ndarray) -> float:
    """Median run-out year over all runs; NaN when most runs never run out by 2076."""
    d = np.where(dep > 0, dep, np.inf).astype(float)
    m = float(np.median(d))
    return m if np.isfinite(m) else float("nan")


def lever_sources(settings: dict) -> list[str]:
    src = []
    if settings.get("base_scenario", "S0") != "S0":
        src.append(settings["base_scenario"])
    if "retirement_age" in settings:
        src += ["A6", "P10"]
    if settings.get("pension_formula") == "accrual" or "accrual_rate_percent" in settings:
        src.append("A7")
    elif settings.get("pension_formula") == "current":
        src.append("A1")
    if "contribution_rise_points" in settings:
        src += ["A4", "A8"]
    if "pay_rule" in settings:
        src += ["A14"] + (["P9"] if settings["pay_rule"] == "recent_average" else [])
    if settings.get("restrain_hiring"):
        src.append("A11")
    return src or ["S0"]


def tool_run_reform(env: Env, settings: dict) -> Result:
    sc = scenario_from(settings)
    s, dep = _run(env, sc)
    s0, dep0 = _run(env, C.S0)
    f = {
        "cost_2046": Fact(s["total_cost_gdp_2046_p50"], "pct_gdp", "Total cost in 2046, % of GDP (this package)",
                          "ថ្លៃដើមសរុបឆ្នាំ ២០៤៦ % នៃ GDP (កញ្ចប់នេះ)"),
        "cost_2046_s0": Fact(s0["total_cost_gdp_2046_p50"], "pct_gdp", "Total cost in 2046, % of GDP (status quo)",
                             "ថ្លៃដើមសរុបឆ្នាំ ២០៤៦ % នៃ GDP (ស្ថានភាពបច្ចុប្បន្ន)"),
        "cost_change_2046": Fact(s["total_cost_gdp_2046_p50"] - s0["total_cost_gdp_2046_p50"], "pp",
                                 "Change in 2046 total cost vs status quo, points of GDP",
                                 "បម្រែបម្រួលថ្លៃដើមសរុប ២០៤៦ ធៀបនឹងស្ថានភាពបច្ចុប្បន្ន (ពិន្ទុនៃ GDP)"),
        "cost_2076": Fact(s["total_cost_gdp_2076_p50"], "pct_gdp", "Total cost in 2076, % of GDP",
                          "ថ្លៃដើមសរុបឆ្នាំ ២០៧៦ % នៃ GDP"),
        "wage_bill_2046": Fact(s["wage_bill_gdp_2046_p50"], "pct_gdp", "Wage bill in 2046, % of GDP",
                               "ថវិកាបៀវត្សឆ្នាំ ២០៤៦ % នៃ GDP"),
        "gov_pension_2046": Fact(s["gov_pension_cost_gdp_2046_p50"], "pct_gdp",
                                 "Government pension cost in 2046, % of GDP", "ថ្លៃសោធនរបស់រដ្ឋឆ្នាំ ២០៤៦ % នៃ GDP"),
        "replacement": Fact(s["avg_replacement_2047_2076"], "pct",
                            "Average replacement rate of new retirees 2047-2076 (this package)",
                            "អត្រាជំនួសមធ្យមរបស់អ្នកចូលនិវត្តន៍ថ្មី ២០៤៧-២០៧៦ (កញ្ចប់នេះ)"),
        "replacement_s0": Fact(s0["avg_replacement_2047_2076"], "pct",
                               "Average replacement rate of new retirees 2047-2076 (status quo)",
                               "អត្រាជំនួសមធ្យម ២០៤៧-២០៧៦ (ស្ថានភាពបច្ចុប្បន្ន)"),
        "share_on_minimum": Fact(s["share_on_minimum_2047_2076"], "pct", "Share of new retirees on the minimum pension",
                                 "ចំណែកអ្នកចូលនិវត្តន៍ថ្មីដែលទទួលសោធនអប្បបរមា"),
        "runout_year": Fact(_runout_year(dep), "year", "Pension fund run-out year, median (this package)",
                            "ឆ្នាំមូលនិធិសោធនអស់ទុន មធ្យមភាគ (កញ្ចប់នេះ)"),
        "runout_year_s0": Fact(_runout_year(dep0), "year", "Pension fund run-out year, median (status quo)",
                               "ឆ្នាំមូលនិធិអស់ទុន (ស្ថានភាពបច្ចុប្បន្ន)"),
        "runout_chance_2040": Fact(_chance_by(dep, 2040), "pct", "Chance the fund has run out by 2040",
                                   "ឱកាសដែលមូលនិធិអស់ទុននៅឆ្នាំ ២០៤០"),
        "runout_chance_2076": Fact(_chance_by(dep, 2076), "pct", "Chance the fund has run out by 2076",
                                   "ឱកាសដែលមូលនិធិអស់ទុននៅឆ្នាំ ២០៧៦"),
        "headcount_2046": Fact(s["headcount_2046"], "count", "Civil servants in 2046", "ចំនួនមន្ត្រីរាជការឆ្នាំ ២០៤៦"),
        "runs": Fact(len(dep), "count", "Monte Carlo runs", "ចំនួនការក្លែងធ្វើ Monte Carlo"),
    }
    src = lever_sources(settings) + ["A9", "D1", "D2", "D3"] + (["D8"] if env.synthetic else [])
    return Result("run_reform", settings, f, list(dict.fromkeys(src)))


def tool_compare(env: Env, settings: dict) -> Result:
    f = {}
    for c in settings["codes"]:
        r = env.table.loc[c]
        f[f"{c}_cost_2046"] = Fact(r["total_cost_gdp_2046_p50"], "pct_gdp", f"{c}: total cost in 2046, % of GDP",
                                   f"{c}: ថ្លៃដើមសរុបឆ្នាំ ២០៤៦ % នៃ GDP")
        f[f"{c}_replacement"] = Fact(r["avg_replacement_2047_2076"], "pct", f"{c}: average replacement rate 2047-2076",
                                     f"{c}: អត្រាជំនួសមធ្យម ២០៤៧-២០៧៦")
        f[f"{c}_runout_year"] = Fact(r["median_depletion_year"] if r["prob_fund_depleted_by_2076"] >= 0.5 else np.nan, "year", f"{c}: fund run-out year (median)",
                                     f"{c}: ឆ្នាំមូលនិធិអស់ទុន")
        f[f"{c}_runout_chance"] = Fact(r["prob_fund_depleted_by_2076"], "pct", f"{c}: chance the fund runs out by 2076",
                                       f"{c}: ឱកាសមូលនិធិអស់ទុនមុន ២០៧៦")
    f["runs"] = Fact(env.runs, "count", "Monte Carlo runs", "ចំនួនការក្លែងធ្វើ Monte Carlo")
    src = list(settings["codes"]) + ["A9", "D1", "D2", "D3"] + (["D8"] if env.synthetic else [])
    return Result("compare_scenarios", settings, f, src)


def tool_best(env: Env, settings: dict) -> Result:
    from . import optimize as O
    key = "pension_pay" if settings["include_pay_levers"] else "pension"
    src = ["D6", "A9", "D1", "A6", "A7", "A8"] + (["A14"] if settings["include_pay_levers"] else []) + \
          (["D8"] if env.synthetic else [])
    opt = env.optimiser.get(key)
    if opt is None:
        return Result("find_best_reform", settings, {"status": Fact("not_run", "text", "Optimiser status", "")}, src)
    front, refs = opt["front"], opt["scenarios"]
    cheapest = settings["goal"] == "cheapest_for_replacement"
    pick = O.best_for_target(front, min_adequacy=settings["target"] / 100) if cheapest \
        else O.best_for_target(front, max_cost=settings["target"])
    f = {"target": Fact(settings["target"] / 100 if cheapest else settings["target"], "pct" if cheapest else "pct_gdp",
                        "Target replacement rate" if cheapest else "Cost limit, % of GDP (average 2027-2076)",
                        "អត្រាជំនួសគោលដៅ" if cheapest else "ដែនកំណត់ថ្លៃដើម % នៃ GDP"),
         "s0_cost": Fact(refs.loc["S0", "cost"], "pct_gdp", "Status quo: average total cost 2027-2076, % of GDP",
                         "ស្ថានភាពបច្ចុប្បន្ន៖ ថ្លៃដើមសរុបមធ្យម ២០២៧-២០៧៦ % នៃ GDP"),
         "s0_replacement": Fact(refs.loc["S0", "adequacy"], "pct", "Status quo: average replacement rate 2047-2076",
                                "ស្ថានភាពបច្ចុប្បន្ន៖ អត្រាជំនួសមធ្យម")}
    f["status"] = Fact("found" if pick is not None else "closest", "text", "Result", "")
    if pick is None:  # nothing meets the target: report the closest package
        pick = front.loc[front.adequacy.idxmax()] if cheapest else front.loc[front.cost.idxmin()]
    f["pkg_retirement_age"] = Fact(pick["retirement_age"], "age", "Package: retirement age", "កញ្ចប់៖ អាយុចូលនិវត្តន៍")
    f["pkg_formula"] = Fact(pick["pension_formula"], "text", "Package: pension formula", "កញ្ចប់៖ រូបមន្តសោធន")
    if pick["pension_formula"] == "accrual":
        f["pkg_accrual"] = Fact(pick["accrual_rate"], "pct", "Package: accrual rate per year of service",
                                "កញ្ចប់៖ អត្រាក្នុងមួយឆ្នាំសេវា")
    f["pkg_contribution_rise"] = Fact(pick["contribution_rise_per_side"] * 100, "age",
                                      "Package: contribution rise, points on each side", "កញ្ចប់៖ ការដំឡើងភាគទានម្ខាងៗ (ពិន្ទុ)")
    f["pkg_pay_rule"] = Fact(pick["salary_rule"], "text", "Package: pay rule", "កញ្ចប់៖ ច្បាប់បៀវត្ស")
    f["pkg_restrain"] = Fact("yes" if pick["restrain_non_priority"] else "no", "text", "Package: restrain hiring",
                             "កញ្ចប់៖ ទប់ការជ្រើសរើស")
    f["pkg_cost"] = Fact(pick["cost"], "pct_gdp", "Package: average total cost 2027-2076, % of GDP",
                         "កញ្ចប់៖ ថ្លៃដើមសរុបមធ្យម ២០២៧-២០៧៦ % នៃ GDP")
    f["pkg_replacement"] = Fact(pick["adequacy"], "pct", "Package: average replacement rate 2047-2076",
                                "កញ្ចប់៖ អត្រាជំនួសមធ្យម ២០៤៧-២០៧៦")
    f["runs"] = Fact(opt.get("paths", 200), "count", "Monte Carlo runs per package", "ចំនួនការក្លែងធ្វើក្នុងមួយកញ្ចប់")
    return Result("find_best_reform", settings, f, src)


def _match_unit(name: str, units: list[str]) -> str | None:
    low = {u.lower(): u for u in units}
    n = name.lower().replace("province", "").replace("ministry of", "").strip()
    if n in low:
        return low[n]
    for u in units:
        if n and (n in u.lower() or u.lower() in n):
            return u
    m = difflib.get_close_matches(n, list(low), n=1, cutoff=0.7)
    return low[m[0]] if m else None


def tool_exits(env: Env, settings: dict) -> Result:
    src = ["D7", "A6", "P13", "P11"] + (["D8"] if env.synthetic else [])
    if env.cells is None:
        return Result("staff_exits", settings, {"status": Fact("no_data", "text", "Result", "")}, src)
    years = np.arange(C.BASE_YEAR, C.BASE_YEAR + W.HORIZON)
    if "exits" not in env.cache:
        env.cache["exits"] = W.exit_paths(env.cells, env.ctx.workforce, C.S0, years)
    h, lvl = settings["years_ahead"], settings["level"]
    tab, nat = W.unit_table(env.cells, env.cache["exits"], years, lvl, horizon=h)
    units = [str(u) for u in tab.index]
    f = {"years_ahead": Fact(h, "age", "Years ahead", "ចំនួនឆ្នាំខាងមុខ"),
         "national_share": Fact(nat["share_h"], "pct", f"Share of all staff leaving within {h} years",
                                f"ចំណែកមន្ត្រីទាំងអស់ដែលចាកចេញក្នុង {h} ឆ្នាំ"),
         "n_high": Fact(int((tab.tier == "high").sum()), "count", "Units with a High warning", "អង្គភាពមានការព្រមានខ្ពស់"),
         "n_units": Fact(len(tab), "count", "Units", "ចំនួនអង្គភាព")}
    unit = _match_unit(settings["unit"], units) if settings.get("unit") else None
    if settings.get("unit") and unit is None:
        f["status"] = Fact("unit_not_found", "text", "Result", "")
    if unit is not None:
        r = tab.loc[unit]
        settings = {**settings, "unit": unit}
        f.update({"unit": Fact(unit, "text", "Unit", "អង្គភាព"),
                  "unit_staff": Fact(r["staff"], "count", "Staff today", "មន្ត្រីបច្ចុប្បន្ន"),
                  "unit_share": Fact(r["leave_h"], "pct", f"Share of its staff leaving within {h} years",
                                     f"ចំណែកមន្ត្រីដែលចាកចេញក្នុង {h} ឆ្នាំ"),
                  "unit_vs_national": Fact(r["vs_national"], "ratio", "Times the national share", "ដងធៀបនឹងជាតិ"),
                  "unit_tier": Fact(r["tier"], "text", "Warning", "ការព្រមាន"),
                  "unit_peak_year": Fact(r["peak_year"], "year", "Year with the most exits", "ឆ្នាំដែលចាកចេញច្រើនបំផុត")})
    else:
        for i, (u, r) in enumerate(tab.head(3).iterrows(), 1):
            f[f"top{i}"] = Fact(str(u), "text", f"Unit ranked {i}", f"អង្គភាពលំដាប់ទី {i}")
            f[f"top{i}_share"] = Fact(r["leave_h"], "pct", f"Unit ranked {i}: share leaving within {h} years",
                                      f"អង្គភាពលំដាប់ទី {i}៖ ចំណែកចាកចេញ")
    return Result("staff_exits", settings, f, src, table=tab.head(10))


def tool_explain(env: Env, settings: dict) -> Result:
    ids = K.RETRIEVER.search(settings.get("topic", ""), k=3)
    return Result("explain_assumption", settings, {}, ids)


def run_tool(tool: str, settings: dict, env: Env) -> Result:
    fn = {"run_reform": tool_run_reform, "compare_scenarios": tool_compare, "find_best_reform": tool_best,
          "staff_exits": tool_exits, "explain_assumption": tool_explain}.get(tool)
    return fn(env, settings) if fn else Result("none", {}, {}, [])


# ---------------------------------------------------------------- keyword baseline (no AI)

KM_TOPICS = {"រូបមន្តសោធន": "current pension formula", "ភាគទាន": "contribution rates", "អប្បបរមា": "minimum pension",
             "ចំនួនមន្ត្រី": "headcount", "ថ្លៃដើមសរុប": "total cost definition", "អតិផរណា": "inflation",
             "អាយុចូលនិវត្តន៍": "retirement age", "មូលនិធិ": "fund reserve", "GDP": "GDP"}
SCENARIO_WORDS = {"status quo": "S0", "inflation-indexed pay": "S1", "targeted pay": "S2", "combined reform": "S5",
                  "workforce restraint": "S6", "ស្ថានភាពបច្ចុប្បន្ន": "S0", "កំណែទម្រង់រួម": "S5"}
UNIT_WORDS = ("province", "provinces", "ministry", "ministries", "ខេត្ត", "ក្រសួង")
EXIT_WORDS = ("leave", "leaving", "lose", "exits", "retirement wave", "shortage", "ចាកចេញ", "បាត់បង់", "រលក")


def parse_rules(question: str, units: list[str] = ()) -> tuple[str, dict]:
    """Keyword and pattern matching: the no-AI baseline and the assistant's mode without an API key."""
    q = question.translate(KM_DIGITS)
    ql = q.lower()
    codes = [f"S{d}" for d in re.findall(r"\bs([0-6])\b", ql)]
    codes = list(dict.fromkeys(codes + [c for w, c in SCENARIO_WORDS.items() if w in ql]))
    pct = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", q)]

    if re.search(r"cheapest|lowest cost|least cost|ថោកបំផុត", ql):
        return "find_best_reform", {"goal": "cheapest_for_replacement", "target": pct[0] if pct else 60,
                                    "include_pay_levers": "pay" in ql or "បៀវត្ស" in q}
    if re.search(r"most generous|highest (pension|replacement)|best pension", ql) and re.search(r"gdp", ql):
        return "find_best_reform", {"goal": "most_adequate_within_cost", "target": pct[0] if pct else 3.5}
    if any(w in ql for w in UNIT_WORDS) and any(w in ql for w in EXIT_WORDS) or \
            any(u.lower() in ql for u in units) and any(w in ql for w in EXIT_WORDS):
        yrs = re.search(r"(\d+)\s*(?:years?|ឆ្នាំ)", ql)
        a = {"level": "ministry" if ("ministr" in ql or "ក្រសួង" in q) else "province",
             "years_ahead": int(yrs.group(1)) if yrs else 10}
        named = [u for u in units if u.lower() in ql]
        if named:
            a["unit"] = named[0]
        return "staff_exits", a

    a = {}
    m = re.search(r"(?:retire\w*|retirement age|អាយុចូលនិវត្តន៍|ចូលនិវត្តន៍)\D{0,25}?(\d{2})\b", ql)
    if m and 50 <= int(m.group(1)) <= 70:
        a["retirement_age"] = int(m.group(1))
    if re.search(r"accrual|per year of service|តាមឆ្នាំសេវា|ប្រមូលផ្តុំ", ql):
        a["pension_formula"] = "accrual"
        m = re.search(r"(\d(?:\.\d+)?)\s*%", q[q.lower().find("accrual") if "accrual" in ql else 0:])
        if m and 0.5 <= float(m.group(1)) <= 4:
            a["accrual_rate_percent"] = float(m.group(1))
    m = re.search(r"(?:contribution\w*|ភាគទាន)\D{0,40}?(\d(?:\.\d+)?)\s*(?:points?|pp|percentage|%|ពិន្ទុ)", ql)
    if m:
        a["contribution_rise_points"] = float(m.group(1))
    if re.search(r"inflation|អតិផរណា", ql) and re.search(r"pay|salar|wage|raise|index|បៀវត្ស", ql):
        a["pay_rule"] = "inflation"
    elif re.search(r"targeted|teachers and health|education and health|អប់រំ និងសុខាភិបាល", ql) and \
            re.search(r"pay|salar|wage|raise|បៀវត្ស", ql):
        a["pay_rule"] = "targeted"
    if re.search(r"only replace|replace only|restrain|hiring freeze|freeze hiring|ទប់", ql):
        a["restrain_hiring"] = True
    if a:
        if len(codes) == 1:
            a["base_scenario"] = codes[0]
        return "run_reform", a
    if codes or re.search(r"compare|which\b.*\bscenarios?|ប្រៀបធៀប|សេណារីយ៉ូណា", ql):
        return "compare_scenarios", {"codes": codes}
    if re.search(r"fund.*(run out|deplet|risk)|status quo|មូលនិធិ.*អស់", ql):
        return "run_reform", {}
    if re.search(r"\b(what|how|where|which|define|explain)\b|assum|formula|rate|source|រូបមន្ត|សន្មត|អត្រា|ប្រភព", ql):
        topic = q if re.search(r"[A-Za-z]{3}", q) else " ".join(v for k, v in KM_TOPICS.items() if k in q)
        if topic and K.RETRIEVER.search(topic):
            return "explain_assumption", {"topic": topic}
    return "none", {}


# ---------------------------------------------------------------- Gemma client

class AssistantError(RuntimeError):
    pass


ROUTER_PROMPT = """You route questions about Cambodia's civil service pay, workforce and NSSF-C pension simulator to exactly one tool.
Questions may be in Khmer or English. Call one function. Fill only the settings the question states and leave the rest out.
- What-if questions about retirement age, pension formula, accrual rate, contributions, pay raises or hiring, or about cost, pension levels or when the pension fund runs out: run_reform. If the question changes a preset scenario, set base_scenario.
- Questions that name preset scenarios or ask which scenario is best: compare_scenarios.
- Questions asking for the cheapest or best reform that meets a target: find_best_reform.
- Questions about staff leaving or retiring by province or ministry: staff_exits.
- Questions about the model's rules, assumptions, data sources or definitions: explain_assumption, with the topic in English.
If the question is about something else, do not call a function; reply OUT_OF_SCOPE.
Preset scenarios: {scenarios}."""

WRITER_PROMPT = """You explain results from Cambodia's civil service pay and pension simulator to government officials.
Write the answer in {language}, in two to five short sentences of plain prose, with no headings or lists.
Use only the FACTS and SOURCES given. Do not add numbers, estimates, reasons or advice of your own.
Write every number as its placeholder in curly braces, for example {{cost_2046}}. Do not type digits yourself, except a value quoted exactly from a source.
After each claim, cite the ids of the sources it rests on in square brackets, for example [A6] or [A9, D1]. Cite only ids listed under SOURCES.
If the facts do not answer the question, say so in one sentence."""


class GemmaClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 90, session=None, max_tries: int = 4):
        import requests
        self.key, self.model, self.timeout, self.max_tries = api_key, model, timeout, max_tries
        self.http = session or requests.Session()

    def _call(self, body: dict) -> dict:
        url = API_URL.format(model=self.model)
        wait = 2.0
        for attempt in range(self.max_tries):
            try:
                r = self.http.post(url, json=body, timeout=self.timeout,
                                   headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            except Exception as e:  # network error
                err = f"network error: {type(e).__name__}"
            else:
                if r.status_code == 200:
                    return r.json()
                try:
                    msg = r.json().get("error", {}).get("message", "")
                except ValueError:
                    msg = r.text[:200]
                err = f"HTTP {r.status_code}: {msg[:200]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise AssistantError(err)
                m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r'"retryDelay":\s*"([\d.]+)s"', r.text)
                if m:
                    wait = min(float(m.group(1)) + 1, 60)
            if attempt < self.max_tries - 1:
                time.sleep(wait)
                wait = min(wait * 2, 60)
        raise AssistantError(err)

    @staticmethod
    def _parts(resp: dict) -> list[dict]:
        try:
            return resp["candidates"][0]["content"].get("parts", [])
        except (KeyError, IndexError, TypeError):
            return []

    @classmethod
    def _text(cls, resp: dict) -> str:
        return "".join(p.get("text", "") for p in cls._parts(resp) if not p.get("thought")).strip()

    def route(self, question: str) -> tuple[str, dict]:
        scen = "; ".join(f"{k} {sc.name}" for k, sc in C.SCENARIOS.items())
        body = {"systemInstruction": {"parts": [{"text": ROUTER_PROMPT.format(scenarios=scen)}]},
                "contents": [{"role": "user", "parts": [{"text": question}]}],
                "tools": [{"functionDeclarations": TOOLS}],
                "generationConfig": {"temperature": 0}}
        resp = self._call(body)
        for p in self._parts(resp):
            if "functionCall" in p:
                fc = p["functionCall"]
                return (fc.get("name") if fc.get("name") in TOOL_NAMES else "none"), dict(fc.get("args") or {})
        text = self._text(resp)
        m = re.search(r"\{.*\}", text, flags=re.S)  # a call written as JSON text instead of a function call
        if m:
            try:
                j = json.loads(m.group(0))
                name = j.get("name") or j.get("tool")
                if name in TOOL_NAMES:
                    return name, dict(j.get("args") or j.get("arguments") or j.get("parameters") or {})
            except ValueError:
                pass
        return "none", {}

    def write(self, question: str, lang: str, result: Result) -> str:
        facts = "\n".join(f"{{{k}}} = {fmt(f, 'en')} | {f.en}" for k, f in result.facts.items()) or "(none)"
        srcs = "\n".join(f"[{i}] {K.BY_ID[i]['title']}: {K.BY_ID[i]['text']}" for i in result.sources if i in K.BY_ID)
        user = (f"QUESTION: {question}\n\nWHAT WAS RUN: {describe(result, 'en', cite=False)}\n\n"
                f"FACTS (placeholder = value | meaning):\n{facts}\n\nSOURCES:\n{srcs or '(none)'}")
        body = {"systemInstruction": {"parts": [{"text": WRITER_PROMPT.format(language="Khmer" if lang == "km" else "English")}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700}}
        return self._text(self._call(body))


def client_from_settings(get=None) -> GemmaClient | None:
    """A client when GEMINI_API_KEY is set (Streamlit secrets via `get`, or the environment), else None."""
    def read(name):
        v = None
        if get is not None:
            try:
                v = get(name)
            except Exception:
                v = None
        return v or os.environ.get(name)
    key = read("GEMINI_API_KEY") or read("GOOGLE_API_KEY")
    return GemmaClient(key, read("GEMMA_MODEL") or DEFAULT_MODEL) if key else None


# ---------------------------------------------------------------- grounding

CITE = re.compile(r"\[\s*([A-Z]\d{1,2}(?:\s*[,;]\s*[A-Z]\d{1,2})*)\s*\]")


def _norm(n: str) -> str:
    n = n.replace(",", "")
    return n.rstrip("0").rstrip(".") if "." in n else n


def numbers_in(text: str) -> list[str]:
    t = CITE.sub(" ", text.translate(KM_DIGITS))
    t = re.sub(r"\bS[0-6]\b", " ", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    return [_norm(n) for n in re.findall(r"\d+(?:\.\d+)?", t)]


def allowed_numbers(result: Result, question: str) -> set[str]:
    ok = {str(y) for y in LABEL_YEARS} | set(numbers_in(question))
    for f in result.facts.values():
        ok |= set(numbers_in(fmt(f, "en")))
        if isinstance(f.value, (int, float, np.floating)) and not (isinstance(f.value, float) and np.isnan(f.value)):
            v = float(f.value) * (100 if f.kind == "pct" else 1)
            ok |= {_norm(f"{abs(v):.{d}f}") for d in (0, 1, 2)}
    for i in result.sources:
        if i in K.BY_ID:
            ok |= set(numbers_in(K.BY_ID[i]["text"]))
    ok |= {"1", "2", "3", "4", "5", "10"}  # small counting words ("the 3 scenarios", "every 5 years")
    return ok


def fill(text: str, result: Result, lang: str) -> tuple[str, list[str]]:
    """Replace {placeholders} with formatted facts. Returns the text and any unknown placeholders."""
    unknown = []

    def sub(m):
        k = m.group(1)
        if k in result.facts:
            return fmt(result.facts[k], lang)
        unknown.append(k)
        return m.group(0)
    return re.sub(r"\{\s*([a-zA-Z0-9_]+)\s*\}", sub, text), unknown


def cited_ids(text: str) -> list[str]:
    ids = []
    for g in CITE.findall(text):
        ids += [x.strip() for x in re.split(r"[,;]", g)]
    return list(dict.fromkeys(ids))


def check(text: str, result: Result, question: str) -> tuple[bool, str]:
    """True when every number in the answer is a model value and every citation is one of the result's sources."""
    if "{" in text or "}" in text:
        return False, "unfilled placeholder"
    bad = [n for n in numbers_in(text) if n not in allowed_numbers(result, question)]
    if bad:
        return False, "numbers not from the model: " + ", ".join(bad[:5])
    wrong = [i for i in cited_ids(text) if i not in result.sources]
    if wrong:
        return False, "citations not among the sources: " + ", ".join(wrong)
    if result.sources and not cited_ids(text):
        return False, "no citations"
    return True, ""


# ---------------------------------------------------------------- template answers

def _c(ids, cite=True) -> str:
    ids = [i for i in ids if i]
    return f" [{', '.join(ids)}]" if cite and ids else ""


def describe(result: Result, lang: str, cite: bool = True) -> str:
    """One sentence saying what was run, with the sources of each setting."""
    s, km = result.settings, lang == "km"
    if result.tool == "run_reform":
        parts = []
        if s.get("base_scenario", "S0") != "S0":
            b = s["base_scenario"]
            parts.append((f"ផ្អែកលើ {b}" if km else f"starting from {b} {C.SCENARIOS[b].name}") + _c([b], cite))
        if "retirement_age" in s:
            a = s["retirement_age"]
            parts.append((f"អាយុចូលនិវត្តន៍ {a} ឆ្នាំ (ដំឡើងក្នុងរយៈពេល ១០ ឆ្នាំ)" if km else
                          f"retirement age {a}, phased in over 10 years") + _c(["A6"], cite))
        if s.get("pension_formula") == "accrual":
            r = s.get("accrual_rate_percent", C.S0.accrual_rate * 100)
            parts.append((f"រូបមន្តសោធនតាមឆ្នាំសេវា {r:g}% ក្នុងមួយឆ្នាំ" if km else
                          f"accrual pension of {r:g}% per year of service") + _c(["A7"], cite))
        elif s.get("pension_formula") == "current":
            parts.append(("រូបមន្តសោធនបច្ចុប្បន្ន" if km else "current pension formula") + _c(["A1"], cite))
        if "contribution_rise_points" in s:
            p = s["contribution_rise_points"]
            parts.append((f"ភាគទានកើន {p:g} ពិន្ទុម្ខាងៗ (១ ពិន្ទុរៀងរាល់ ៥ ឆ្នាំ)" if km else
                          f"contributions up {p:g} points on each side, 1 point every 5 years") + _c(["A4", "A8"], cite))
        if "pay_rule" in s:
            parts.append({"recent_average": ("បៀវត្សកើន ៦% ក្នុងមួយឆ្នាំ" if km else "pay rises 6% a year"),
                          "inflation": ("បៀវត្សកើនតាមអតិផរណា" if km else "pay follows inflation"),
                          "targeted": ("ដំឡើងបៀវត្សខ្ពស់ជាងសម្រាប់អប់រំ និងសុខាភិបាល" if km else
                                       "higher raises for education and health")}[s["pay_rule"]]
                         + _c(["A14"], cite))
        if s.get("restrain_hiring"):
            parts.append(("ជំនួសតែអ្នកចាកចេញក្នុងវិស័យអប់រំ និងសុខាភិបាល" if km else
                          "outside education and health, leavers are not replaced") + _c(["A11"], cite))
        if not parts:
            return ("ស្ថានភាពបច្ចុប្បន្ន (S0)" if km else "the status quo (S0)") + _c(["S0"], cite)
        return "; ".join(parts)
    if result.tool == "compare_scenarios":
        return ("ប្រៀបធៀប " if km else "compared ") + ", ".join(s["codes"])
    if result.tool == "find_best_reform":
        if s["goal"] == "cheapest_for_replacement":
            return (f"កញ្ចប់ថោកបំផុតដែលរក្សាអត្រាជំនួស {s['target']:g}%" if km else
                    f"cheapest optimiser package keeping an average replacement rate of at least {s['target']:g}%")
        return (f"កញ្ចប់ដែលផ្តល់សោធនល្អបំផុត ក្នុងថ្លៃដើម {s['target']:g}% នៃ GDP" if km else
                f"most generous optimiser package costing at most {s['target']:g}% of GDP")
    if result.tool == "staff_exits":
        who = s.get("unit") or ({"province": "ខេត្ត", "ministry": "ក្រសួង"}[s["level"]] if km else s["level"] + "s")
        return (f"ការចាកចេញរបស់មន្ត្រី {who} ក្នុង {s['years_ahead']} ឆ្នាំខាងមុខ" if km else
                f"expected staff exits, {who}, next {s['years_ahead']} years")
    if result.tool == "explain_assumption":
        return ("ស្វែងរកក្នុងការសន្មតរបស់ម៉ូដែល៖ " if km else "looked up the model's assumptions: ") + s.get("topic", "")
    return ""


def template(result: Result, lang: str) -> str:
    """A fixed answer built only from the facts; used without an API key and when a generated answer fails the checks."""
    f, km = result.facts, lang == "km"
    v = lambda k: fmt(f[k], lang)  # noqa: E731
    syn = _c(["D8"]) if "D8" in result.sources else ""
    if result.tool == "run_reform":
        d = describe(result, lang)
        d = d[0].upper() + d[1:]
        never = isinstance(f["runout_year"].value, float) and np.isnan(f["runout_year"].value)
        runout = ({True: "ក្នុងការក្លែងធ្វើភាគច្រើន មូលនិធិសោធនមិនអស់ទុនមុនឆ្នាំ ២០៧៦ ទេ",
                   False: f"មូលនិធិសោធនអស់ទុននៅឆ្នាំ {v('runout_year')}"} if km else
                  {True: "In most runs the pension fund does not run out by 2076",
                   False: f"The pension fund runs out in {v('runout_year')}"})[never]
        if km:
            return (f"{d}។ ថ្លៃដើមសរុបឆ្នាំ ២០៤៦ គឺ {v('cost_2046')} នៃ GDP ធៀបនឹង {v('cost_2046_s0')} ក្នុងស្ថានភាពបច្ចុប្បន្ន "
                    f"({v('cost_change_2046')} ពិន្ទុ) [A9]។ អ្នកចូលនិវត្តន៍ថ្មីឆ្នាំ ២០៤៧-២០៧៦ ទទួលបានជាមធ្យម "
                    f"{v('replacement')} នៃប្រាក់បៀវត្សចុងក្រោយ ធៀបនឹង {v('replacement_s0')} [D1]។ {runout} "
                    f"(ស្ថានភាពបច្ចុប្បន្ន៖ {v('runout_year_s0')}) ហើយឱកាសអស់ទុនមុនឆ្នាំ ២០៧៦ គឺ "
                    f"{v('runout_chance_2076')} [D2]។ លទ្ធផលពីការក្លែងធ្វើ {v('runs')} ដង{_c(['D3'])}{syn}។")
        return (f"{d}. Total cost in 2046 is {v('cost_2046')} of GDP, against {v('cost_2046_s0')} under the status quo "
                f"({v('cost_change_2046')} points) [A9]. New retirees in 2047-2076 get on average {v('replacement')} of "
                f"their final pay, against {v('replacement_s0')} now [D1]. {runout} (status quo: {v('runout_year_s0')}), with a "
                f"{v('runout_chance_2076')} chance of running out by 2076 [D2]. Medians of {v('runs')} Monte Carlo runs{_c(['D3'])}{syn}.")
    if result.tool == "compare_scenarios":
        from .i18n import scenario_name
        lines = []
        for c in result.settings["codes"]:
            n = scenario_name(c, lang)
            lines.append(f"{c} {n}: " + (f"ថ្លៃដើម {v(c + '_cost_2046')} នៃ GDP ឆ្នាំ ២០៤៦ អត្រាជំនួស {v(c + '_replacement')} "
                                        f"មូលនិធិអស់ទុន {v(c + '_runout_year')}" if km else
                                        f"total cost {v(c + '_cost_2046')} of GDP in 2046, replacement rate "
                                        f"{v(c + '_replacement')}, fund runs out {v(c + '_runout_year')}") + _c([c]))
        tail = (" លទ្ធផលមធ្យមពីការក្លែងធ្វើ " if km else " Medians of ") + v("runs") + \
               (" ដង" if km else " Monte Carlo runs") + _c(["A9", "D1", "D2"]) + syn
        return ("។ " if km else ". ").join(lines) + ("។" if km else ".") + tail
    if result.tool == "find_best_reform":
        st = f["status"].value
        if st == "not_run":
            return ("ឧបករណ៍ AI ស្វែងរកកំណែទម្រង់មិនទាន់ដំណើរការលើទិន្នន័យនេះទេ។ សូមដំណើរការវានៅផ្ទាំង AI ស្វែងរកកំណែទម្រង់ជាមុនសិន"
                    if km else "The AI reform optimiser has not been run on this data yet. Run it in the AI reform "
                               "optimiser tab first") + _c(["D6"]) + ("។" if km else ".")
        pre = ""
        if st == "closest":
            pre = (f"គ្មានកញ្ចប់ណាសម្រេចគោលដៅ {v('target')} ទេ [D6]។ កញ្ចប់ដែលជិតបំផុត៖ " if km else
                   f"No package the optimiser scored meets the target of {v('target')} [D6]. The closest one: ")
        form = (f"តាមឆ្នាំសេវា {v('pkg_accrual')}" if km else f"accrual at {v('pkg_accrual')} a year") \
            if "pkg_accrual" in f else ("រូបមន្តបច្ចុប្បន្ន" if km else "the current formula")
        if km:
            return pre + (f"កញ្ចប់៖ អាយុចូលនិវត្តន៍ {v('pkg_retirement_age')} [A6], សោធន{form} [A7], ភាគទានកើន "
                    f"{v('pkg_contribution_rise')} ពិន្ទុម្ខាងៗ [A8]។ ថ្លៃដើមសរុបមធ្យម ២០២៧-២០៧៦ គឺ {v('pkg_cost')} នៃ GDP "
                    f"ធៀបនឹង {v('s0_cost')} ក្នុងស្ថានភាពបច្ចុប្បន្ន [A9] ហើយអត្រាជំនួសមធ្យមគឺ {v('pkg_replacement')} "
                    f"(ស្ថានភាពបច្ចុប្បន្ន៖ {v('s0_replacement')}) [D1]។ លទ្ធផលពីឧបករណ៍ AI ស្វែងរកកំណែទម្រង់ [D6]{syn}។")
        lead = "retirement age" if pre else "The optimiser's package: retirement age"
        return pre + (f"{lead} {v('pkg_retirement_age')} [A6], {form} [A7], contributions up "
                f"{v('pkg_contribution_rise')} points on each side [A8], pay rule {v('pkg_pay_rule').replace('_', ' ')}. "
                f"Its average total cost for 2027-2076 is {v('pkg_cost')} of GDP, against {v('s0_cost')} for the status "
                f"quo [A9], with an average replacement rate of {v('pkg_replacement')} (status quo: "
                f"{v('s0_replacement')}) [D1]. From the AI reform optimiser [D6]{syn}.")
    if result.tool == "staff_exits":
        st = f.get("status")
        if st is not None and st.value == "no_data":
            return ("មិនមានទិន្នន័យមន្ត្រីតាមអង្គភាពទេ។" if km else "No staff records by unit are loaded.")
        if "unit" in f:
            if km:
                return (f"{v('unit')}៖ {v('unit_share')} នៃមន្ត្រី {v('unit_staff')} នាក់ ត្រូវបានរំពឹងថានឹងចាកចេញក្នុង "
                        f"{v('years_ahead')} ឆ្នាំ ធៀបនឹង {v('national_share')} ទូទាំងប្រទេស ({v('unit_vs_national')} ដង "
                        f"ការព្រមាន៖ {v('unit_tier')}) [D7]។ ឆ្នាំដែលចាកចេញច្រើនបំផុតគឺ {v('unit_peak_year')} [A6, P13]{syn}។")
            return (f"{v('unit')}: {v('unit_share')} of its {v('unit_staff')} staff are expected to leave within "
                    f"{v('years_ahead')} years, against {v('national_share')} nationally ({v('unit_vs_national')} times "
                    f"the national share; warning: {v('unit_tier')}) [D7]. Exits peak in {v('unit_peak_year')} "
                    f"[A6, P13]{syn}.")
        miss = ("រកមិនឃើញអង្គភាពដែលបានដាក់ឈ្មោះ។ " if km else "The named unit was not found. ") \
            if st is not None and st.value == "unit_not_found" else ""
        if km:
            return (f"{miss}ទូទាំងប្រទេស {v('national_share')} នៃមន្ត្រីនឹងចាកចេញក្នុង {v('years_ahead')} ឆ្នាំ។ ខ្ពស់បំផុត៖ "
                    f"{v('top1')} ({v('top1_share')}), {v('top2')} ({v('top2_share')}), {v('top3')} ({v('top3_share')})។ "
                    f"{v('n_high')} ក្នុងចំណោម {v('n_units')} មានការព្រមានខ្ពស់ [D7]{syn}។")
        return (f"{miss}Nationally, {v('national_share')} of staff are expected to leave within {v('years_ahead')} years. "
                f"Highest: {v('top1')} ({v('top1_share')}), {v('top2')} ({v('top2_share')}), {v('top3')} "
                f"({v('top3_share')}). {v('n_high')} of {v('n_units')} units have a High warning [D7]{syn}.")
    if result.tool == "explain_assumption":
        if not result.sources:
            return ("រកមិនឃើញការសន្មតដែលពាក់ព័ន្ធទេ។" if km else "I could not find a matching assumption in the model.")
        head = "យោងតាមការសន្មតរបស់ម៉ូដែល៖ " if km else ""
        return head + " ".join(f"{K.BY_ID[i]['title']}: {K.BY_ID[i]['text']} [{i}]" for i in result.sources[:2])
    return ("ខ្ញុំអាចឆ្លើយសំណួរអំពីសេណារីយ៉ូបៀវត្ស បុគ្គលិក និងសោធនរបស់មន្ត្រីរាជការក្នុងម៉ូដែលនេះ ឧទាហរណ៍ "
            "«បើអាយុចូលនិវត្តន៍ ៦២ ឆ្នាំ?»" if km else
            "I can answer questions about civil service pay, staff and pension scenarios in this model, for example "
            "\"What if the retirement age is 62?\"")


# ---------------------------------------------------------------- main entry

def answer(question: str, env: Env, client: GemmaClient | None = None, lang: str | None = None) -> Answer:
    t0 = time.time()
    lang = lang or ("km" if is_khmer(question) else "en")
    units = [str(u) for col in ("province", "ministry") for u in env.cells[col].cat.categories] \
        if env.cells is not None else []
    note, error, grounded = "", "", None
    if client is not None:
        try:
            tool, args = client.route(question)
        except AssistantError as e:
            client, error = None, str(e)
    if client is None:
        tool, args = parse_rules(question, units)
    settings = clean_args(tool, args)
    result = run_tool(tool, settings, env)
    text = template(result, lang)
    if client is not None and result.tool != "none":
        try:
            raw = client.write(question, lang, result)
            filled, unknown = fill(raw, result, lang)
            ok, why = check(filled, result, question) if not unknown else (False, "unknown placeholders: " + ", ".join(unknown))
            grounded = ok
            if ok and filled.strip():
                text = filled
            else:
                note = why or "empty answer"
        except AssistantError as e:
            error = str(e)
    return Answer(question, lang, "gemma" if client is not None else "rules", result.tool, result.settings, result,
                  text, cited_ids(text), grounded, note, error, time.time() - t0)


def env_from_bundle(bundle: dict, ctx=None) -> Env:
    """Assistant environment for the committed synthetic results."""
    from types import SimpleNamespace
    E = bundle["engine"]
    ctx = ctx or SimpleNamespace(workforce=E["workforce"], stock0=E["stock0"], data={"pensioners": E["pensioners"]},
                                 paths=E["paths"])
    return Env(ctx=ctx, table=bundle["table"], runs=bundle["runs"], synthetic=True,
               optimiser=dict(bundle.get("optimiser") or {}), cells=bundle.get("wave_cells"))
