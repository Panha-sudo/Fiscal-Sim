"""Knowledge base the scenario assistant cites: the model's own inputs and assumptions.

Entries mirror the thesis appendices so an answer's citations can be checked there:
    P1-P14  input placeholders (Appendix B.1), values read from config.py
    A1-A16  modelling assumptions (Appendix C), values read from config.py where they are set
    S0-S6   the policy scenarios
    D1-D8   definitions of the results the assistant reports
Texts are built from config at import time, so they stay true when a placeholder is replaced.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as C

B, P, MC, S0 = C.BASE, C.PENSION, C.MC, C.S0


def _pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


def _riel(x: float) -> str:
    return f"{x / 1e6:.2f}m riel"


def _shares(d: dict) -> str:
    return ", ".join(f"{k.replace('_', ' ')} {_pct(v)}" for k, v in d.items())


def _entries() -> list[dict]:
    s5 = C.SCENARIOS["S5"]
    e = [
        # ---- inputs (Appendix B.1) ----
        ("P1", "Civil servant headcount", f"The base-year headcount of civil servants under the Common Statute is {B.headcount:,}. "
         "Placeholder; official source: MCS / HRMIS aggregate.", "config.BaseYear.headcount", "headcount total number of civil servants how many staff workforce size employees"),
        ("P2", "Headcount by sector", f"Headcount split by sector: {_shares(B.sector_share)}. Placeholder; source: HRMIS.",
         "config.BaseYear.sector_share", "teachers education health administration sector share"),
        ("P3", "Headcount by framework", f"Headcount split by framework (grade group): {_shares(B.framework_share)}. "
         "Placeholder; source: HRMIS.", "config.BaseYear.framework_share", "grade framework category A B C D"),
        ("P4", "Entry basic salary and step increment", "Monthly entry basic salary by framework: "
         + ", ".join(f"{k} {_riel(v)}" for k, v in B.base_salary.items())
         + f"; basic salary rises {_pct(B.step_increment, 1)} per year of service. Placeholder; source: pay-scale sub-decree, "
         "2026 Budget Law.", "config.BaseYear.base_salary, step_increment", "pay scale basic salary riel grade steps"),
        ("P5", "Allowances", f"Allowances as a share of basic salary: {_shares(B.allowance_rate)}. Placeholder; source: allowance "
         "sub-decrees and payroll.", "config.BaseYear.allowance_rate", "allowance allowances bonus function position"),
        ("P6", "Pensioners and average pension", f"{B.pensioners:,} pensioners in the base year with an average pension of "
         f"{_riel(B.avg_pension_monthly)} a month. Placeholder; source: NSSF-C register.", "config.BaseYear.pensioners",
         "retirees pensioners number average pension"),
        ("P7", "Pension fund reserve", f"The NSSF-C reserve at the start of 2026 is {B.pension_fund_reserve / 1e12:.1f} trillion riel. "
         "Placeholder; source: NSSF-C financial statements.", "config.BaseYear.pension_fund_reserve",
         "fund reserve balance assets NSSF savings"),
        ("P8", "GDP and exchange rate", f"Nominal GDP in 2026 is {B.gdp / 1e12:.0f} trillion riel; {C.KHR_PER_USD:,} riel per US dollar. "
         "Placeholder; source: NIS, IMF WEO, NBC.", "config.BaseYear.gdp; KHR_PER_USD", "GDP economy size exchange rate dollar"),
        ("P9", "Recent average pay raise", f"Basic salaries rose on average {_pct(S0.recent_raise)} a year in 2019-2025. "
         "Placeholder; source: Budget Laws.", "config.Scenario.recent_raise", "pay raise salary increase recent average"),
        ("P10", "Staff allowed to work to 60", f"{_pct(B.share_eligible_age60)} of staff are in bodies allowed to work to age 60. "
         "Placeholder; source: MCS.", "config.BaseYear.share_eligible_age60", "work to sixty eligible bodies retirement 60"),
        ("P11", "Staff history", "Staff history 2012-2025 is a synthetic person-year panel. To be replaced by yearly HRMIS "
         "snapshots.", "data/staff_history.parquet", "history panel records exits resignations data"),
        ("P12", "Macro history", "GDP growth, inflation and revenue share for 1995-2025 are synthetic series. To be replaced by "
         "NIS, NBC, MEF, IMF and World Bank data.", "data/macro_history.csv", "inflation growth revenue history macro data"),
        ("P13", "Mortality table", "Mortality follows a Gompertz-Makeham table with life expectancy at 55 of about 22 years. "
         "Placeholder; source: UN World Population Prospects, NIS.", "synthetic.mortality_table",
         "life expectancy mortality deaths longevity"),
        ("P14", "Long-run economic anchors", f"Long-run real growth {_pct(MC.long_run_real_growth, 1)}, inflation "
         f"{_pct(MC.long_run_inflation)}, revenue {_pct(MC.long_run_revenue_share)} of GDP; forecasts vary around these in "
         f"{MC.runs:,} Monte Carlo runs. Source: IMF WEO; MEF medium-term revenue strategy.", "config.MonteCarlo",
         "growth inflation revenue long run assumptions uncertainty economy"),
        # ---- assumptions (Appendix C) ----
        ("A1", "Current pension formula", f"The current pension is {_pct(P.replacement_of_final_basic)} of final basic salary after at "
         f"least {P.min_service_years} years of service, the same after 20 or 30 years. Confirm with NSSF-C.",
         "config.PensionRules", "pension formula benefit rule final salary current"),
        ("A2", "Minimum pension", f"The minimum pension is {_riel(P.min_pension)} a month in 2026 and moves with the pay index.",
         "config.PensionRules.min_pension", "minimum pension floor poverty lowest"),
        ("A3", "Short careers", f"Staff with under {P.min_service_years} years of service get a lump sum of one month of final basic "
         "salary per year served, and no pension.", "config.PensionRules.lump_sum_months_per_year",
         "lump sum short service less than twenty years"),
        ("A4", "Contribution rates", f"Contributions are {_pct(S0.employee_contribution)} of basic salary from the employee and "
         f"{_pct(S0.employer_contribution)} from the government as employer.", "config.Scenario.employee_contribution",
         "contribution rate employee employer payroll deduction"),
        ("A5", "Pension indexation", "Pensions in payment are indexed to pay raises, not prices.", "config.PensionRules.indexation",
         "indexation increase pensions in payment prices inflation"),
        ("A6", "Retirement age", f"Staff retire at {S0.retirement_age:.0f}; staff in eligible bodies at 60 ([P10]). Reforms that raise "
         "the age phase it in over 10 years.", "config.Scenario.retirement_age", "retirement age retire 55 60 62"),
        ("A7", "Accrual pension formula", f"The accrual formula pays {_pct(S0.accrual_rate)} a year of service times the revalued "
         f"{S0.accrual_avg_years}-year average basic salary, capped at {_pct(P.cap_replacement)}. Other accrual rates can be tested.",
         "config.Scenario.accrual_rate", "accrual formula career average per year of service"),
        ("A8", "Contribution increases", f"Contribution rises go up 1 point on each side every 5 years; S5 goes up to "
         f"+{s5.contribution_cap * 100:.0f} points.", "config.Scenario.contribution_step", "contribution increase rise step"),
        ("A9", "Total cost", "Total cost = wage bill + employer pension contribution + government top-up after the fund runs out, "
         "as a share of GDP.", "simulate.run_scenario", "total cost definition fiscal cost budget"),
        ("A10", "Forecast convergence", f"Machine-learning macro forecasts blend into the long-run anchors ([P14]) over "
         f"{MC.convergence_years} years.", "config.MonteCarlo.convergence_years", "forecast convergence long run"),
        ("A11", "Hiring growth", f"Headcount grows {_pct(S0.hiring_growth)} a year net of exits.", "config.Scenario.hiring_growth",
         "hiring recruitment headcount growth new staff"),
        ("A12", "Fund return", f"The pension reserve earns a real return of {_pct(P.fund_return_real, 1)} a year.",
         "config.PensionRules.fund_return_real", "fund return investment interest reserve"),
        ("A13", "Allowances not pensionable", "Allowances are in the wage bill but do not count for pensions; replacement rates "
         "are measured on total pay.", "m4_pension", "allowances pensionable replacement total pay"),
        ("A14", "Pay raise rules", f"S0 raises basic pay {_pct(S0.recent_raise)} a year; S1 follows inflation; S2 raises education "
         f"and health by {_pct(S0.targeted_raise_priority)} and others by {_pct(S0.targeted_raise_other)}.",
         "config.Scenario.salary_rule", "pay raise rule inflation targeted teachers health"),
        ("A15", "Scope", "Armed forces and police are excluded.", "scope", "armed forces police military scope excluded"),
        ("A16", "No behavioural response", "Hiring, exits and retirement timing do not respond to pay or pension rules.",
         "m2_workforce", "behaviour response limitation incentives"),
        # ---- definitions of reported results ----
        ("D1", "Replacement rate", "Replacement rate = first-year pension divided by final total pay (basic plus allowances), "
         "averaged over new retirees in 2047-2076.", "simulate.summarize", "replacement rate adequacy generous pension level"),
        ("D2", "Fund run-out year", "The run-out year is the first year the NSSF-C reserve is used up; after it the government "
         "covers the gap (top-up). The model reports the median year across Monte Carlo runs and the share of runs that run out.",
         "m4_pension", "fund run out depletion deficit bankrupt reserve exhausted"),
        ("D3", "Monte Carlo uncertainty", "Each scenario runs on the same set of simulated economic paths (common random numbers); "
         "results are medians with 5th-95th percentile ranges.", "simulate.run_scenario", "uncertainty range probability runs"),
        ("D4", "Wage bill", "Wage bill = basic salary plus allowances for all civil servants, as a share of GDP.", "m3_salary",
         "wage bill salaries payroll cost"),
        ("D5", "Share on minimum pension", "Share of new retirees whose pension is raised to the minimum pension ([A2]).",
         "m4_pension", "minimum pension share poverty"),
        ("D6", "AI reform optimiser", "The optimiser searched reform packages (retirement age 55-62, formula and accrual rate, "
         "contribution rises 0-4 points) for the cheapest cost at each replacement rate, using Gaussian-process surrogates checked "
         "against the full grid.", "optimize.search", "optimiser cheapest best reform package pareto"),
        ("D7", "Retirement wave warning", "Expected exits combine the retirement age rule, the mortality table and the machine-"
         "learning exit model; a unit is High if at least 1.25 times the national share leaves in the period, Watch at 1.10 times.",
         "wave.unit_table", "retirement wave exits leave province ministry shortage"),
        ("D8", "Synthetic data", "All current results use synthetic data shaped like the real inputs; they illustrate the method "
         "and are not official estimates.", "README", "synthetic data made up illustrative official"),
    ]
    for code, sc in C.SCENARIOS.items():
        e.append((code, f"Scenario {code}: {sc.name}", _scenario_text(sc), "config.SCENARIOS", f"scenario {code} {sc.name.lower()}"))
    return [{"id": i, "title": t, "text": x, "where": w, "keywords": k} for i, t, x, w, k in e]


def _scenario_text(sc: C.Scenario) -> str:
    parts = [{"recent_average": f"pay rises {_pct(sc.recent_raise)} a year", "inflation": "pay follows inflation",
              "targeted": "higher raises for education and health"}[sc.salary_rule]]
    if sc.retirement_age_target != sc.retirement_age:
        parts.append(f"retirement age rises to {sc.retirement_age_target:.0f} over {sc.retirement_phase_years} years")
    parts.append("current pension formula" if sc.pension_formula == "current" else f"accrual formula at {_pct(sc.accrual_rate)} a year")
    if sc.contribution_cap:
        parts.append(f"contributions up {sc.contribution_cap * 100:.0f} points on each side")
    if sc.restrain_non_priority:
        parts.append("only replace leavers outside education and health")
    return "; ".join(parts).capitalize() + "."


ENTRIES = _entries()
BY_ID = {e["id"]: e for e in ENTRIES}


class Retriever:
    """TF-IDF search over titles, texts and keywords (English)."""

    def __init__(self, entries=ENTRIES):
        self.entries = entries
        docs = [f"{e['title']} {e['title']} {e['keywords']} {e['text']}" for e in entries]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.X = self.vec.fit_transform(docs)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[str]:
        if not query or not re.search(r"[A-Za-z]", query):
            return []
        s = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-s)[:k]
        return [self.entries[i]["id"] for i in order if s[i] >= min_score]


RETRIEVER = Retriever()
