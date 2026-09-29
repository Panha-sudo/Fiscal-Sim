"""M4 Pension and fiscal engine: cohort actuarial model with Monte Carlo.

For every Monte Carlo path (macro shocks, pay index, mortality level, fund
return) and every retirement cohort from M2, the engine computes:

  initial benefit   current rule: 80% of final basic salary after 20+ years
                    accrual rule: accrual rate x years x average basic over
                    the last N years (capped); both with a minimum pension.
                    Under 20 years of service: lump sum, no pension.
  in payment        indexed to pay (default) or prices, paid while the
                    retiree survives (mortality table x the path's level)
  contributions     employee + employer rate x basic payroll
  fund              reserve earns a nominal return; when it runs out the
                    government covers the deficit (top-up)

Run 0 is the deterministic central path, which is also the proposal's
"deterministic projection" baseline for M4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .m2_workforce import AGES, SERVICE
from .m3_salary import ALLOW, monthly_basic_grid
from .synthetic import mortality_table

_Q = mortality_table().to_numpy()
_CUMHAZ = np.concatenate([[0.0], np.cumsum(-np.log(1 - np.clip(_Q, 0, 0.999999)))])  # H(0..x)


def survival(age0: int, n: np.ndarray, mort_mult: np.ndarray) -> np.ndarray:
    """P(alive n years after age0) for each path's mortality multiplier. Returns [runs, len(n)]."""
    a_end = np.minimum(age0 + n, len(_Q))
    H = _CUMHAZ[a_end] - _CUMHAZ[min(age0, len(_Q))]
    return np.exp(-mort_mult[:, None] * H[None, :])


def contribution_rates(sc: C.Scenario, T: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(T)
    extra = np.zeros(T)
    if sc.contribution_step:
        extra = np.minimum((k // sc.contribution_step_every) * sc.contribution_step, sc.contribution_cap)
    return sc.employee_contribution + extra, sc.employer_contribution + extra


def _avg_factor(sc: C.Scenario, idx: np.ndarray, t: int, revalue: bool) -> np.ndarray:
    """Average basic over the last N years relative to final basic, [runs, sector]."""
    N = sc.accrual_avg_years
    step = 1 + C.BASE.step_increment
    acc = np.zeros(idx.shape[::2])
    for j in range(N):
        if revalue:
            ratio = 1.0
        elif t - j >= 0:
            ratio = idx[:, t - j] / idx[:, t]
        else:  # before the base year: back-cast at the recent raise
            ratio = idx[:, 0] / idx[:, t] / (1 + sc.recent_raise) ** (j - t)
        acc = acc + ratio / step ** j
    return acc / N


def run(sc: C.Scenario, proj: dict, idx: np.ndarray, paths: dict, pensioners: pd.DataFrame,
        basic_payroll: np.ndarray, rules: C.PensionRules = C.PENSION) -> dict:
    R, T = paths["cpi"].shape
    mort = paths["mortality_mult"]
    years = proj["years"]
    # index that pensions in payment and the minimum pension follow, per sector: [runs, T, sector]
    if rules.indexation == "salary":
        pidx = idx
    else:
        pidx = np.repeat(paths["cpi"][:, :, None], idx.shape[2], axis=2)
    w0 = proj["stocks"][0].sum(axis=(1, 2, 3, 4))  # base-year headcount by sector
    pidx_all = (pidx * (w0 / w0.sum())[None, None, :]).sum(axis=2)  # economy-wide, for old pensioners

    spend = np.zeros((R, T))
    lump = np.zeros((R, T))
    new_pensioners = np.zeros(T)
    adequacy = []
    grid = monthly_basic_grid() * 12  # [fw, sv] annual basic, base-year pay
    floor0 = rules.min_pension * 12

    # existing pensioners
    by_age = pensioners.groupby("age")["monthly_pension"].sum() * 12
    for age, amount in by_age.items():
        spend += amount * pidx_all * survival(int(age), np.arange(T), mort)

    for t in range(T - 1):
        ret = proj["retirees"][t].sum(axis=2)  # [sector, fw, age, sv]
        if ret.sum() == 0:
            continue
        n_after = np.arange(T - t - 1)  # payment years t+1..T-1
        s_i, f_i, a_i, v_i = np.nonzero(ret)
        counts = ret[s_i, f_i, a_i, v_i]
        final = grid[f_i, v_i][None, :] * idx[:, t, s_i]  # [runs, cells]
        eligible = SERVICE[v_i] >= rules.min_service_years
        if sc.pension_formula == "accrual":
            avg = _avg_factor(sc, idx, t, rules.revalue_average_salary)
            rate = np.minimum(sc.accrual_rate * SERVICE[v_i], rules.cap_replacement)
            formula = rate[None, :] * final * avg[:, s_i]
        else:
            formula = rules.replacement_of_final_basic * final
        floor = floor0 * pidx[:, t, s_i]
        benefit = np.where(eligible[None, :], np.maximum(formula, floor), 0.0)
        lump[:, t] += (np.where(eligible, 0.0, 1.0)[None, :] * counts[None, :] * final / 12
                       * rules.lump_sum_months_per_year * SERVICE[v_i][None, :]).sum(axis=1)
        new_pensioners[t] = counts[eligible].sum()

        w = counts * eligible
        if w.sum():
            rr = benefit[0] / (final[0] * (1 + ALLOW[s_i]))
            adequacy.append({"year": int(years[t]), "retirees": float(w.sum()),
                             "avg_replacement": float(np.average(rr, weights=w)),
                             "avg_replacement_basic": float(np.average(benefit[0] / final[0], weights=w)),
                             "min_replacement": float(rr[eligible].min()),
                             "share_on_minimum": float(w[formula[0] <= floor[0]].sum() / w.sum()),
                             "avg_service": float(np.average(SERVICE[v_i], weights=w))})

        # payments from t+1 on, indexed, while alive
        ages = AGES[a_i]
        for age in np.unique(ages):
            surv = survival(int(age) + 1, n_after, mort)
            for s in np.unique(s_i):
                m = (ages == age) & (s_i == s)
                amount = (benefit[:, m] * counts[m][None, :]).sum(axis=1) / pidx[:, t, s]
                spend[:, t + 1:] += amount[:, None] * pidx[:, t + 1:, s] * surv

    ee, er = contribution_rates(sc, T)
    contrib_ee = basic_payroll * ee[None, :]
    contrib_er = basic_payroll * er[None, :]
    nominal_ret = (1 + paths["real_return"]) * (1 + paths["inflation"]) - 1

    fund = np.zeros((R, T))
    topup = np.zeros((R, T))
    F = np.full(R, C.BASE.pension_fund_reserve)
    for t in range(T):
        F = F * (1 + nominal_ret[:, t]) + contrib_ee[:, t] + contrib_er[:, t] - spend[:, t] - lump[:, t]
        short = F < 0
        topup[:, t] = np.where(short, -F, 0.0)
        F = np.where(short, 0.0, F)
        fund[:, t] = F
    depleted = topup > 0
    depletion_year = np.where(depleted.any(axis=1), years[np.argmax(depleted, axis=1)], 0)

    return {"pension_spending": spend + lump, "contrib_employee": contrib_ee, "contrib_employer": contrib_er,
            "gov_topup": topup, "fund_balance": fund, "depletion_year": depletion_year,
            "new_pensioners": new_pensioners, "adequacy": pd.DataFrame(adequacy)}
