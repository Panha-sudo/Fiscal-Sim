"""M3 Salary engine: pay-scale rules applied to the projected workforce.

Basic salary of a cell = entry salary of its framework x step increment per
year of service x the scenario's salary index. Allowances are a sector share
of basic salary. The salary index is where the scenario's pay rule acts:

  recent_average  fixed annual raise (baseline: fixed-growth assumption)
  inflation       raise equal to the forecast inflation path (from M5)
  targeted        higher raise for education and health, lower elsewhere

Units are computed once per workforce projection at base-year pay; the
index then scales them for every Monte Carlo path.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .m2_workforce import SERVICE

FW_BASE = np.array([C.BASE.base_salary[f] for f in C.FRAMEWORKS])
STEP = (1 + C.BASE.step_increment) ** SERVICE
ALLOW = np.array([C.BASE.allowance_rate[s] for s in C.SECTORS])


def monthly_basic_grid() -> np.ndarray:
    """Base-year monthly basic salary by [fw, service]."""
    return FW_BASE[:, None] * STEP[None, :]


def wage_units(stocks: np.ndarray) -> dict:
    """Annual pay at base-year salary levels, by [year, sector]."""
    basic = monthly_basic_grid()[None, None, :, None, None, :] * 12  # [1,1,fw,1,1,sv]
    basic_units = (stocks * basic).sum(axis=(2, 3, 4, 5))
    total_units = basic_units * (1 + ALLOW)[None, :]
    return {"basic": basic_units, "total": total_units}


def salary_index(sc: C.Scenario, cpi: np.ndarray) -> np.ndarray:
    """Index of pay levels [runs, years, sector], 1.0 in the base year."""
    runs, T = cpi.shape
    k = np.arange(T)
    if sc.salary_rule == "inflation":
        idx = np.repeat(cpi[:, :, None], len(C.SECTORS), axis=2)
    elif sc.salary_rule == "targeted":
        r = np.array([sc.targeted_raise_priority if s in C.PRIORITY_SECTORS else sc.targeted_raise_other
                      for s in C.SECTORS])
        idx = np.broadcast_to((1 + r)[None, None, :] ** k[None, :, None], (runs, T, len(C.SECTORS)))
    else:
        idx = np.broadcast_to(((1 + sc.recent_raise) ** k)[None, :, None], (runs, T, len(C.SECTORS)))
    return np.asarray(idx, dtype=float)


def wage_bill(units: dict, idx: np.ndarray) -> dict:
    """Nominal annual wage bill (basic + allowances) and basic payroll, [runs, years]."""
    return {"wage_bill": np.einsum("ts,rts->rt", units["total"], idx),
            "basic_payroll": np.einsum("ts,rts->rt", units["basic"], idx)}

