"""Base-year assumptions and policy scenarios S0-S6.

Every number here is an ILLUSTRATIVE PLACEHOLDER shaped like the real inputs
(HRMIS, Budget Laws, NSSF-C rules, NIS/NBC/IMF macro data). Replace each value
marked `# SOURCE:` with the official figure before using results in the thesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

BASE_YEAR = 2026
END_YEAR = 2076
HORIZONS = (2036, 2046, 2076)
SEED = 20260927

SECTORS = ("education", "health", "general_admin")
PRIORITY_SECTORS = ("education", "health")
FRAMEWORKS = ("A", "B", "C", "D")  # Cambodian civil service frameworks, A = most senior

KHR_PER_USD = 4100


@dataclass(frozen=True)
class BaseYear:
    # SOURCE: MCS / HRMIS aggregate headcount (civil servants under the Common Statute)
    headcount: int = 200_000
    # SOURCE: HRMIS sector split
    sector_share: dict = field(default_factory=lambda: {"education": 0.46, "health": 0.13, "general_admin": 0.41})
    # SOURCE: HRMIS framework split
    framework_share: dict = field(default_factory=lambda: {"A": 0.10, "B": 0.38, "C": 0.42, "D": 0.10})
    # SOURCE: pay-scale sub-decree. Monthly basic salary in riel at entry to each framework (2026 level).
    base_salary: dict = field(default_factory=lambda: {"A": 2_300_000, "B": 1_750_000, "C": 1_500_000, "D": 1_430_000})
    step_increment: float = 0.012  # basic salary rises 1.2% per year of service (grade steps)
    # Allowances as a share of basic salary (function, position, sector allowances).
    allowance_rate: dict = field(default_factory=lambda: {"education": 0.35, "health": 0.40, "general_admin": 0.30})
    # SOURCE: NSSF-C beneficiary register
    pensioners: int = 45_000
    avg_pension_monthly: float = 1_250_000
    pension_fund_reserve: float = 1.2e12  # riel, NSSF-C reserve at start of base year
    # SOURCE: NIS / IMF WEO. Nominal GDP in the base year, riel.
    gdp: float = 205e12
    share_eligible_age60: float = 0.12  # staff in bodies allowed to work to 60


@dataclass(frozen=True)
class Scenario:
    code: str
    name: str
    salary_rule: str = "recent_average"  # recent_average | inflation | targeted
    recent_raise: float = 0.06  # SOURCE: average basic-salary raise 2019-2025 (Budget Laws)
    targeted_raise_priority: float = 0.08
    targeted_raise_other: float = 0.03
    retirement_age: float = 55.0
    retirement_age_target: float = 55.0  # phased increase target
    retirement_phase_years: int = 0
    pension_formula: str = "current"  # current | accrual
    accrual_rate: float = 0.02  # per year of service, accrual formula
    accrual_avg_years: int = 10  # averaging window for pensionable salary
    # SOURCE: NSSF-C contribution rules. Shares of basic salary.
    employee_contribution: float = 0.06
    employer_contribution: float = 0.12
    contribution_step: float = 0.0  # added to each side every `contribution_step_every` years
    contribution_step_every: int = 5
    contribution_cap: float = 0.0  # max total increase per side
    hiring_growth: float = 0.01  # net headcount growth per year (on top of replacing exits)
    restrain_non_priority: bool = False  # S6: replace-only hiring outside education and health

    def with_(self, **kw) -> "Scenario":
        return replace(self, **kw)


S0 = Scenario("S0", "Status quo")
SCENARIOS = {
    "S0": S0,
    "S1": S0.with_(code="S1", name="Inflation-indexed pay", salary_rule="inflation"),
    "S2": S0.with_(code="S2", name="Targeted pay (education and health)", salary_rule="targeted"),
    "S3": S0.with_(code="S3", name="Retirement age to 60 over 10 years", retirement_age_target=60.0, retirement_phase_years=10),
    "S4": S0.with_(code="S4", name="Accrual pension formula", pension_formula="accrual"),
    "S5": S0.with_(
        code="S5", name="Combined reform", salary_rule="inflation", retirement_age_target=60.0,
        retirement_phase_years=10, pension_formula="accrual",
        contribution_step=0.01, contribution_step_every=5, contribution_cap=0.03,
    ),
    "S6": S0.with_(code="S6", name="Workforce restraint", restrain_non_priority=True),
}


@dataclass(frozen=True)
class PensionRules:
    """Current NSSF-C benefit rule as modelled (SOURCE: NSSF-C sub-decree, ILO actuarial review)."""
    min_service_years: int = 20
    replacement_of_final_basic: float = 0.80  # same benefit after 20 or 30 years (ILO finding)
    min_pension: float = 1_400_000  # monthly riel floor in base year, moves with the pay index
    indexation: str = "salary"  # pensions in payment follow pay raises ("salary") or prices ("cpi")
    revalue_average_salary: bool = True  # accrual formula: career salaries revalued to retirement-year pay
    lump_sum_months_per_year: float = 1.0  # below minimum service: lump sum per year served
    cap_replacement: float = 0.80
    fund_return_real: float = 0.015  # real return on reserve


@dataclass(frozen=True)
class MonteCarlo:
    runs: int = 10_000
    long_run_real_growth: float = 0.045  # SOURCE: IMF WEO / World Bank long-run projection
    long_run_inflation: float = 0.03
    long_run_revenue_share: float = 0.24  # SOURCE: MEF medium-term revenue strategy
    convergence_years: int = 15  # ML forecasts blend into long-run anchors over this span
    growth_sd: float = 0.018
    inflation_sd: float = 0.015
    revenue_share_sd: float = 0.008
    mortality_sd: float = 0.10  # lognormal sd of mortality level multiplier
    return_sd: float = 0.02


BASE = BaseYear()
PENSION = PensionRules()
MC = MonteCarlo()
