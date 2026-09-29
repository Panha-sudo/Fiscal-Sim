"""Synthetic inputs shaped like the real data sources in the architecture diagram.

Real HRMIS, payroll and NSSF-C microdata are not available, so this module
creates data with the same fields and plausible structure, calibrated to the
placeholder totals in `config.py` (proposal section 4.2.1, synthetic fallback).

Outputs:
  staff_history.parquet    person-year panel 2012-2025 with province and ministry (trains M2)
  hrmis_2026.csv           base-year staff records with injected anomalies (M1 input)
  hrmis_2026_truth.csv     which records are anomalies (for M1 evaluation only)
  pensioners_2026.csv      NSSF-C pensioner register
  macro_history.csv        GDP, inflation, revenue 1995-2025 (M5 input)
  mortality.csv            q(x) table
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

PROVINCES = [
    "Phnom Penh", "Kandal", "Siem Reap", "Battambang", "Kampong Cham", "Prey Veng", "Takeo",
    "Kampong Speu", "Banteay Meanchey", "Kampot", "Pursat", "Kampong Thom", "Svay Rieng",
    "Kratie", "Kampong Chhnang", "Preah Sihanouk", "Tbong Khmum", "Koh Kong", "Ratanakiri",
    "Mondulkiri", "Preah Vihear", "Stung Treng", "Oddar Meanchey", "Pailin", "Kep",
]
MINISTRIES = {
    "education": ["MoEYS"],
    "health": ["MoH"],
    "general_admin": ["MEF", "MoI", "MAFF", "MCS", "MoC", "MLVT", "MPWT", "MoE", "MoJ", "MRD"],
}
# 2019 census population, thousands (NIS). Civil servants are assumed to follow population,
# with ministry headquarters staff concentrated in Phnom Penh.
PROVINCE_POP = [2282, 1195, 1006, 997, 895, 1057, 900, 877, 861, 592, 411, 677, 525, 373, 527, 310, 776,
                123, 204, 88, 254, 160, 276, 72, 42]
FAMILY = ["Sok", "Chan", "Kim", "Heng", "Ly", "Chea", "Keo", "Nguon", "Sam", "Touch", "Pich", "Mao",
          "Ouk", "Srey", "Hun", "Lim", "Tep", "Yim", "Phan", "Meas", "Prak", "Seng", "Vann", "Noun"]
GIVEN = ["Dara", "Sophea", "Vuthy", "Sokha", "Bopha", "Rithy", "Chenda", "Pisey", "Visal", "Sreymom",
         "Kosal", "Channary", "Samnang", "Rachana", "Veasna", "Kanha", "Sopheak", "Bunthoeun",
         "Sreyneang", "Piseth", "Leakhena", "Makara", "Sovann", "Thida", "Vannak", "Molika",
         "Sothea", "Kimheng", "Ratha", "Sreypov", "Panha", "Chantha", "Narith", "Sokunthea"]
FIRST_HIST_YEAR = 2012
LAST_HIST_YEAR = C.BASE_YEAR - 1


def mortality_table(level: float = 1.0) -> pd.Series:
    """Gompertz-Makeham q(x), ages 0-110, tuned so life expectancy at 55 is about 22 years."""
    ages = np.arange(0, 111)
    mu = 0.0006 + 0.00004 * np.exp(0.092 * ages)
    q = 1 - np.exp(-mu * level)
    q[-1] = 1.0
    return pd.Series(np.clip(q, 0, 1), index=ages, name="qx")


def true_exit_logit(age, service, sector_idx, fw_idx, year):
    """Data-generating separation hazard (non-retirement exits). Deliberately nonlinear."""
    x = -5.0 + 1.6 * np.exp(-service / 2.0)  # early-career attrition
    x += np.where((age >= 50) & (age < 55), 0.9, 0.0)  # voluntary early retirement window
    x += np.array([0.0, 0.25, 0.35])[sector_idx]
    x += np.array([0.3, 0.0, -0.1, 0.1])[fw_idx]
    # rising private-sector pull for young health and admin staff after 2018
    pull = np.clip(year - 2018, 0, None) * 0.06
    x += np.where((sector_idx > 0) & (age < 35), pull, 0.0)
    return x


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _initial_stock(rng, n):
    sector = rng.choice(3, n, p=list(C.BASE.sector_share.values()))
    fw = rng.choice(4, n, p=list(C.BASE.framework_share.values()))
    # aging workforce: bulge between 40 and 54
    age = np.clip(np.round(rng.triangular(19, 30, 60, n)), 19, 59).astype(int)
    entry_age = np.clip(np.round(rng.normal(np.array([27, 24, 22, 21])[fw], 2.5)), 18, age).astype(int)
    service = age - entry_age
    eligible60 = rng.random(n) < C.BASE.share_eligible_age60
    too_old = ~eligible60 & (age > 54)
    age = np.where(too_old, rng.integers(22, 55, n), age)
    service = np.minimum(service, age - 18)
    return dict(sector=sector, fw=fw, age=age, service=service, eligible60=eligible60)


def simulate_history(rng, n0: int = 170_000, growth: float = 0.012):
    q = mortality_table().to_numpy()
    s = _initial_stock(rng, n0)
    n = n0
    pid = np.arange(n)
    birth_year = FIRST_HIST_YEAR - s["age"]
    frames = []
    next_id = n
    for year in range(FIRST_HIST_YEAR, LAST_HIST_YEAR + 1):
        age, service, sector, fw = s["age"], s["service"], s["sector"], s["fw"]
        ret_age = np.where(s["eligible60"], 60, 55)
        retire = age >= ret_age
        die = rng.random(n) < q[np.minimum(age, 110)]
        sep = rng.random(n) < _sigmoid(true_exit_logit(age, service, sector, fw, year))
        exit_type = np.where(retire, "retirement", np.where(die, "death", np.where(sep, "separation", "")))
        # promotion D->C->B->A
        p_up = np.array([0.0, 0.020, 0.035, 0.030])[fw] * np.where(service >= 3, 1.0, 0.2)
        promoted = (rng.random(n) < p_up) & (exit_type == "")
        frames.append(pd.DataFrame({
            "person_id": pid, "year": year, "age": age, "service": service,
            "sector": np.array(C.SECTORS)[sector], "framework": np.array(C.FRAMEWORKS)[fw],
            "eligible60": s["eligible60"], "exit": exit_type != "", "exit_type": exit_type,
            "promoted": promoted,
        }))
        stay = exit_type == ""
        new_fw = np.where(promoted, fw - 1, fw)
        n_hire = int(round((~stay).sum() + n * growth))
        h = _initial_stock(rng, n_hire)
        h_fw = h["fw"]
        h_age = np.clip(np.round(rng.normal(np.array([27, 24, 22, 21])[h_fw], 2.5)), 18, 40).astype(int)
        s = {
            "sector": np.concatenate([sector[stay], h["sector"]]),
            "fw": np.concatenate([new_fw[stay], h_fw]),
            "age": np.concatenate([age[stay] + 1, h_age]),
            "service": np.concatenate([service[stay] + 1, np.zeros(n_hire, int)]),
            "eligible60": np.concatenate([s["eligible60"][stay], h["eligible60"]]),
        }
        pid = np.concatenate([pid[stay], np.arange(next_id, next_id + n_hire)])
        birth_year = np.concatenate([birth_year[stay], year + 1 - h_age])
        next_id += n_hire
        n = len(pid)
    stock = pd.DataFrame({
        "person_id": pid, "birth_year": birth_year, "age": s["age"], "service": s["service"],
        "sector": np.array(C.SECTORS)[s["sector"]], "framework": np.array(C.FRAMEWORKS)[s["fw"]],
        "eligible60": s["eligible60"],
    })
    return pd.concat(frames, ignore_index=True), stock


def assign_units(history: pd.DataFrame, stock: pd.DataFrame, seed: int = C.SEED + 2):
    """Give every person a province and ministry of posting, fixed over their career.

    Postings follow population (Phnom Penh x4 for general administration). Each province and
    ministry also gets a random hiring trend, so some hired more in recent years and others
    less, which gives units different age mixes: the pattern a retirement-wave warning looks
    for. The trends are synthetic, not estimates for real provinces. Uses its own random
    generator, so every other synthetic draw, and every published result, is unchanged.
    """
    rng = np.random.default_rng(seed)
    people = pd.concat([history[["person_id", "sector", "year", "service"]],
                        stock.assign(year=C.BASE_YEAR)[["person_id", "sector", "year", "service"]]])
    people = people.drop_duplicates("person_id")
    entry = ((people["year"] - people["service"]).to_numpy() - 2010) / 10  # decades from 2010

    def draw(base_w, trend, mask):
        logw = np.log(base_w)[None, :] + entry[mask, None] * trend[None, :]
        return np.argmax(logw + rng.gumbel(size=logw.shape), axis=1)  # categorical draw per person

    pop = np.array(PROVINCE_POP, float)
    prov_trend = rng.normal(0, 0.3, len(PROVINCES))
    province = np.empty(len(people), dtype=object)
    ministry = np.empty(len(people), dtype=object)
    for sector in C.SECTORS:
        m = (people["sector"] == sector).to_numpy()
        w = pop.copy()
        if sector == "general_admin":
            w[0] *= 4
        trend = prov_trend + rng.normal(0, 0.15, len(PROVINCES))
        province[m] = np.array(PROVINCES)[draw(w, trend, m)]
        names = MINISTRIES[sector]
        ministry[m] = np.array(names)[draw(np.ones(len(names)), rng.normal(0, 0.25, len(names)), m)]
    units = pd.DataFrame({"person_id": people["person_id"].to_numpy(), "province": province, "ministry": ministry})
    return history.merge(units, on="person_id", how="left"), stock.merge(units, on="person_id", how="left")


def basic_salary(framework, service):
    base = pd.Series(C.BASE.base_salary)[framework].to_numpy()
    return base * (1 + C.BASE.step_increment) ** np.asarray(service)


def hrmis_extract(rng, stock: pd.DataFrame):
    """Base-year staff records, with duplicates, unconfirmed-identity records and salary outliers injected."""
    n = len(stock)
    df = stock.copy()
    df["national_id"] = rng.choice(10**9, n, replace=False) + 10**9
    df["sex"] = rng.choice(["F", "M"], n, p=[0.45, 0.55])
    df["full_name"] = [f"{a} {b}" for a, b in zip(rng.choice(FAMILY, n), rng.choice(GIVEN, n))]
    df["birth_date"] = (pd.to_datetime(df["birth_year"].astype(str) + "-01-01")
                        + pd.to_timedelta(rng.integers(0, 365, n), unit="D")).dt.date
    rng.choice(PROVINCES, n)  # kept so later draws, and so all published results, stay unchanged
    prov_rng = np.random.default_rng(C.SEED + 1)
    pop = np.array(PROVINCE_POP, float)
    hq = pop.copy()
    hq[0] *= 4  # Phnom Penh weight for general administration
    df["province"] = ""
    for sector in C.SECTORS:
        m = (df["sector"] == sector).to_numpy()
        w = hq if sector == "general_admin" else pop
        df.loc[m, "province"] = prov_rng.choice(PROVINCES, m.sum(), p=w / w.sum())
    df["ministry"] = [rng.choice(MINISTRIES[s]) for s in df["sector"]]
    if "province" in stock:  # postings from assign_units; the draws above are kept for the same reason
        df["province"], df["ministry"] = stock["province"].to_numpy(), stock["ministry"].to_numpy()
    df["basic_salary"] = np.round(basic_salary(df["framework"], df["service"]), -3)
    df["allowance"] = np.round(df["basic_salary"] * df["sector"].map(C.BASE.allowance_rate)
                               * rng.normal(1, 0.08, n), -3)
    df["attendance_days_q"] = np.clip(np.round(rng.normal(60, 3, n)), 40, 66).astype(int)
    df["bank_account"] = rng.choice(10**11, n, replace=False)
    df["anomaly"] = "none"

    k_dup, k_unconf, k_out = int(n * 0.010), int(n * 0.015), int(n * 0.005)
    dup = df.sample(k_dup, random_state=1).copy()
    dup["person_id"] = np.arange(df.person_id.max() + 1, df.person_id.max() + 1 + k_dup)
    dup["province"] = rng.choice(PROVINCES, k_dup)  # re-registered in another province
    typo = rng.random(k_dup) < 0.6  # national ID re-keyed with one wrong digit
    dup.loc[typo, "national_id"] += rng.choice([1, 10, 100, 1000], typo.sum()) * rng.integers(1, 9, typo.sum())
    dup["bank_account"] = rng.choice(10**11, k_dup)
    dup["anomaly"] = "duplicate"

    unconf = df.sample(k_unconf, random_state=2).copy()
    unconf["person_id"] = np.arange(dup.person_id.max() + 1, dup.person_id.max() + 1 + k_unconf)
    unconf["national_id"] = rng.choice(10**9, k_unconf) + 3 * 10**9
    unconf["full_name"] = [f"{a} {b}" for a, b in zip(rng.choice(FAMILY, k_unconf), rng.choice(GIVEN, k_unconf))]
    unconf["attendance_days_q"] = rng.integers(0, 8, k_unconf)
    older = rng.random(k_unconf) < 0.5  # half are past retirement age
    unconf.loc[older, "age"] = rng.integers(56, 72, older.sum())
    unconf["birth_year"] = C.BASE_YEAR - unconf["age"]
    shared = rng.random(k_unconf) < 0.4  # shares a bank account with another record
    unconf.loc[shared, "bank_account"] = df["bank_account"].sample(shared.sum(), random_state=3).to_numpy()
    unconf["anomaly"] = "identity_unconfirmed"

    df = pd.concat([df, dup, unconf], ignore_index=True)
    out_idx = df[df.anomaly == "none"].sample(k_out, random_state=4).index
    df.loc[out_idx, "basic_salary"] *= rng.uniform(2.5, 8, k_out)
    df.loc[out_idx, "anomaly"] = "salary_outlier"
    df = df.sample(frac=1, random_state=5).reset_index(drop=True)
    df["record_id"] = np.arange(len(df))
    truth = df[["record_id", "anomaly"]]
    cols = ["record_id", "person_id", "national_id", "full_name", "birth_date", "birth_year", "age", "sex", "sector", "ministry",
            "province", "framework", "service", "eligible60", "basic_salary", "allowance",
            "attendance_days_q", "bank_account"]
    return df[cols], truth


def pensioner_register(rng):
    n = C.BASE.pensioners
    age = np.clip(np.round(55 + rng.gamma(2.2, 5.5, n)), 55, 100).astype(int)
    pension = np.round(C.BASE.avg_pension_monthly * rng.lognormal(0, 0.18, n)
                       / np.exp(0.18**2 / 2), -3)
    return pd.DataFrame({"pensioner_id": np.arange(n), "age": age, "monthly_pension": pension})


def macro_history(rng):
    """Stylised annual series 1995-2025. SOURCE: replace with NIS, NBC and IMF WEO data."""
    years = np.arange(1995, LAST_HIST_YEAR + 1)
    growth = np.full(len(years), 0.07)
    known = {2008: 0.067, 2009: 0.001, 2010: 0.060, 2020: -0.031, 2021: 0.030,
             2022: 0.052, 2023: 0.050, 2024: 0.060, 2025: 0.052}
    growth = growth + rng.normal(0, 0.012, len(years))
    infl = 0.035 + rng.normal(0, 0.012, len(years))
    known_i = {2008: 0.250, 2009: -0.007, 2011: 0.055, 2020: 0.029, 2021: 0.029, 2022: 0.053,
               2023: 0.021, 2024: 0.008, 2025: 0.025}
    for y, v in known.items():
        growth[years == y] = v
    for y, v in known_i.items():
        infl[years == y] = v
    real_idx = np.cumprod(1 + growth)
    price_idx = np.cumprod(1 + infl)
    nominal = real_idx * price_idx
    gdp = C.BASE.gdp / (nominal[-1] * (1 + 0.052 + 0.025)) * nominal
    rev_share = np.clip(0.11 + 0.0045 * (years - 1995) + rng.normal(0, 0.006, len(years)), 0.1, 0.23)
    rev_share[years == 2020] -= 0.02
    return pd.DataFrame({"year": years, "real_growth": growth, "inflation": infl,
                         "gdp_nominal": gdp, "revenue_share_gdp": rev_share,
                         "revenue": gdp * rev_share})


def generate(out_dir: Path, seed: int = C.SEED) -> dict:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, stock = simulate_history(rng)
    history, stock = assign_units(history, stock)
    hrmis, truth = hrmis_extract(rng, stock)
    pens = pensioner_register(rng)
    macro = macro_history(rng)
    history.to_parquet(out_dir / "staff_history.parquet", index=False)
    hrmis.to_csv(out_dir / "hrmis_2026.csv", index=False)
    truth.to_csv(out_dir / "hrmis_2026_truth.csv", index=False)
    pens.to_csv(out_dir / "pensioners_2026.csv", index=False)
    macro.to_csv(out_dir / "macro_history.csv", index=False)
    mortality_table().to_frame().rename_axis("age").to_csv(out_dir / "mortality.csv")
    return {"history": history, "hrmis": hrmis, "truth": truth, "pensioners": pens, "macro": macro}
