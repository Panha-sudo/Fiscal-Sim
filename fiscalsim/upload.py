"""Read and check a user's own payroll or pensioner file (CSV or Excel) before it replaces the base year.

Required columns are the ones the model cannot do without. Optional ones get
a stated default so the M1 checks still run; the defaults weaken the check
that needs them (for example, no names means duplicates are matched on
national ID only).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config as C

REQUIRED = {
    "age": "age in years in the base year",
    "sector": "education, health or general_admin",
    "framework": "A, B, C or D",
    "service": "completed years of service",
    "basic_salary": "monthly basic salary in riel",
}
OPTIONAL = {
    "allowance": ("monthly allowances in riel", "basic_salary x sector allowance rate"),
    "ministry": ("ministry or institution", "'Unknown'"),
    "province": ("province of posting", "'Unknown'"),
    "eligible60": ("True if the post allows work to 60", "False"),
    "national_id": ("national ID number", "a unique number per row"),
    "full_name": ("name, used to find duplicates", "a unique placeholder per row"),
    "birth_date": ("date of birth, used to find duplicates", "1 January of the birth year"),
    "sex": ("F or M", "'U'"),
    "attendance_days_q": ("days present last quarter", "60"),
    "bank_account": ("salary account, used to find shared accounts", "a unique number per row"),
}
PENSIONER_REQUIRED = {"age": "age in years", "monthly_pension": "monthly pension in riel"}


HISTORY_REQUIRED = {
    "person_id": "the same ID for a person in every year",
    "year": "year of the snapshot",
    "age": "age in years that year",
    "service": "completed years of service that year",
    "sector": "education, health or general_admin",
    "framework": "A, B, C or D",
}
HISTORY_OPTIONAL = {
    "exit_type": ("why the person left after this year: retirement, death, separation, or blank if they stayed",
                  "worked out from the last year each person appears: past retirement age = retirement, "
                  "otherwise separation"),
    "promoted": ("True if promoted to a higher framework this year", "worked out from framework changes"),
    "eligible60": ("True if the post allows work to 60", "False"),
    "province": ("province of posting", "'Unknown'"),
    "ministry": ("ministry or institution", "'Unknown'"),
}
MACRO_REQUIRED = {
    "year": "calendar year, one row per year with no gaps",
    "real_growth": "real GDP growth (0.052 or 5.2 for 5.2%)",
    "inflation": "CPI inflation (0.025 or 2.5 for 2.5%)",
    "revenue_share_gdp": "government revenue as a share of GDP (0.21 or 21 for 21%)",
}
MACRO_OPTIONAL = {"gdp_nominal": ("nominal GDP in riel (shown only)", "not used")}
EXIT_TYPES = ("retirement", "death", "separation")
MIN_HISTORY_YEARS, MIN_MACRO_YEARS = 6, 15


def read_table(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if name.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def prepare_hrmis(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Returns (records ready for M1, errors, notes). Records is None when there are errors."""
    df = _normalise_columns(df)
    errors, notes = [], []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return None, errors, notes
    _check_codes(df, errors)
    _numeric(df, ("age", "service", "basic_salary"), errors)
    if errors:
        return None, errors, notes

    n = len(df)
    rng = np.random.default_rng(0)
    df["age"] = df["age"].round().astype(int)
    df["service"] = df["service"].round().clip(lower=0).astype(int)
    if "allowance" not in df:
        df["allowance"] = df["basic_salary"] * df["sector"].map(C.BASE.allowance_rate)
    defaults = {
        "ministry": "Unknown", "province": "Unknown", "eligible60": False, "sex": "U",
        "attendance_days_q": 60,
        "national_id": np.arange(n) + 9 * 10**9,
        "full_name": [f"row-{i}" for i in range(n)],
        "bank_account": rng.choice(10**12, n, replace=False),
    }
    for col, val in defaults.items():
        if col not in df:
            df[col] = val
            notes.append(f"No '{col}' column: used {OPTIONAL[col][1]}.")
    if "birth_date" not in df:
        df["birth_date"] = (C.BASE_YEAR - df["age"]).astype(str) + "-01-01"
        notes.append(f"No 'birth_date' column: used {OPTIONAL['birth_date'][1]}.")
    df["eligible60"] = df["eligible60"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    df["birth_year"] = C.BASE_YEAR - df["age"]
    df["record_id"] = np.arange(n)
    df["person_id"] = df["person_id"] if "person_id" in df else np.arange(n)
    return df, errors, notes


def prepare_pensioners(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    df = _normalise_columns(df)
    missing = [c for c in PENSIONER_REQUIRED if c not in df.columns]
    if missing:
        return None, ["Missing required columns: " + ", ".join(missing)]
    out = pd.DataFrame({"age": pd.to_numeric(df["age"], errors="coerce"),
                        "monthly_pension": pd.to_numeric(df["monthly_pension"], errors="coerce")}).dropna()
    out["age"] = out["age"].round().astype(int).clip(40, 110)
    return out, []


def _check_codes(df: pd.DataFrame, errors: list[str]) -> None:
    df["sector"] = df["sector"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    df["framework"] = df["framework"].astype(str).str.strip().str.upper()
    bad_sector = sorted(set(df["sector"]) - set(C.SECTORS))
    bad_fw = sorted(set(df["framework"]) - set(C.FRAMEWORKS))
    if bad_sector:
        errors.append(f"Unknown sector values {bad_sector[:5]}. Use one of {list(C.SECTORS)}.")
    if bad_fw:
        errors.append(f"Unknown framework values {bad_fw[:5]}. Use one of {list(C.FRAMEWORKS)}.")


def _numeric(df: pd.DataFrame, cols, errors: list[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n = int(df[c].isna().sum())
        if n:
            errors.append(f"{n:,} rows have a missing or non-numeric '{c}'.")


def _truthy(col: pd.Series) -> pd.Series:
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def prepare_history(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Yearly staff snapshots -> the person-year panel M2, the wave backtest and the wage bill forecast use.

    Returns (history, errors, notes); history is None when there are errors.
    """
    df = _normalise_columns(df)
    errors, notes = [], []
    missing = [c for c in HISTORY_REQUIRED if c not in df.columns]
    if missing:
        return None, ["Missing required columns: " + ", ".join(missing)], notes
    _check_codes(df, errors)
    _numeric(df, ("year", "age", "service"), errors)
    if errors:
        return None, errors, notes
    df["person_id"] = df["person_id"].astype(str).str.strip()
    for c in ("year", "age", "service"):
        df[c] = df[c].round().astype(int)
    dup = int(df.duplicated(["person_id", "year"]).sum())
    if dup:
        errors.append(f"{dup:,} rows repeat a person_id in the same year. Each person should appear once a year.")
    years = np.sort(df["year"].unique())
    if len(years) < MIN_HISTORY_YEARS:
        errors.append(f"The file covers {len(years)} years ({years.min()}-{years.max()}). The exit model and its "
                      f"test need at least {MIN_HISTORY_YEARS} consecutive years.")
    elif (np.diff(years) != 1).any():
        errors.append(f"Some years are missing between {years.min()} and {years.max()}. Snapshots must be yearly.")
    if errors:
        return None, errors, notes

    if "eligible60" in df:
        df["eligible60"] = _truthy(df["eligible60"])
    else:
        df["eligible60"] = False
        notes.append("No 'eligible60' column: everyone retires at the standard age.")
    for col in ("province", "ministry"):
        if col not in df:
            df[col] = "Unknown"
            notes.append(f"No '{col}' column: used 'Unknown', so the retirement-wave test and the wage bill "
                         f"forecast have no {col} breakdown.")
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df = df.sort_values(["person_id", "year"]).reset_index(drop=True)
    nxt = df.groupby("person_id")["year"].shift(-1)
    if "exit_type" in df:
        et = df["exit_type"].fillna("").astype(str).str.strip().str.lower().replace({"none": "", "nan": "", "stay": ""})
        bad = sorted(set(et) - {"", *EXIT_TYPES})
        if bad:
            return None, [f"Unknown exit_type values {bad[:5]}. Use {', '.join(EXIT_TYPES)} or leave blank."], notes
        df["exit_type"] = et
    else:
        last = nxt.isna() & (df["year"] < years.max())
        ret_age = np.where(df["eligible60"], 60, 55)
        df["exit_type"] = np.where(last, np.where(df["age"] >= ret_age, "retirement", "separation"), "")
        notes.append("No 'exit_type' column: a person's last year before the final snapshot counts as an exit, a "
                     "retirement if they were at retirement age, otherwise a separation. Deaths cannot be told "
                     "apart from resignations this way.")
    if "promoted" in df:
        df["promoted"] = _truthy(df["promoted"])
    else:
        rank = df["framework"].map({f: i for i, f in enumerate(C.FRAMEWORKS)})
        nxt_rank = rank.groupby(df["person_id"]).shift(-1)
        df["promoted"] = (nxt == df["year"] + 1) & (nxt_rank < rank)
        notes.append("No 'promoted' column: a move to a more senior framework the next year counts as a promotion.")
    df["exit"] = df["exit_type"] != ""
    df["service"] = df["service"].clip(lower=0)
    rate = df.loc[df["exit_type"] == "separation"].shape[0] / max(len(df), 1)
    notes.append(f"{len(df):,} person-years, {df['person_id'].nunique():,} people, {years.min()}-{years.max()}; "
                 f"{rate:.1%} of person-years end in a separation.")
    cols = ["person_id", "year", "age", "service", "sector", "framework", "eligible60", "exit", "exit_type",
            "promoted", "province", "ministry"]
    return df[cols], errors, notes


def prepare_macro(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Annual growth, inflation and revenue share -> the M5 input. Percentages are converted to shares."""
    df = _normalise_columns(df)
    errors, notes = [], []
    missing = [c for c in MACRO_REQUIRED if c not in df.columns]
    if missing:
        return None, ["Missing required columns: " + ", ".join(missing)], notes
    _numeric(df, MACRO_REQUIRED, errors)
    if errors:
        return None, errors, notes
    df = df.sort_values("year").reset_index(drop=True)
    df["year"] = df["year"].round().astype(int)
    if df["year"].duplicated().any() or (np.diff(df["year"]) != 1).any():
        errors.append("Years must be consecutive, one row per year, with no gaps or repeats.")
    if len(df) < MIN_MACRO_YEARS:
        errors.append(f"The file has {len(df)} years. The forecast test needs at least {MIN_MACRO_YEARS}.")
    if errors:
        return None, errors, notes
    for c in ("real_growth", "inflation", "revenue_share_gdp"):
        if df[c].abs().median() > 1:
            df[c] = df[c] / 100
            notes.append(f"'{c}' looks like percentages, so it was divided by 100.")
    if not df["revenue_share_gdp"].between(0.02, 0.6).all():
        errors.append("Some 'revenue_share_gdp' values are outside 2% to 60% of GDP. Check the units.")
    if df[["real_growth", "inflation"]].abs().gt(0.5).any().any():
        errors.append("Some growth or inflation values are above 50%. Check the units.")
    if errors:
        return None, errors, notes
    last = int(df["year"].max())
    if last < C.BASE_YEAR - 1:
        notes.append(f"The data end in {last}. The simulation starts in {C.BASE_YEAR}, so the forecast for the years "
                     f"after {last} is used from {C.BASE_YEAR} on.")
    notes.append(f"{len(df)} years, {int(df['year'].min())}-{last}.")
    keep = ["year", *[c for c in MACRO_REQUIRED if c != "year"]] + [c for c in MACRO_OPTIONAL if c in df]
    return df[keep], errors, notes


def column_table(required: dict, optional: dict) -> pd.DataFrame:
    """Columns a file can have, for the help table under each upload."""
    return pd.DataFrame([(c, d, "") for c, d in required.items()] + [(c, d, dflt) for c, (d, dflt) in optional.items()],
                        columns=["column", "meaning", "default if missing"])


def example_history(n0: int = 300, seed: int = C.SEED) -> pd.DataFrame:
    """A small synthetic staff history in the upload format (exit and promotion columns included)."""
    from . import synthetic
    h, stock = synthetic.simulate_history(np.random.default_rng(seed), n0=n0)
    h, _ = synthetic.assign_units(h, stock)
    return h[["person_id", "year", "age", "service", "sector", "framework", "eligible60", "exit_type", "promoted",
              "province", "ministry"]]


def template_csv(sample: pd.DataFrame | None = None) -> bytes:
    """A small example file with every column the upload understands."""
    cols = list(REQUIRED) + list(OPTIONAL)
    if sample is not None:
        return sample[[c for c in cols if c in sample]].head(50).to_csv(index=False).encode()
    return pd.DataFrame(columns=cols).to_csv(index=False).encode()
