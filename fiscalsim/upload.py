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


def read_table(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
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
    df["sector"] = df["sector"].astype(str).str.strip().str.lower().str.replace(" ", "_")
    df["framework"] = df["framework"].astype(str).str.strip().str.upper()
    bad_sector = sorted(set(df["sector"]) - set(C.SECTORS))
    bad_fw = sorted(set(df["framework"]) - set(C.FRAMEWORKS))
    if bad_sector:
        errors.append(f"Unknown sector values {bad_sector[:5]}. Use one of {list(C.SECTORS)}.")
    if bad_fw:
        errors.append(f"Unknown framework values {bad_fw[:5]}. Use one of {list(C.FRAMEWORKS)}.")
    for c in ("age", "service", "basic_salary"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n = int(df[c].isna().sum())
        if n:
            errors.append(f"{n:,} rows have a missing or non-numeric '{c}'.")
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


def template_csv(sample: pd.DataFrame | None = None) -> bytes:
    """A small example file with every column the upload understands."""
    cols = list(REQUIRED) + list(OPTIONAL)
    if sample is not None:
        return sample[[c for c in cols if c in sample]].head(50).to_csv(index=False).encode()
    return pd.DataFrame(columns=cols).to_csv(index=False).encode()
