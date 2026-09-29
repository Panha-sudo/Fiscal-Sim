"""M1 Data quality: flag duplicate, inactive and anomalous payroll records.

Two detectors, as in proposal Table 4:
  - baseline: rule-based checks only (exact national-ID duplicates, age past
    retirement, zero attendance, salary outside the pay scale)
  - model: rule-based record matching + Isolation Forest on payroll features

Records flagged by the chosen detector are removed before the base year
workforce is built, so data quality feeds straight into the fiscal baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C
from .synthetic import basic_salary


def _features(df: pd.DataFrame) -> pd.DataFrame:
    expected = basic_salary(df["framework"], df["service"].clip(lower=0))
    acct_count = df.groupby("bank_account")["record_id"].transform("count")
    return pd.DataFrame({
        "salary_ratio": np.log(df["basic_salary"] / expected),
        "allowance_ratio": df["allowance"] / df["basic_salary"],
        "attendance": df["attendance_days_q"],
        "age_over_ret": df["age"] - np.where(df["eligible60"], 60, 55),
        "service_gap": df["age"] - df["service"],
        "shared_account": (acct_count > 1).astype(int),
    }, index=df.index)


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline: deterministic checks a payroll clerk would run."""
    ret_age = np.where(df["eligible60"], 60, 55)
    expected = basic_salary(df["framework"], df["service"].clip(lower=0))
    flags = pd.DataFrame(index=df.index)
    # the later registration (higher person_id) of a pair is the one flagged
    flags["dup_national_id"] = df.sort_values("person_id").duplicated("national_id", keep="first").reindex(df.index)
    flags["past_retirement"] = df["age"] > ret_age
    flags["no_attendance"] = df["attendance_days_q"] == 0
    flags["off_scale_salary"] = (df["basic_salary"] > expected * 2.0)
    return flags


def record_matching(df: pd.DataFrame) -> pd.Series:
    """Duplicate match that survives a mistyped national ID: same name, birth date and sex."""
    d = df.sort_values("person_id")
    hit = d.duplicated(["full_name", "birth_date", "sex"], keep="first") | d.duplicated("national_id", keep="first")
    return hit.reindex(df.index)


def isolation_forest(df: pd.DataFrame, contamination: float = 0.02, seed: int = C.SEED):
    X = _features(df)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(X)
    score = -model.score_samples(X)
    return pd.Series(model.predict(X) == -1, index=df.index), pd.Series(score, index=df.index), model, X


def _metrics(pred: pd.Series, truth: pd.Series) -> dict:
    y = truth != "none"
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"flagged": int(pred.sum()), "true_positive": tp, "false_positive": fp, "missed": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3)}


def run(hrmis: pd.DataFrame, truth: pd.DataFrame | None = None) -> dict:
    rules = rule_flags(hrmis)
    baseline_flag = rules.any(axis=1)
    iso_flag, iso_score, model, X = isolation_forest(hrmis)
    model_flag = record_matching(hrmis) | iso_flag | rules[["past_retirement", "no_attendance", "off_scale_salary"]].any(axis=1)

    out = hrmis.copy()
    out["rule_flag"] = baseline_flag
    out["iforest_flag"] = iso_flag
    out["anomaly_score"] = iso_score.round(4)
    out["flag"] = model_flag
    result = {"records": out, "clean": hrmis[~model_flag].copy(), "model": model, "features": X}

    if truth is not None:
        t = hrmis[["record_id"]].merge(truth, on="record_id", how="left")["anomaly"]
        t.index = hrmis.index
        result["evaluation"] = pd.DataFrame({
            "rule_based_only": _metrics(baseline_flag, t),
            "isolation_forest_only": _metrics(iso_flag, t),
            "matching_plus_iforest": _metrics(model_flag, t),
        }).T
        by_type = pd.DataFrame({"anomaly": t, "rule": baseline_flag, "model": model_flag})
        result["recall_by_type"] = (by_type[by_type.anomaly != "none"]
                                    .groupby("anomaly")[["rule", "model"]].mean().round(3))
        wage = (hrmis["basic_salary"] + hrmis["allowance"]) * 12
        result["payroll_leakage_riel"] = float(wage[t != "none"].sum())
        result["payroll_removed_riel"] = float(wage[model_flag].sum())
    return result
