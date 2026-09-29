"""Ministry and province breakdowns.

Base year: exact aggregates of the cleaned payroll, plus what M1 flagged, by
ministry and province.

Projections: the engine projects by sector, so a ministry's or province's
future wage bill is its base-year share of its sector's pay times the
sector's projected wage bill. This assumes each unit keeps its share of its
sector, which is stated wherever the numbers are shown.
"""
from __future__ import annotations

import pandas as pd


def base_table(records: pd.DataFrame) -> pd.DataFrame:
    """Records from M1 (with its `flag` column) -> one row per sector x ministry x province."""
    r = records.assign(annual_pay=(records["basic_salary"] + records["allowance"]) * 12)
    clean, flagged = r[~r["flag"]], r[r["flag"]]
    keys = ["sector", "ministry", "province"]
    g = clean.groupby(keys).agg(headcount=("record_id", "size"), wage_bill=("annual_pay", "sum"),
                                mean_age=("age", "mean"), aged_50_plus=("age", lambda a: int((a >= 50).sum())))
    f = flagged.groupby(keys).agg(flagged_records=("record_id", "size"), flagged_payroll=("annual_pay", "sum"))
    return g.join(f, how="outer").fillna(0).reset_index()


def by(base: pd.DataFrame, level: str) -> pd.DataFrame:
    """Aggregate the base table to 'ministry' or 'province'."""
    agg = base.groupby(level).agg(headcount=("headcount", "sum"), wage_bill=("wage_bill", "sum"),
                                  aged_50_plus=("aged_50_plus", "sum"),
                                  flagged_records=("flagged_records", "sum"),
                                  flagged_payroll=("flagged_payroll", "sum"))
    w = base.assign(age_x=base["mean_age"] * base["headcount"]).groupby(level)["age_x"].sum()
    agg["mean_age"] = w / agg["headcount"].where(agg["headcount"] > 0)
    agg["flagged_share_of_payroll"] = agg["flagged_payroll"] / (agg["wage_bill"] + agg["flagged_payroll"])
    return agg.sort_values("wage_bill", ascending=False)


def projected(base: pd.DataFrame, sector_wb: pd.DataFrame, level: str, year: int) -> pd.Series:
    """Projected wage bill in `year` for each ministry or province, from sector totals ([year x sector])."""
    share = base.groupby(["sector", level])["wage_bill"].sum()
    share = share / share.groupby(level="sector").transform("sum")
    sector_total = sector_wb.loc[year]
    vals = share * sector_total.reindex(share.index.get_level_values("sector")).to_numpy()
    return vals.groupby(level=level).sum().sort_values(ascending=False)
