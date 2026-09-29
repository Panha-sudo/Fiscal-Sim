"""Test the scenario assistant on fixed questions in English and Khmer.

QUESTIONS (development set) were in view while the keyword rules were written, so the keyword baseline is tuned
to them. HELD_OUT was written after the rules were frozen and never used to tune anything: it is the fair test.
Each question's gold answer names the tool, the settings and the sources an answer must cite. Scores:
    right tool       the assistant chose the expected tool
    right settings   ... and ran exactly the expected scenario, target or unit
    cites correctly  every expected source is cited, and nothing outside the tool's sources is
    model numbers    every number in the final answer is a model value
    AI text used     share of answers where Gemma's own wording passed the checks (Gemma only)

    python -m fiscalsim.assistant_eval           # keyword baseline, plus Gemma when GEMINI_API_KEY is set
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd

from . import assistant as A

# (question, language, [alternatives: (tool, settings, sources that must be cited)])
QUESTIONS = [
    ("What if the retirement age is raised to 62?", "en", [("run_reform", {"retirement_age": 62}, ["A6"])]),
    ("What happens if we move to the accrual pension formula at 2.5% per year of service?", "en",
     [("run_reform", {"pension_formula": "accrual", "accrual_rate_percent": 2.5}, ["A7"])]),
    ("If contributions go up by 3 percentage points on each side, when does the pension fund run out?", "en",
     [("run_reform", {"contribution_rise_points": 3}, ["A8"])]),
    ("Index pay to inflation instead of 6% raises. What would the wage bill be in 2046?", "en",
     [("run_reform", {"pay_rule": "inflation"}, ["A14"]), ("compare_scenarios", {"codes": ["S1"]}, ["S1"])]),
    ("Compare the status quo with the combined reform.", "en", [("compare_scenarios", {"codes": ["S0", "S5"]}, ["S0", "S5"])]),
    ("Which of the seven scenarios keeps pensions most generous?", "en",
     [("compare_scenarios", {"codes": A.CODES}, ["D1"])]),
    ("What is the cheapest reform that keeps an average replacement rate of 58%?", "en",
     [("find_best_reform", {"goal": "cheapest_for_replacement", "target": 58}, ["D6"])]),
    ("What is the most generous pension package that costs no more than 3.5% of GDP on average?", "en",
     [("find_best_reform", {"goal": "most_adequate_within_cost", "target": 3.5}, ["D6"])]),
    ("Which provinces will lose the most staff over the next 10 years?", "en",
     [("staff_exits", {"level": "province", "years_ahead": 10}, ["D7"])]),
    ("How many staff are expected to leave Takeo in the next 5 years?", "en",
     [("staff_exits", {"level": "province", "unit": "Takeo", "years_ahead": 5}, ["D7"])]),
    ("Which ministries face a retirement wave?", "en", [("staff_exits", {"level": "ministry", "years_ahead": 10}, ["D7"])]),
    ("What is the current pension formula?", "en", [("explain_assumption", {}, ["A1"])]),
    ("What contribution rates does the model assume?", "en", [("explain_assumption", {}, ["A4"])]),
    ("Where does the headcount figure come from?", "en", [("explain_assumption", {}, ["P1"])]),
    ("How is total cost defined in this model?", "en", [("explain_assumption", {}, ["A9"])]),
    ("What is the minimum pension?", "en", [("explain_assumption", {}, ["A2"])]),
    ("Retire at 60 with a 2% accrual formula: what replacement rate do new retirees get?", "en",
     [("run_reform", {"retirement_age": 60, "pension_formula": "accrual", "accrual_rate_percent": 2}, ["A6", "A7"])]),
    ("Take the combined reform but raise the retirement age to 62 instead of 60.", "en",
     [("run_reform", {"base_scenario": "S5", "retirement_age": 62}, ["S5", "A6"])]),
    ("What if we only replace leavers outside education and health?", "en",
     [("run_reform", {"restrain_hiring": True}, ["A11"]), ("compare_scenarios", {"codes": ["S6"]}, ["S6"])]),
    ("Give teachers and health workers bigger raises: what does that do to cost in 2046?", "en",
     [("run_reform", {"pay_rule": "targeted"}, ["A14"]), ("compare_scenarios", {"codes": ["S2"]}, ["S2"])]),
    ("Is the pension fund at risk under the status quo?", "en",
     [("run_reform", {}, ["D2"]), ("compare_scenarios", {"codes": ["S0"]}, ["D2"])]),
    ("What's the weather in Phnom Penh today?", "en", [("none", {}, [])]),
    ("បើដំឡើងអាយុចូលនិវត្តន៍ដល់ ៦២ ឆ្នាំ តើមានអ្វីកើតឡើង?", "km", [("run_reform", {"retirement_age": 62}, ["A6"])]),
    ("ប្រៀបធៀបស្ថានភាពបច្ចុប្បន្ន (S0) និងកំណែទម្រង់រួម (S5)", "km",
     [("compare_scenarios", {"codes": ["S0", "S5"]}, ["S0", "S5"])]),
    ("តើកំណែទម្រង់ណាដែលថោកបំផុត ដែលរក្សាអត្រាជំនួសមធ្យមបាន ៥៨%?", "km",
     [("find_best_reform", {"goal": "cheapest_for_replacement", "target": 58}, ["D6"])]),
    ("ខេត្តណាខ្លះនឹងបាត់បង់មន្ត្រីច្រើនជាងគេ ក្នុងរយៈពេល ១០ ឆ្នាំខាងមុខ?", "km",
     [("staff_exits", {"level": "province", "years_ahead": 10}, ["D7"])]),
    ("តើរូបមន្តសោធនបច្ចុប្បន្នគឺយ៉ាងដូចម្តេច?", "km", [("explain_assumption", {}, ["A1"])]),
    ("តើម៉ូដែលសន្មតអត្រាភាគទានប៉ុន្មាន?", "km", [("explain_assumption", {}, ["A4"])]),
    ("បើភាគទានកើន ៣ ពិន្ទុម្ខាងៗ តើមូលនិធិសោធនអស់ទុននៅឆ្នាំណា?", "km",
     [("run_reform", {"contribution_rise_points": 3}, ["A8"])]),
    ("ចូលនិវត្តន៍នៅអាយុ ៦០ ជាមួយរូបមន្តសោធនតាមឆ្នាំសេវា ២% តើអត្រាជំនួសប៉ុន្មាន?", "km",
     [("run_reform", {"retirement_age": 60, "pension_formula": "accrual", "accrual_rate_percent": 2}, ["A6", "A7"])]),
    ("បើប្រាក់បៀវត្សកើនតាមអតិផរណា តើថ្លៃដើមសរុបឆ្នាំ ២០៤៦ ប៉ុន្មាន?", "km",
     [("run_reform", {"pay_rule": "inflation"}, ["A14"]), ("compare_scenarios", {"codes": ["S1"]}, ["S1"])]),
    ("តើអាកាសធាតុនៅភ្នំពេញថ្ងៃនេះយ៉ាងដូចម្តេច?", "km", [("none", {}, [])]),
]

HELD_OUT = [
    ("Suppose civil servants keep working until 62. How does that change the budget?", "en",
     [("run_reform", {"retirement_age": 62}, ["A6"])]),
    ("Could we lift the pension age to sixty?", "en", [("run_reform", {"retirement_age": 60}, ["A6"])]),
    ("What would a career-average pension paying 1.75% for each year worked cost?", "en",
     [("run_reform", {"pension_formula": "accrual", "accrual_rate_percent": 1.75}, ["A7"])]),
    ("If staff and the state each paid two points more into the fund, how long would it last?", "en",
     [("run_reform", {"contribution_rise_points": 2}, ["A8"])]),
    ("Let salaries just keep pace with prices. What happens to spending?", "en",
     [("run_reform", {"pay_rule": "inflation"}, ["A14"]), ("compare_scenarios", {"codes": ["S1"]}, ["S1"])]),
    ("How does S3 stack up against S4?", "en", [("compare_scenarios", {"codes": ["S3", "S4"]}, ["S3", "S4"])]),
    ("Out of all the policy options, which one is cheapest in 2046?", "en",
     [("compare_scenarios", {"codes": A.CODES}, ["A9"])]),
    ("I want pensions of at least 55% of final pay. What's the least expensive way to get there?", "en",
     [("find_best_reform", {"goal": "cheapest_for_replacement", "target": 55}, ["D6"])]),
    ("Where will we run short of staff because of retirements?", "en",
     [("staff_exits", {"level": "province", "years_ahead": 10}, ["D7"])]),
    ("What share of Kampot's civil servants will be gone within 15 years?", "en",
     [("staff_exits", {"level": "province", "unit": "Kampot", "years_ahead": 15}, ["D7"])]),
    ("Does the model count allowances when working out pensions?", "en", [("explain_assumption", {}, ["A13"])]),
    ("What return does the reserve fund earn in the model?", "en", [("explain_assumption", {}, ["A12"])]),
    ("Why do you assume people retire at 55?", "en", [("explain_assumption", {}, ["A6"])]),
    ("Who won the football match last night?", "en", [("none", {}, [])]),
    ("សន្មតថាមន្ត្រីរាជការធ្វើការរហូតដល់អាយុ ៦៣ ឆ្នាំ តើថវិកាផ្លាស់ប្តូរយ៉ាងណា?", "km",
     [("run_reform", {"retirement_age": 63}, ["A6"])]),
    ("តើ S3 និង S4 មួយណាប្រសើរជាង?", "km", [("compare_scenarios", {"codes": ["S3", "S4"]}, ["S3", "S4"])]),
    ("ចង់បានសោធនយ៉ាងហោចណាស់ ៥៥% នៃប្រាក់បៀវត្សចុងក្រោយ តើវិធីណាចំណាយតិចបំផុត?", "km",
     [("find_best_reform", {"goal": "cheapest_for_replacement", "target": 55}, ["D6"])]),
    ("តើក្រសួងណាខ្លះនឹងមានមន្ត្រីចូលនិវត្តន៍ច្រើន?", "km",
     [("staff_exits", {"level": "ministry", "years_ahead": 10}, ["D7"])]),
    ("តើមូលនិធិបម្រុងទទួលបានផលចំណេញប៉ុន្មានក្នុងម៉ូដែល?", "km", [("explain_assumption", {}, ["A12"])]),
    ("បើរដ្ឋ និងមន្ត្រីបង់ភាគទានបន្ថែម ២ ពិន្ទុម្ខាងៗ តើមូលនិធិនៅបានដល់ពេលណា?", "km",
     [("run_reform", {"contribution_rise_points": 2}, ["A8"])]),
]
SETS = {"development": QUESTIONS, "held_out": HELD_OUT}
METRICS = ("right_tool", "right_settings", "cites_correctly", "model_numbers")


def same_settings(tool: str, got: dict, gold: dict) -> bool:
    if tool == "run_reform":
        return A.scenario_signature(A.scenario_from(A.clean_args(tool, gold))) == \
            A.scenario_signature(A.scenario_from(got)) and got.get("base_scenario", "S0") == gold.get("base_scenario", "S0")
    if tool == "compare_scenarios":
        return set(got.get("codes", [])) == set(gold["codes"]) or (len(gold["codes"]) == 1 and gold["codes"][0] in got["codes"])
    if tool == "find_best_reform":
        return got["goal"] == gold["goal"] and abs(got["target"] - gold["target"]) < 0.01 and \
            got["include_pay_levers"] == gold.get("include_pay_levers", False)
    if tool == "staff_exits":
        return got["level"] == gold["level"] and got["years_ahead"] == gold["years_ahead"] and \
            (got.get("unit") or "").lower() == (gold.get("unit") or "").lower()
    return True  # explain_assumption (judged by its citations) and none


def score(ans: A.Answer, alts: list) -> dict:
    tool_ok = any(ans.tool == t for t, _, _ in alts)
    match = next(((t, s, c) for t, s, c in alts if t == ans.tool and same_settings(t, ans.settings, s)), None)
    cites = (match or next((a for a in alts if a[0] == ans.tool), alts[0]))[2]
    valid = all(i in ans.result.sources for i in ans.cited)
    ok, _ = A.check(ans.text, ans.result, ans.question)
    return {"right_tool": tool_ok, "right_settings": match is not None,
            "cites_correctly": tool_ok and all(i in ans.cited for i in cites) and valid,
            "model_numbers": ok or ans.tool == "none"}


def evaluate(env: A.Env, client=None, pause: float = 0.0, progress=None) -> tuple[pd.DataFrame, dict]:
    rows = []
    items = [(name, q) for name, qs in SETS.items() for q in qs]
    for i, (qset, (q, lang, alts)) in enumerate(items):
        ans = A.answer(q, env, client)
        rows.append({"set": qset, "question": q, "language": lang, "tool": ans.tool, "settings": ans.settings,
                     **score(ans, alts), "ai_text_used": ans.grounded_raw, "note": ans.fallback or ans.error, "cited": ans.cited,
                     "seconds": round(ans.seconds, 2), "answer": ans.text})
        if progress:
            progress((i + 1) / len(items))
        if pause and client is not None:
            time.sleep(pause)
    df = pd.DataFrame(rows)
    return df, summary(df)


def summary(df: pd.DataFrame) -> dict:
    out = {}
    for qset in SETS:
        for lang in ("all", "en", "km"):
            d = df[(df.set == qset) & ((df.language == lang) | (lang == "all"))]
            row = {k: float(d[k].mean()) for k in METRICS}
            used = d["ai_text_used"].dropna()
            row["ai_text_used"] = float(used.astype(bool).mean()) if len(used) else None
            row["questions"] = len(d)
            out[(qset, lang)] = row
    return out


def summary_table(results: dict) -> pd.DataFrame:
    """One row per mode, question set and language."""
    return pd.DataFrame([{"mode": mode, "set": qset, "language": lang, **v}
                         for mode, s in results.items() for (qset, lang), v in s.items()])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parent.parent / "outputs" / "dashboard_bundle.pkl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs"))
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between Gemma questions (free-tier rate limits)")
    a = ap.parse_args(argv)
    with open(a.bundle, "rb") as f:
        env = A.env_from_bundle(pickle.load(f))
    out = Path(a.out)
    results = {}
    df, results["keywords (no AI)"] = evaluate(env)
    df.to_csv(out / "assistant_eval_keywords.csv", index=False)
    client = A.client_from_settings()
    if client is not None:
        dg, results[f"Gemma ({client.model})"] = evaluate(env, client, pause=a.pause)
        dg.to_csv(out / "assistant_eval_gemma.csv", index=False)
    else:
        print("GEMINI_API_KEY is not set: only the keyword baseline was tested.")
    table = summary_table(results)
    table.to_csv(out / "assistant_eval_summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
