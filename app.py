"""M6 Dashboard: compare scenarios, build and save custom ones, break results down by
ministry and province, run the model on your own payroll file, and export to Excel.

    python -m fiscalsim.run          # once, writes outputs/dashboard_bundle.pkl
    streamlit run app.py
"""
from __future__ import annotations

import copy
import hashlib
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def drop_stale_package(pkg: str = "fiscalsim") -> None:
    """Streamlit Community Cloud pulls new commits into a server that keeps running, so modules
    imported before the pull stay in memory next to the new app.py. Drop them, and cached results
    built with them, when the package files on disk are newer than the loaded copy."""
    loaded = sys.modules.get(pkg)
    if loaded is None:
        return
    on_disk = max(p.stat().st_mtime for p in (Path(__file__).parent / pkg).glob("*.py"))
    if getattr(loaded, "SOURCE_STAMP", None) != on_disk:
        for name in [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]:
            del sys.modules[name]
        st.cache_resource.clear()


drop_stale_package()

from fiscalsim import assistant as A
from fiscalsim import assistant_eval as AE
from fiscalsim import breakdown, export, upload
from fiscalsim import knowledge as K
from fiscalsim import config as C
from fiscalsim import foundation as FD
from fiscalsim import fund_risk as R
from fiscalsim import m1_data_quality as m1
from fiscalsim import optimize as O
from fiscalsim import refit
from fiscalsim import simulate as S
from fiscalsim import wave as W
from fiscalsim.i18n import EN as I18N_EN
from fiscalsim.i18n import LANGS, check_label, metric_label, scenario_name, t
from fiscalsim.report import LABELS, PALETTE

st.set_page_config(page_title="Cambodia civil service fiscal simulator", layout="wide")
BUNDLE = Path(__file__).parent / "outputs" / "dashboard_bundle.pkl"
CUSTOM_COLOR, MAX_CUSTOM = PALETTE[7], 3
DASHES = ["solid", "dash", "dot"]


@st.cache_resource
def load():
    with open(BUNDLE, "rb") as f:
        return pickle.load(f)


if not BUNDLE.exists():  # first launch without committed results: build them (a few minutes)
    with st.spinner("Building the model and running scenarios for the first time. This takes a few minutes."):
        from fiscalsim.run import main as build
        build(["--runs", "2000", "--out", str(BUNDLE.parent), "--data", str(BUNDLE.parent.parent / "data")])
B = load()
E = B["engine"]


def engine_ctx(stock0=None, pensioners=None, workforce=None, paths=None):
    return SimpleNamespace(workforce=E["workforce"] if workforce is None else workforce,
                           stock0=E["stock0"] if stock0 is None else stock0,
                           data={"pensioners": E["pensioners"] if pensioners is None else pensioners},
                           paths=E["paths"] if paths is None else paths)


def synthetic_state():
    return {"source": "synthetic", "runs": B["runs"], "ctx": engine_ctx(), "base": B["base_breakdown"],
            "frames": {k: B[k] for k in ("table", "fans", "workforce", "adequacy", "wage_bill_sector")},
            "m1": None, "cells": B.get("wave_cells")}


ss = st.session_state
ss.setdefault("data", synthetic_state())
ss.setdefault("custom", {})  # name -> (Scenario, result)
D = ss["data"]
M = {**B, "m2_origin": S.m2.TRAIN_END, **D.get("models", {})}  # published results, with any refitted on uploads
UP = D.get("uploads", {})  # kind -> file name, for the files in use

# ---------- sidebar ----------
with st.sidebar:
    lang = st.radio(t("language"), list(LANGS), format_func=LANGS.get, horizontal=True, key="lang")
    metric = st.selectbox(t("indicator", lang), list(LABELS), format_func=lambda m: metric_label(m, lang), index=4)
    options = list(C.SCENARIOS) + list(ss["custom"])
    chosen = st.multiselect(t("scenarios", lang), options,
                            default=[k for k in ["S0", "S3", "S5", *ss["custom"]] if k in options],
                            format_func=lambda k: f"{k} {scenario_name(k, lang)}" if k in C.SCENARIOS else k)
    band = st.checkbox(t("band", lang), value=True)
    horizon = st.select_slider(t("horizon", lang), options=list(C.HORIZONS), value=2046)


if lang == "km":  # Khmer webfont, with Windows and macOS Khmer fonts as fallbacks
    st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=Kantumruy+Pro:wght@400;600;700&display=swap');
    html, body, [class*="st-"], h1, h2, h3, p, label, button, div {font-family: "Kantumruy Pro", "Khmer UI", "Khmer OS",
    "Noto Sans Khmer", "Leelawadee UI", sans-serif;}
    [data-testid="stIconMaterial"] {font-family: "Material Symbols Rounded" !important;}</style>""", unsafe_allow_html=True)


def custom_frames():
    if not ss["custom"]:
        return None
    return S.results_frames({k: r for k, (_, r) in ss["custom"].items()})


def combined():
    """Scenario frames plus saved custom scenarios."""
    f = copy.copy(D["frames"])
    cf = custom_frames()
    if cf is not None:
        for k in f:
            f[k] = pd.concat([f[k], cf[k]])
    return f


F = combined()
fans, table = F["fans"], F["table"]


def color(k):
    return PALETTE[list(C.SCENARIOS).index(k)] if k in C.SCENARIOS else CUSTOM_COLOR


def dash(k):
    return DASHES[list(ss["custom"]).index(k) % len(DASHES)] if k in ss["custom"] else "solid"


def label(k):
    return f"{k} {scenario_name(k, lang)}" if k in C.SCENARIOS else k


def fan_figure(df, keys, metric, band=True):
    fig = go.Figure()
    for k in keys:
        d = df[(df.scenario == k) & (df.metric == metric)]
        if d.empty:
            continue
        c = color(k)
        if band:
            fig.add_trace(go.Scatter(x=list(d.year) + list(d.year[::-1]), y=list(d.p95) + list(d.p5[::-1]),
                                     fill="toself", fillcolor=c, opacity=0.15, line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=d.year, y=d.p50, name=label(k), line=dict(color=c, width=2, dash=dash(k)),
                                 hovertemplate=f"{k} %{{x}}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(height=430, hovermode="x unified", yaxis_title=metric_label(metric, lang),
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
    return fig


def line_figure(df, ycol, keys, ytitle, scale=1.0):
    fig = go.Figure()
    for k in keys:
        d = df[df.scenario == k]
        fig.add_trace(go.Scatter(x=d.year, y=d[ycol] * scale, name=label(k),
                                 line=dict(color=color(k), width=2, dash=dash(k))))
    fig.update_layout(height=360, yaxis_title=ytitle, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
    return fig


def hbar(s, xtitle):
    s = s.sort_values()
    return go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=PALETTE[0])).update_layout(
        height=max(260, 26 * len(s) + 60), xaxis_title=xtitle, margin=dict(l=10, r=10, t=10, b=10))


# ---------- your own files ----------
def parse_upload(kind, f):
    """(table or None, errors, notes) for one uploaded file. Kept in this session only, one copy per file
    (a staff history can be a few hundred MB, so it is not copied into Streamlit's shared cache)."""
    store = ss.setdefault("parsed", {})
    if kind in store and store[kind][0] == f.file_id:
        return store[kind][1]
    try:
        df = upload.read_table(f.name, f.getvalue())
    except Exception as e:  # unreadable file: say why rather than fail the page
        out = None, [t("data_read_error", lang, e=str(e)[:200])], []
    else:
        if kind == "pensioners":
            table, errors = upload.prepare_pensioners(df)
            out = table, errors, []
        else:
            out = {"hrmis": upload.prepare_hrmis, "history": upload.prepare_history,
                   "macro": upload.prepare_macro}[kind](df)
        del df
    store[kind] = (f.file_id, out)
    return out


@st.cache_data(show_spinner=False)
def example_history_csv():
    return upload.example_history().to_csv(index=False).encode()


UPLOADS = [  # kind, title, what it changes, required and optional columns, example file, example file name
    ("hrmis", "data_hrmis_title", "data_hrmis_what", upload.REQUIRED, upload.OPTIONAL,
     lambda: upload.template_csv(B.get("hrmis_sample")), "hrmis_example.csv"),
    ("pensioners", "data_pens_title", "data_pens_what", upload.PENSIONER_REQUIRED, {},
     lambda: E["pensioners"].head(50).to_csv(index=False).encode(), "pensioners_example.csv"),
    ("history", "data_hist_title", "data_hist_what", upload.HISTORY_REQUIRED, upload.HISTORY_OPTIONAL,
     example_history_csv, "staff_history_example.csv"),
    ("macro", "data_macro_title", "data_macro_what", upload.MACRO_REQUIRED, upload.MACRO_OPTIONAL,
     lambda: B["macro"][[*upload.MACRO_REQUIRED, "gdp_nominal"]].to_csv(index=False).encode(), "macro_example.csv"),
]


def run_uploads(ready: dict, files: dict, log) -> dict:
    """Refit what depends on each uploaded file and rerun every scenario. Returns the new data state."""
    models, rows_used = {}, None
    workforce, paths = E["workforce"], E["paths"]
    if "history" in ready:
        hm = refit.history_models(ready["history"], log)
        workforce, rows_used = hm["workforce"], hm["rows_used"]
        models |= {k: hm[k] for k in ("m2", "km", "wave_backtest", "hierarchy")}
        models |= {"m2_origin": hm["wave_backtest"]["origin"], "shap": {**B["shap"], "exit": hm["shap_exit"]}}
    if "macro" in ready:
        mm = refit.macro_models(ready["macro"], log)
        paths = mm["paths"]
        models |= {k: mm[k] for k in ("m5", "m5_choice", "central", "macro", "foundation")}
    r1 = None
    if "hrmis" in ready:
        log("payroll")
        r1 = m1.run(ready["hrmis"])
    ctx = engine_ctx(S.m2.base_stock(r1["clean"]) if r1 else None, ready.get("pensioners"), workforce, paths)
    log("scenarios")
    results = {k: S.run_scenario(ctx, sc) for k, sc in C.SCENARIOS.items()}
    names = {k: files[k].name for k in ready}
    sha = hashlib.sha1()
    for k in sorted(ready):
        sha.update(files[k].getvalue())
    digest = sha.hexdigest()[:6]
    state = {"source": f"{' + '.join(names.values())} · {digest}", "runs": len(paths["cpi"]), "ctx": ctx,
             "frames": S.results_frames(results), "m1": r1, "uploads": names, "models": models, "rows_used": rows_used,
             "base": breakdown.base_table(r1["records"]) if r1 else B["base_breakdown"],
             "cells": W.cells(r1["clean"]) if r1 else B.get("wave_cells")}
    if r1:
        state |= {"n_records": len(r1["records"]), "n_flagged": int(r1["records"]["flag"].sum())}
    return state


# ---------- header ----------
st.title(t("title", lang))
if D["source"] == "synthetic":
    st.caption(t("caption_synthetic", lang, runs=D["runs"]))
else:
    st.caption(t("caption_files", lang, files=", ".join(UP.values()) or D["source"]) + " "
               + (t("caption_uploaded", lang, n=D["n_records"], flagged=D["n_flagged"], runs=D["runs"])
                  if D["m1"] is not None else t("caption_runs", lang, runs=D["runs"])))

TAB_KEYS = ("tab_compare", "tab_ask", "tab_build", "tab_opt", "tab_wave", "tab_risk", "tab_forecast", "tab_adequacy", "tab_units", "tab_drivers", "tab_checks", "tab_data", "tab_export")
TAB = dict(zip(TAB_KEYS, st.tabs([t(k, lang) for k in TAB_KEYS])))

# ---------- compare ----------
with TAB["tab_compare"]:
    st.plotly_chart(fan_figure(fans, chosen, metric, band), width="stretch")
    h = horizon
    cols = {f"total_cost_gdp_{h}_p50": t("col_total", lang), f"total_cost_gdp_{h}_p5": t("col_p5", lang),
            f"total_cost_gdp_{h}_p95": t("col_p95", lang), f"wage_bill_revenue_{h}_p50": t("col_wb_rev", lang),
            f"pension_spending_gdp_{h}_p50": t("col_pension", lang),
            "prob_fund_depleted_by_2076": t("col_depleted", lang), "median_depletion_year": t("col_dep_year", lang),
            f"headcount_{h}": t("col_headcount", lang)}
    tb = table[list(cols)].rename(columns=cols)
    tb.insert(0, t("col_scenario", lang), [scenario_name(k, lang, table.loc[k, "name"]) for k in table.index])
    v = list(cols.values())
    st.subheader(t("all_scenarios_in", lang, h=h))
    st.dataframe(tb.style.format({c: "{:.2f}" for c in v[:5]} | {v[5]: "{:.0%}", v[6]: "{:.0f}", v[7]: "{:,.0f}"},
                                 na_rep="–"), width="stretch")

# ---------- ask the model (scenario assistant) ----------
ASK_EXAMPLES = {
    "en": ["What if the retirement age is raised to 62?", "Compare the status quo with the combined reform.",
           "What is the cheapest reform that keeps an average replacement rate of 58%?",
           "Which provinces will lose the most staff over the next 10 years?", "What is the current pension formula?"],
    "km": ["បើដំឡើងអាយុចូលនិវត្តន៍ដល់ ៦២ ឆ្នាំ តើមានអ្វីកើតឡើង?", "ប្រៀបធៀបស្ថានភាពបច្ចុប្បន្ន (S0) និងកំណែទម្រង់រួម (S5)",
           "តើកំណែទម្រង់ណាដែលថោកបំផុត ដែលរក្សាអត្រាជំនួសមធ្យមបាន ៥៨%?",
           "ខេត្តណាខ្លះនឹងបាត់បង់មន្ត្រីច្រើនជាងគេ ក្នុងរយៈពេល ១០ ឆ្នាំខាងមុខ?", "តើរូបមន្តសោធនបច្ចុប្បន្នគឺយ៉ាងដូចម្តេច?"],
}


def ask_env():
    """Assistant environment for the data currently loaded (synthetic or uploaded)."""
    envs = ss.setdefault("ask_env", {})
    if D["source"] not in envs:
        envs[D["source"]] = A.Env(ctx=D["ctx"], table=D["frames"]["table"], runs=D["runs"],
                                  synthetic=D["source"] == "synthetic", cells=D["cells"])
    env = envs[D["source"]]
    opt = dict(B.get("optimiser") or {}) if D["source"] == "synthetic" else {}
    opt.update({k[1]: v for k, v in ss.get("opt", {}).items() if k[0] == D["source"]})
    env.optimiser = opt
    return env


def ask_client():
    def secret(name):
        return st.secrets.get(name)
    return A.client_from_settings(secret)


def show_answer(ans):
    st.markdown(f"**{ans.question}**")
    st.write(ans.text)
    how = t("ask_mode_ai" if ans.mode == "gemma" else "ask_mode_rules", lang)
    st.caption(f"{t('ask_understood', lang)}: {A.describe(ans.result, lang, cite=False) or '–'} · {how}")
    if ans.error:
        st.caption(t("ask_ai_error", lang, why=ans.error))
    if ans.fallback:
        st.caption(t("ask_fallback", lang, why=ans.fallback))
    if ans.result.facts or ans.result.sources:
        with st.expander(t("ask_details", lang)):
            facts = {k: f for k, f in ans.result.facts.items() if f.kind != "text" or f.en not in ("Result",)}
            if facts:
                st.dataframe(pd.DataFrame({t("ask_col_item", lang): [f.km if lang == "km" and f.km else f.en for f in facts.values()],
                                           t("ask_col_value", lang): [A.fmt(f, lang) for f in facts.values()]}),
                             hide_index=True, width="stretch")
            if ans.result.sources:
                st.markdown(f"**{t('ask_sources', lang)}**")
                for i in ans.result.sources:
                    e = K.BY_ID[i]
                    mark = "**" if i in ans.cited else ""
                    st.markdown(f"- {mark}[{i}] {e['title']}{mark}: {e['text']} ({t('ask_set_in', lang)} `{e['where']}`)")


with TAB["tab_ask"]:
    st.write(t("ask_intro", lang))
    client = ask_client()
    if client is not None:
        st.caption(t("ask_ai_mode", lang, model=client.model) + " " + t("ask_privacy", lang))
    else:
        st.info(t("ask_no_key", lang))
    st.caption(t("ask_examples", lang))
    picked = None
    for i, q in enumerate(ASK_EXAMPLES[lang]):
        if st.button(q, key=f"ask_ex_{lang}_{i}", type="tertiary"):
            picked = q
    with st.form("ask_form"):
        typed = st.text_input(t("ask_label", lang), key="ask_q")
        asked = st.form_submit_button(t("ask_button", lang), type="primary")
    question = picked or (typed.strip() if asked else "")
    log = ss.setdefault("ask_log", [])
    if question:
        with st.spinner("..."):
            log.insert(0, A.answer(question, ask_env(), client))
        del log[5:]
    for k, ans in enumerate(log):
        if k:
            st.divider()
        show_answer(ans)
    if log and st.button(t("ask_clear", lang)):
        ss["ask_log"] = []
        st.rerun()
    n_q = sum(len(v) for v in AE.SETS.values())
    with st.expander(t("ask_test", lang)):
        st.write(t("ask_test_intro", lang, n=n_q, calls=2 * n_q))
        if st.button(t("ask_test_run", lang), key="ask_test_run"):
            bar = st.progress(0.0, text=t("ask_test_running", lang))
            df, summ = AE.evaluate(ask_env(), client, pause=4.0,
                                   progress=lambda f: bar.progress(min(f, 1.0), text=t("ask_test_running", lang)))
            mode = f"Gemma ({client.model})" if client is not None else "keywords (no AI)"
            ss["ask_test"] = (AE.summary_table({mode: summ}), df)
        if "ask_test" in ss:
            summ_df, df = ss["ask_test"]
            st.dataframe(summ_df.style.format({k: "{:.0%}" for k in (*AE.METRICS, "ai_text_used")}, na_rep="–"),
                         hide_index=True, width="stretch")
            st.download_button(t("ask_test_download", lang), df.to_csv(index=False).encode("utf-8-sig"),
                               "assistant_test.csv", "text/csv")

# ---------- build ----------
with TAB["tab_build"]:
    st.write(t("build_intro", lang))
    c1, c2, c3 = st.columns(3)
    with c1:
        rule = st.radio(t("salary_rule", lang), ["recent_average", "inflation", "targeted"],
                        format_func=lambda r: t(f"rule_{r}", lang))
        raise_ = st.slider(t("fixed_raise", lang), 0.0, 0.10, 0.06, 0.005, format="%.3f")
        growth = st.slider(t("hiring_growth", lang), -0.01, 0.03, 0.01, 0.005, format="%.3f")
        restrain = st.checkbox(t("restrain", lang))
    with c2:
        ret = st.slider(t("ret_age", lang), 55, 65, 55)
        phase = st.slider(t("phase", lang), 1, 20, 10)
        formula = st.radio(t("formula", lang), ["current", "accrual"], format_func=lambda f: t(f"formula_{f}", lang))
        accrual = st.slider(t("accrual", lang), 0.01, 0.035, 0.02, 0.001, format="%.3f")
    with c3:
        ee = st.slider(t("ee", lang), 0.0, 0.12, 0.06, 0.005, format="%.3f")
        er = st.slider(t("er", lang), 0.0, 0.20, 0.12, 0.005, format="%.3f")
        step = st.slider(t("step", lang), 0.0, 0.02, 0.0, 0.0025, format="%.4f")
        cap = st.slider(t("cap", lang), 0.0, 0.06, 0.03, 0.005, format="%.3f")
    custom = C.SCENARIOS["S0"].with_(
        code="Custom", name="Custom", salary_rule=rule, recent_raise=raise_, hiring_growth=growth,
        restrain_non_priority=restrain, retirement_age_target=float(ret), retirement_phase_years=phase,
        pension_formula=formula, accrual_rate=accrual, employee_contribution=ee, employer_contribution=er,
        contribution_step=step, contribution_cap=cap)
    with st.spinner(t("simulating", lang)):
        res = S.run_scenario(D["ctx"], custom)
    row = S.summarize(res)
    one = S.results_frames({"Custom": res})
    st.plotly_chart(fan_figure(pd.concat([fans[fans.scenario == "S0"], one["fans"]]), ["S0", "Custom"], metric, band),
                    width="stretch")
    s0 = table.loc["S0"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("kpi_total", lang, h=horizon), f"{row[f'total_cost_gdp_{horizon}_p50']:.2f}",
              f"{row[f'total_cost_gdp_{horizon}_p50'] - s0[f'total_cost_gdp_{horizon}_p50']:+.2f} {t('vs_s0', lang)}",
              delta_color="inverse")
    k2.metric(t("kpi_rr", lang), f"{row['avg_replacement_2037_2046']:.0%}",
              f"{(row['avg_replacement_2037_2046'] - s0['avg_replacement_2037_2046']) * 100:+.0f} pp {t('vs_s0', lang)}")
    k3.metric(t("kpi_min", lang), f"{row['share_on_minimum_2037_2046']:.0%}")
    k4.metric(t("kpi_dep", lang), f"{row['prob_fund_depleted_by_2076']:.0%}")
    n1, n2 = st.columns([3, 1])
    name = n1.text_input(t("save_custom", lang), value=f"Custom {len(ss['custom']) + 1}", max_chars=30, key="custom_name")
    n2.write("")
    if n2.button(t("save_button", lang), disabled=not name.strip()):
        key = name.strip()
        ss["custom"][key] = (custom.with_(code=key, name=key), S.run_scenario(D["ctx"], custom.with_(code=key, name=key)))
        while len(ss["custom"]) > MAX_CUSTOM:
            ss["custom"].pop(next(iter(ss["custom"])))
        st.success(t("saved", lang))
        st.rerun()

# ---------- AI reform optimiser ----------
def describe(row, lang):
    a = int(row["retirement_age"])
    parts = [t("desc_age_phase" if a > 55 else "desc_age", lang, a=a),
             t("desc_formula_current", lang) if row["pension_formula"] == "current"
             else t("desc_formula_accrual", lang, r=row["accrual_rate"]),
             t("desc_contrib", lang, c=row["contribution_rise_per_side"] * 100) if row["contribution_rise_per_side"]
             else t("desc_contrib_none", lang),
             t(f"rule_{row['salary_rule']}", lang)]
    if row["restrain_non_priority"]:
        parts.append(t("desc_restrain", lang))
    return "; ".join(parts)


with TAB["tab_opt"]:
    st.write(t("opt_intro", lang))
    pay = st.toggle(t("opt_pay", lang), value=False)
    if pay:
        st.caption(t("opt_pay_note", lang))
    okey = (D["source"], "pension_pay" if pay else "pension")
    store = ss.setdefault("opt", {})
    if okey not in store and D["source"] == "synthetic" and "optimiser" in B:
        store[okey] = B["optimiser"][okey[1]]
    if okey not in store:
        st.info(t("opt_not_run", lang))
        if st.button(t("opt_run", lang), type="primary"):
            bar = st.progress(0.0, text=t("opt_running", lang))
            store[okey] = O.search(O.paths_subset(D["ctx"], 200), pay_levers=pay,
                                   progress=lambda f: bar.progress(min(f, 1.0), text=t("opt_running", lang)))
            st.rerun()
    else:
        r = store[okey]
        front, scored, refs = r["front"], r["scored"], r["scenarios"]
        m1_, m2_, m3_ = st.columns(3)
        m1_.metric(t("opt_runs", lang), f"{r['evaluations']:,}", t("opt_runs_of", lang, n=r["grid_size"]),
                   delta_color="off")
        m2_.metric(t("opt_r2_cost", lang), f"{r['surrogate_r2']['cost']:.3f}")
        m3_.metric(t("opt_r2_adequacy", lang), f"{r['surrogate_r2']['adequacy']:.3f}")
        if "grid_check" in r:
            st.caption(t("opt_grid", lang, hv=r["grid_check"]["hypervolume_ratio"]))
        lo, hi = float(front.adequacy.min()), float(front.adequacy.max())
        target = st.slider(t("opt_target", lang), round(lo * 100), int(hi * 100),
                           min(int(hi * 100), round(refs.loc["S0", "adequacy"] * 100) - 1), 1, format="%d%%") / 100
        pick = O.best_for_target(front, min_adequacy=target)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scored.cost, y=scored.adequacy * 100, mode="markers", name=t("opt_scored", lang),
                                 marker=dict(color="#b4b2a9", size=6),
                                 hovertemplate="%{x:.2f}% GDP, %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=front.cost, y=front.adequacy * 100, mode="lines+markers", name=t("opt_front", lang),
                                 line=dict(color=PALETTE[0], width=2), marker=dict(size=8),
                                 text=[describe(row, lang) for _, row in front.iterrows()],
                                 hovertemplate="%{text}<br>%{x:.2f}% GDP, %{y:.1f}%<extra></extra>"))
        # compare like with like: without pay levers, show only the scenarios that change pension rules alone
        for k, row in (refs if pay else refs.loc[["S0", "S3", "S4"]]).iterrows():
            fig.add_trace(go.Scatter(x=[row.cost], y=[row.adequacy * 100], mode="markers+text", text=[k],
                                     textposition="top center", name=label(k), showlegend=False,
                                     marker=dict(color=color(k), size=11, symbol="diamond"),
                                     hovertemplate=f"{label(k)}<br>%{{x:.2f}}% GDP, %{{y:.1f}}%<extra></extra>"))
        if pick is not None:
            fig.add_trace(go.Scatter(x=[pick.cost], y=[pick.adequacy * 100], mode="markers", name=t("opt_picked", lang),
                                     marker=dict(color=CUSTOM_COLOR, size=18, symbol="star",
                                                 line=dict(color="white", width=1))))
        fig.add_hline(y=target * 100, line=dict(color="#888", dash="dot", width=1))
        fig.update_layout(height=460, xaxis_title=t("opt_cost", lang), yaxis_title=t("opt_adequacy", lang),
                          margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

        if pick is None:
            st.warning(t("opt_none", lang))
        else:
            s0 = refs.loc["S0"]
            st.subheader(t("opt_pick", lang))
            st.success(describe(pick, lang))
            k1, k2, k3 = st.columns(3)
            k1.metric(t("opt_cost", lang), f"{pick.cost:.2f}", f"{pick.cost - s0.cost:+.2f} {t('vs_s0', lang)}",
                      delta_color="inverse")
            k2.metric(t("opt_adequacy", lang), f"{pick.adequacy:.0%}",
                      f"{round((pick.adequacy - s0.adequacy) * 100, 1) + 0.0:+.1f} pp {t('vs_s0', lang)}")
            k3.metric(t("opt_depleted", lang), f"{pick.prob_fund_depleted:.0%}")
            if st.button(t("opt_save", lang)):
                name = f"AI pick {len(ss['custom']) + 1}"
                sc = O.package_from_row(pick).scenario(code=name)
                ss["custom"][name] = (sc, S.run_scenario(D["ctx"], sc))
                while len(ss["custom"]) > MAX_CUSTOM:
                    ss["custom"].pop(next(iter(ss["custom"])))
                st.success(t("opt_saved", lang, name=name))
                st.rerun()

        st.subheader(t("opt_table", lang))
        show = front.assign(
            **{t("col_ret", lang): front.retirement_age,
               t("col_formula", lang): [t(f"formula_{f}", lang) + (f" {a:.2%}" if f == "accrual" else "")
                                        for f, a in zip(front.pension_formula, front.accrual_rate)],
               t("col_contrib", lang): (front.contribution_rise_per_side * 100).map("+{:.0f} pp".format),
               t("col_pay", lang): [t(f"rule_{x}", lang) for x in front.salary_rule],
               t("col_restrain", lang): front.restrain_non_priority,
               t("opt_cost", lang): front.cost.round(2),
               t("opt_adequacy", lang): (front.adequacy * 100).round(1),
               t("opt_depleted", lang): (front.prob_fund_depleted * 100).round(0)})
        cols = [t(k, lang) for k in ("col_ret", "col_formula", "col_contrib", "col_pay", "col_restrain",
                                     "opt_cost", "opt_adequacy", "opt_depleted")]
        st.dataframe(show[cols if pay else cols[:3] + cols[5:]], width="stretch", hide_index=True)

# ---------- shared by the retirement-wave and fund-risk tabs ----------
def scenario_of(k):
    return C.SCENARIOS[k] if k in C.SCENARIOS else ss["custom"][k][0]


def unit_name(u):
    parts = u if isinstance(u, tuple) else (u,)
    return " · ".join(t(f"sector_{p}", lang) if p in C.SECTORS else str(p) for p in parts)


def depletion_of(k):
    """Run-out year per Monte Carlo run for a scenario (cached per data source)."""
    if k in ss["custom"]:
        return R.depletion(ss["custom"][k][1])
    cache = ss.setdefault("depletion", {})
    if (D["source"], k) not in cache:
        cache[(D["source"], k)] = R.depletion(S.run_scenario(D["ctx"], C.SCENARIOS[k]))
    return cache[(D["source"], k)]


LEVELS = {"province": ["province"], "ministry": ["ministry"], "province_sector": ["province", "sector"]}
TIER_COLORS = {"high": "#c0392b", "watch": "#d68910", "normal": "#1e8449", "few_staff": "#8a8a8a"}
LIGHT_COLORS = {"green": "#1e8449", "amber": "#e0a100", "red": "#c0392b"}
LIGHT_YEARS = (2030, 2035, 2040, 2050, 2060, 2076)


def lights(probs):
    cols = st.columns(len(probs))
    for col, (year, p) in zip(cols, probs.items()):
        light = R.traffic_light(p)
        col.markdown(
            f"<div style='text-align:center'><div style='width:30px;height:30px;border-radius:50%;margin:0 auto 4px;"
            f"background:{LIGHT_COLORS[light]}'></div><div style='font-weight:600'>{t('risk_by', lang, y=year)}</div>"
            f"<div style='font-size:1.3rem'>{p:.0%}</div><div style='font-size:0.8rem;opacity:0.75'>"
            f"{t(f'light_{light}', lang)}</div></div>", unsafe_allow_html=True)


def year_text(y):
    return t("risk_after", lang) if y > C.END_YEAR else str(int(y))


# ---------- retirement waves ----------
with TAB["tab_wave"]:
    st.write(t("wave_intro", lang))
    if D.get("cells") is None:
        st.info(t("wave_missing", lang))
    else:
        opts = list(C.SCENARIOS) + list(ss["custom"])
        c1, c2, c3 = st.columns([2, 3, 2])
        wsc = c1.selectbox(t("wave_scenario", lang), opts, format_func=label, key="wave_sc")
        lvl = c2.radio(t("wave_level", lang), list(LEVELS), format_func=lambda k: t(f"lvl_{k}", lang),
                       horizontal=True, key="wave_lvl")
        hz = c3.slider(t("wave_horizon", lang), 5, 15, 10, key="wave_hz")
        if D["source"] != "synthetic":
            st.caption(t("wave_upload_hist" if "history" in UP else "wave_upload_note", lang))
        wcells = D["cells"]
        wyears = np.arange(C.BASE_YEAR, C.BASE_YEAR + W.HORIZON)
        wex = W.exit_paths(wcells, D["ctx"].workforce, scenario_of(wsc), wyears)
        wtab, wnat = W.unit_table(wcells, wex, wyears, LEVELS[lvl], hz)
        k1, k2, k3 = st.columns(3)
        k1.metric(t("wave_staff", lang), f"{wnat['staff']:,.0f}")
        k2.metric(t("wave_leave_h", lang, h=hz), f"{wnat['share_h']:.0%}")
        k3.metric(t("wave_high_units", lang), f"{int((wtab.tier == 'high').sum())} / {len(wtab)}")

        top = wtab[wtab.tier != "few_staff"].head(15)
        heat = wnat["by_year"].reindex(top.index).div(top["staff"], axis=0).iloc[:, :hz] * 100
        st.subheader(t("wave_heat", lang))
        fig = go.Figure(go.Heatmap(z=heat.values, x=[str(y) for y in heat.columns], y=[unit_name(u) for u in heat.index],
                                   colorscale="YlOrRd", colorbar=dict(title="%"),
                                   hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"))
        fig.update_layout(height=max(320, 28 * len(heat) + 80), yaxis=dict(autorange="reversed"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        st.subheader(t("wave_table", lang))
        show = pd.DataFrame({
            t("col_unit", lang): [unit_name(u) for u in wtab.index],
            t("col_tier", lang): [t(f"tier_{x}", lang) for x in wtab.tier],
            t("col_staff", lang): wtab.staff.round(0).astype(int).to_numpy(),
            t("col_mean_age", lang): wtab.mean_age.round(1).to_numpy(),
            t("col_50plus", lang): (wtab.aged_50_plus * 100).round(1).to_numpy(),
            t("col_leave5", lang): (wtab.leave_5y * 100).round(1).to_numpy(),
            t("col_leaveh", lang, h=hz): (wtab.leave_h * 100).round(1).to_numpy(),
            t("col_vs", lang): wtab.vs_national.round(2).to_numpy(),
            t("col_peak", lang): wtab.peak_year.to_numpy(),
            t("col_wave_start", lang): [("" if pd.isna(x) else str(int(x))) for x in wtab.wave_start],
        })
        tier_col = t("col_tier", lang)
        colour = {t(f"tier_{k}", lang): v for k, v in TIER_COLORS.items()}
        fmt = {t("col_staff", lang): "{:,}", t("col_vs", lang): "{:.2f}"} | {
            t(k, lang, h=hz): "{:.1f}" for k in ("col_mean_age", "col_50plus", "col_leave5", "col_leaveh")}
        st.dataframe(show.style.format(fmt)
                     .map(lambda v: f"color: {colour.get(v, 'inherit')}; font-weight: 600", subset=[tier_col]),
                     width="stretch", hide_index=True)
        st.caption(t("wave_rule", lang, high=W.HIGH, watch=W.WATCH, wave=W.WAVE_FACTOR))

        st.subheader(t("wave_why", lang))
        ekey = (D["source"], wsc, repr(scenario_of(wsc)), lvl, hz)
        cache = ss.setdefault("wave_explain", {})
        if ekey not in cache:
            with st.spinner(t("wave_explaining", lang)):
                cache[ekey] = W.explain(wcells, wex, LEVELS[lvl], hz)[0]
        expl = cache[ekey]
        rated = [u for u in wtab.index if wtab.loc[u, "tier"] != "few_staff"] or list(wtab.index)
        unit = st.selectbox(t("wave_why_unit", lang), rated, format_func=unit_name)
        contrib = expl.loc[unit].rename(lambda g: t(f"grp_{g}", lang)).sort_values()
        st.plotly_chart(hbar(contrib, t("wave_why_axis", lang)), width="stretch")
        driver = contrib.abs().idxmax()
        st.write(t("wave_why_text", lang, unit=unit_name(unit), share=wtab.loc[unit, "leave_h"], h=hz,
                   nat=wnat["share_h"], driver=driver.lower() if lang == "en" else driver, pp=contrib[driver]))
        bt = M.get("wave_backtest")
        if bt:
            with st.expander(t("wave_check", lang)):
                st.write(t("wave_check_text", lang, origin=bt["origin"], y0=bt["years"][0], y1=bt["years"][-1],
                           units=bt["units"]))
                st.dataframe(bt["table"], width="stretch")

# ---------- pension fund risk ----------
with TAB["tab_risk"]:
    st.write(t("risk_intro", lang))
    opts = list(C.SCENARIOS) + list(ss["custom"])
    rsc = st.selectbox(t("risk_scenario", lang), opts, format_func=label, key="risk_sc")
    dep = depletion_of(rsc)
    st.subheader(t("risk_sim_title", lang, runs=len(dep)))
    lights(pd.Series({y: float((dep <= y).mean()) for y in LIGHT_YEARS}))
    st.write("")
    ran_out = dep[dep <= C.END_YEAR]
    if len(ran_out) >= 0.5 * len(dep):
        st.write(t("risk_sim_year", lang, med=year_text(np.median(dep)), lo=year_text(np.percentile(dep, 5)),
                   hi=year_text(np.percentile(dep, 95))))
    if len(ran_out) < len(dep):
        st.write(t("risk_sim_never", lang, p=1 - len(ran_out) / len(dep)))

    RK = B.get("fund_risk")
    if RK:
        model = RK["model"]
        st.subheader(t("risk_ai_title", lang))
        st.caption(t("risk_ai_note", lang, n=RK["evaluation"]["train_packages"]))
        if D["source"] != "synthetic":
            st.caption(t("risk_ai_upload", lang))
        rng_ = RK["driver_ranges"]
        cen = RK["central"]
        a1, a2, a3, a4 = st.columns(4)
        slider = lambda col, key, name, step: col.slider(
            t(key, lang), float(np.floor(rng_[name].iloc[0] / step) * step), float(np.ceil(rng_[name].iloc[1] / step) * step),
            float(round(cen[name] / step) * step), step, key=f"risk_{name}")
        assume = {"avg_inflation": slider(a1, "a_inflation", "avg_inflation", 0.1),
                  "avg_fund_real_return": slider(a2, "a_return", "avg_fund_real_return", 0.1),
                  "avg_real_growth": slider(a3, "a_growth", "avg_real_growth", 0.1),
                  "mortality_level": slider(a4, "a_mortality", "mortality_level", 0.01)}
        feats = {**R.lever_features(scenario_of(rsc)), **assume}
        lights(model.curve(feats, LIGHT_YEARS))
        st.write("")
        med, lo, hi = model.year_range(feats)
        st.write(t("risk_ai_year", lang, med=year_text(med), lo=year_text(lo), hi=year_text(hi)))

        grid = model.GRID[model.GRID <= C.END_YEAR]
        ai_curve = model.cdf(pd.DataFrame([feats]))[0][: len(grid)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=[(dep <= y).mean() * 100 for y in grid], name=t("risk_chart_sim", lang),
                                 line=dict(color=PALETTE[0], width=2, shape="hv")))
        fig.add_trace(go.Scatter(x=grid, y=ai_curve * 100, name=t("risk_chart_ai", lang),
                                 line=dict(color=CUSTOM_COLOR, width=2, dash="dash")))
        for y0, y1, c in ((0, 10, LIGHT_COLORS["green"]), (10, 50, LIGHT_COLORS["amber"]), (50, 100, LIGHT_COLORS["red"])):
            fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.06, line_width=0)
        fig.update_layout(height=360, yaxis=dict(title="%", range=[0, 101]), hovermode="x unified",
                          title=t("risk_chart", lang), margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", x=0, y=-0.15))
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        local = model.shap_local(feats)
        local = local[local.abs().sort_values(ascending=False).index[:8]].rename(lambda f: t(f"f_{f}", lang))
        c1.subheader(t("risk_why", lang))
        c1.plotly_chart(hbar(local, "SHAP"), width="stretch")
        c2.subheader(t("risk_drivers", lang))
        c2.plotly_chart(hbar(RK["shap"].head(8).rename(lambda f: t(f"f_{f}", lang)), "|SHAP|"), width="stretch")
        ev = RK["evaluation"]
        with st.expander(t("risk_check", lang)):
            st.write(t("risk_check_text", lang, train=ev["train_packages"], test=ev["test_packages"],
                       cov=ev["target_coverage"]))
            st.dataframe(ev["classifier"], width="stretch")
            st.dataframe(ev["year_range"], width="stretch")

# ---------- AI forecasts: wage bill by unit, pretrained models for the macro series ----------
MODEL_COLORS = {"chronos_2": PALETTE[1], "timesfm_2_5": PALETTE[2], "chronos_bolt": PALETTE[4], "arima": PALETTE[0],
                "lstm": PALETTE[6], "prophet": "#8c8c8c", "linear_trend": "#a6a6a6", "mean_8y": "#bdbdbd", "naive": "#d0d0d0"}


def model_name(m):
    return t(f"fmn_{m}", lang) if f"fmn_{m}" in I18N_EN else FD.NAMES.get(m, m)


def unit_name(sid):
    if sid == "national":
        return t("hf_national", lang)
    level, name = sid.split(":", 1)
    return t(f"sector_{name}", lang) if level == "sector" else name


def units_forecast(HR):
    f = HR["forecast"]
    st.write(t("hf_intro", lang))
    last, end = int(f[f.kind == "history"].year.max()), int(f.year.max())
    nat = f[f.series == "national"].set_index(["kind", "year"])["wage_bill_bn"]
    bn = t("hf_bn", lang)
    c = st.columns(4)
    c[0].metric(t("hf_now", lang, y=last), f"{nat[('history', last)]:,.0f} {bn}")
    c[1].metric(t("hf_then", lang, y=end), f"{nat[('forecast', end)]:,.0f} {bn}")
    c[2].metric(t("hf_growth", lang), f"{(nat[('forecast', end)] / nat[('history', last)]) ** (1 / (end - last)) - 1:.1%}")
    c[3].metric(t("hf_gap", lang), f"{HR['gap']['province']:.2f}%", help=t("hf_gap_help", lang))

    c1, c2 = st.columns(2)
    level = c1.radio(t("hf_level", lang), ["province", "ministry", "sector"], format_func=lambda k: t(f"hl_{k}", lang),
                     horizontal=True, key="hf_level")
    size = (f[(f.level == level) & (f.kind == "history") & (f.year == last)]
            .set_index("series")["wage_bill_bn"].sort_values(ascending=False))
    unit = c2.selectbox(t("hf_unit", lang), ["national", *size.index], format_func=unit_name, key=f"hf_unit_{level}")

    d = f[f.series == unit]
    hist = d[d.kind == "history"]
    joined = lambda k: ([last] + list(d[d.kind == k].year), [hist.wage_bill_bn.iloc[-1]] + list(d[d.kind == k].wage_bill_bn))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.year, y=hist.wage_bill_bn, name=t("hf_hist", lang), line=dict(color=PALETTE[0], width=2)))
    x, y = joined("forecast")
    fig.add_trace(go.Scatter(x=x, y=y, name=t("hf_fc", lang), line=dict(color=PALETTE[1], width=2.5)))
    x, y = joined("base")
    fig.add_trace(go.Scatter(x=x, y=y, name=t("hf_base", lang), line=dict(color="#8c8c8c", width=1.5, dash="dot")))
    fig.update_layout(height=360, hovermode="x unified", yaxis_title=bn, title=t("hf_chart", lang, u=unit_name(unit)),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", x=0, y=-0.15))
    fig.update_traces(hovertemplate="%{y:,.1f}")
    st.plotly_chart(fig, width="stretch")

    st.subheader(t("hf_table", lang, lvl=t(f"hl_pl_{level}", lang)))
    cur = f[f.kind.isin(["history", "forecast"]) & (f.year >= last)]
    tab = cur[cur.level == level].pivot_table(index="series", columns="year", values="wage_bill_bn").loc[size.index]
    tab.loc["__sum__"] = tab.sum()
    tab.loc["national"] = cur[cur.series == "national"].set_index("year")["wage_bill_bn"]
    gcol = t("hf_col_growth", lang, a=last, b=end)
    tab[gcol] = tab[end] / tab[last] - 1
    tab.index = [t("hf_sum", lang) if s == "__sum__" else unit_name(s) for s in tab.index]
    tab.columns = [str(c) for c in tab.columns]
    tab.index.name = t(f"hl_{level}", lang)
    st.dataframe(tab.style.format({**{str(y): "{:,.1f}" for y in range(last, end + 1)}, gcol: "{:+.1%}"})
                 .map(lambda _: "font-weight: 600", subset=pd.IndexSlice[tab.index[-2:], :]),
                 width="stretch", height=min(36 * len(tab) + 40, 460))
    st.caption(t("hf_sum_note", lang, s=tab[str(end)].iloc[-2], y=end, n=tab[str(end)].iloc[-1]))

    with st.expander(t("hf_check", lang)):
        tables = HR["tables"]
        wins = ", ".join(t("hf_wins_item", lang, a=t(f"hm_{k}", lang).lower() if lang == "en" else t(f"hm_{k}", lang), p=v)
                         for k, v in HR["wins"].items())
        st.write(t("hf_check_text", lang, o0=HR["origins"][0], o1=HR["origins"][-1], h=HR["max_h"],
                   m=f"{model_name(HR['base_model'])} + {t('hm_' + HR['method'], lang)}", w=wins))
        show = tables[HR["base_model"]].rename(index=lambda m: t(f"hm_{m}", lang), columns=lambda c: t(f"hl_{c}", lang))
        show.index.name, show.columns.name = None, None
        st.dataframe(show.style.format("{:.2f}").highlight_min(axis=0, props="font-weight: 700"), width="stretch")
        if len(tables) > 1:
            st.write(t("hf_check_fm", lang))
            comp = pd.DataFrame({model_name(b): v["average"] for b, v in tables.items()})
            comp = comp.rename(index=lambda m: t(f"hm_{m}", lang)).rename_axis(None).rename_axis(t("hf_base_col", lang), axis=1)
            st.dataframe(comp.style.format("{:.2f}").highlight_min(axis=None, props="font-weight: 700"), width="stretch")


def macro_models(FMR):
    st.write(t("fm_intro", lang))
    if not any(m in FD.FOUNDATION for m in FMR["models"]):
        st.info(t("fm_upload" if "macro" in UP else "fm_not_run", lang))
    else:
        st.caption(t("fm_ran", lang, date=FMR.get("info", {}).get("date", "")))
    st.subheader(t("fm_summary", lang, h=FMR["h"], o0=FMR["origins"][0], o1=FMR["origins"][-1]))
    ov = FMR["overall"].sort_values("rel_MAE_vs_naive")
    cols = {"rel_MAE_vs_naive": t("fm_col_rel", lang), "mean_rank": t("fm_col_rank", lang),
            "coverage_80_pct": t("fm_col_cov", lang), "quantile_loss_pp": t("fm_col_ql", lang)}
    fmts = {"rel_MAE_vs_naive": "{:.2f}", "mean_rank": "{:.1f}", "coverage_80_pct": "{:.0f}", "quantile_loss_pp": "{:.2f}"}
    # text cells, so models without a range show a dash rather than the table's "None"
    show = pd.DataFrame({cols[c]: [("–" if pd.isna(v) else f.format(v)) for v in ov[c]] for c, f in fmts.items()})
    bold = pd.DataFrame({cols[c]: ["font-weight: 700" if c != "coverage_80_pct" and v == ov[c].min() else ""
                                   for v in ov[c]] for c in fmts})
    show.insert(0, t("fm_col_type", lang), [t("fm_type_found" if m in FD.FOUNDATION else "fm_type_m5", lang) for m in ov.index])
    show.index = bold.index = [model_name(m) for m in ov.index]
    show.index.name = t("fm_col_model", lang)
    st.dataframe(show.style.apply(lambda _: bold, axis=None, subset=list(bold.columns)), width="stretch")
    st.caption(t("fm_summary_note", lang))

    target = st.radio(t("fm_series", lang), list(FD.m5.TARGETS), format_func=lambda k: t(f"fm_{k}", lang),
                      horizontal=True, key="fm_target")
    roll = FMR["rolling"][FMR["rolling"].target == target].set_index("model")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    hs = list(range(1, FMR["h"] + 1))
    for m in FMR["models"]:
        found = m in FD.FOUNDATION
        fig.add_trace(go.Scatter(x=hs, y=[roll.loc[m, f"MAE_h{h}"] for h in hs], name=model_name(m), mode="lines+markers",
                                 line=dict(color=MODEL_COLORS.get(m), width=3 if found else 1.5, dash=None if found else "dot")))
    fig.update_layout(height=380, title=t("fm_by_h", lang), xaxis=dict(title=t("fm_h_axis", lang), dtick=1),
                      yaxis_title=t("fm_err_axis", lang), hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", x=0, y=-0.2))
    c1.plotly_chart(fig, width="stretch")

    macro = M["macro"]
    last = int(macro["year"].max())
    fut = FMR["future"][FMR["future"].series == target]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macro["year"], y=macro[target] * 100, name=t("fm_history", lang), line=dict(color="#333", width=2)))
    for m in [m for m in FMR["models"] if m in FD.FOUNDATION or m == "arima"]:
        d = fut[fut.model == m].sort_values("year")
        col = MODEL_COLORS.get(m)
        fig.add_trace(go.Scatter(x=list(d.year) + list(d.year[::-1]), y=list(d.q90 * 100) + list(d.q10[::-1] * 100),
                                 fill="toself", fillcolor=col, opacity=0.12, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=d.year, y=d.q50 * 100, name=model_name(m), line=dict(color=col, width=2)))
    cen = M["central"][target].iloc[:len(fut["year"].unique())] * 100
    fig.add_trace(go.Scatter(x=cen.index, y=cen.values, name=t("fm_m5_path", lang), line=dict(color="#333", width=2, dash="dash")))
    shown = np.r_[macro.loc[macro.year >= last - 15, target], fut.q10.dropna(), fut.q90.dropna(), cen / 100] * 100
    pad = 0.1 * (shown.max() - shown.min())
    fig.update_layout(height=380, title=t("fm_future", lang, y=last), hovermode="x unified",
                      yaxis=dict(title="%", range=[shown.min() - pad, shown.max() + pad]),
                      xaxis=dict(range=[last - 15, int(fut.year.max()) + 0.5]), margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", x=0, y=-0.2))
    fig.update_traces(hovertemplate="%{y:.1f}")
    c2.plotly_chart(fig, width="stretch")

    if "p_vs_arima" in roll:
        with st.expander(t("fm_dm", lang)):
            dm = roll.loc[[m for m in roll.index if m in FD.FOUNDATION], ["DM_vs_arima", "p_vs_arima", "DM_vs_lstm", "p_vs_lstm"]]
            dm.index = [model_name(m) for m in dm.index]
            st.dataframe(dm.style.format("{:.2f}", na_rep="–"), width="stretch")
            st.caption(t("fm_dm_text", lang, n=len(FMR["origins"])))
    with st.expander(t("fm_same", lang, y0=FMR.get("train_end", FD.m5.TRAIN_END), y1=FMR.get("train_end", FD.m5.TRAIN_END) + 1, y2=last)):
        same = FMR["same_as_m5"].pivot(index="model", columns="target", values="MAPE_%")[list(FD.m5.TARGETS)]
        same = same.reindex([m for m in ov.index if m in same.index])
        same.index = [model_name(m) for m in same.index]
        same.columns = [t(f"fm_{c}", lang) for c in same.columns]
        st.dataframe(same.style.format("{:.2f}").highlight_min(axis=0, props="font-weight: 700"), width="stretch")
        st.caption(t("fm_same_note", lang, n=len(FMR["origins"])))


with TAB["tab_forecast"]:
    HR, FMR = M.get("hierarchy"), M.get("foundation")
    if not HR or not FMR:
        st.info(t("fc_missing", lang))
    else:
        part = st.radio(t("fc_show", lang), ["units", "macro"], format_func=lambda k: t(f"fc_part_{k}", lang),
                        horizontal=True, key="fc_part")
        if part == "units":
            if D["source"] != "synthetic":
                st.caption(t("fc_hist_yours" if "history" in UP else "fc_hist_synth", lang))
            units_forecast(HR)
        else:
            macro_models(FMR)

# ---------- adequacy and workforce ----------
with TAB["tab_adequacy"]:
    ad = F["adequacy"].reset_index()
    st.subheader(t("adequacy_title", lang))
    st.plotly_chart(line_figure(ad, "avg_replacement", chosen, t("rr_axis", lang), 100), width="stretch")
    st.subheader(t("min_title", lang))
    st.plotly_chart(line_figure(ad, "share_on_minimum", chosen, t("min_axis", lang), 100), width="stretch")
    wf = F["workforce"].reset_index()
    st.subheader(t("headcount_title", lang))
    st.plotly_chart(line_figure(wf, "total", chosen, t("headcount_axis", lang)), width="stretch")

# ---------- ministries and provinces ----------
with TAB["tab_units"]:
    st.caption(t("units_intro", lang))
    c1, c2 = st.columns(2)
    level = c1.radio(t("units_level", lang), ["ministry", "province"], format_func=lambda x: t(x, lang), horizontal=True)
    proj_sc = c2.selectbox(t("units_scenario", lang), list(dict.fromkeys(["S0", *chosen, *C.SCENARIOS])), format_func=label)
    base = D["base"]
    agg = breakdown.by(base, level)
    sector_wb = F["wage_bill_sector"].loc[proj_sc]
    proj = breakdown.projected(base, sector_wb, level, horizon)
    base_year_wb = breakdown.projected(base, sector_wb, level, C.BASE_YEAR)
    out = pd.DataFrame({
        t("headcount", lang): agg["headcount"],
        t("mean_age", lang): agg["mean_age"],
        t("aged_50", lang): agg["aged_50_plus"],
        t("wb_base", lang, y=C.BASE_YEAR): agg["wage_bill"] / 1e9,
        t("wb_proj", lang, y=horizon, s=proj_sc): proj.reindex(agg.index) / 1e9,
        t("growth", lang): proj.reindex(agg.index) / base_year_wb.reindex(agg.index) - 1,
        t("flagged", lang): agg["flagged_records"],
        t("flagged_share", lang): agg["flagged_share_of_payroll"],
    })
    out.index.name = t(level, lang)
    cols = list(out.columns)
    st.dataframe(out.style.format({cols[0]: "{:,.0f}", cols[1]: "{:.1f}", cols[2]: "{:,.0f}", cols[3]: "{:,.1f}",
                                   cols[4]: "{:,.1f}", cols[5]: "{:+.0%}", cols[6]: "{:,.0f}", cols[7]: "{:.1%}"}),
                 width="stretch", height=min(38 * len(out) + 40, 520))
    top = out.head(15)
    fig = go.Figure([
        go.Bar(y=top.index[::-1], x=top[cols[3]][::-1], name=str(C.BASE_YEAR), orientation="h", marker_color=PALETTE[0]),
        go.Bar(y=top.index[::-1], x=top[cols[4]][::-1], name=f"{horizon} {proj_sc}", orientation="h",
               marker_color=color(proj_sc) if proj_sc != "S0" else PALETTE[6])])
    fig.update_layout(barmode="group", height=max(320, 34 * len(top) + 80), xaxis_title="bn riel",
                      title=t("units_chart", lang, lvl=t(level, lang)), margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, width="stretch")
    fs = out[cols[7]].sort_values(ascending=False).head(15) * 100
    st.plotly_chart(hbar(fs, "%").update_layout(title=t("units_flag_chart", lang, lvl=t(level, lang)),
                                                margin=dict(l=10, r=10, t=40, b=10)), width="stretch")

# ---------- drivers ----------
with TAB["tab_drivers"]:
    sh = M["shap"]
    if D["source"] != "synthetic":
        st.caption(t("drivers_upload_hist" if "history" in UP else "drivers_upload_synth", lang))
    c1, c2 = st.columns(2)
    c1.subheader(t("drivers_policy", lang))
    c1.plotly_chart(hbar(sh["policy"], t("shap_axis_gdp", lang)), width="stretch")
    c2.subheader(t("drivers_unc", lang))
    c2.plotly_chart(hbar(sh["uncertainty"], t("shap_axis_gdp", lang)), width="stretch")
    st.subheader(t("drivers_exit", lang))
    st.plotly_chart(hbar(sh["exit"].head(10), t("shap_axis_logodds", lang)), width="stretch")

# ---------- checks ----------
with TAB["tab_checks"]:
    st.subheader(t("checks_m1", lang))
    st.info(t("m1_note", lang))
    if D["source"] == "synthetic" and B.get("m1") is not None:
        st.dataframe(B["m1"], width="stretch")
        st.dataframe(B["m1_types"].rename(index=lambda k: check_label(k, lang)), width="stretch")
    else:
        st.caption("Precision and recall need known answers, which only the synthetic data has. "
                   "The Your data tab shows which records M1 asks you to verify and why.")
    st.subheader(t("checks_m2", lang, y0=M["m2_origin"] + 1, y1=M["m2_origin"] + S.m2.TEST_YEARS))
    if "history" in UP:
        st.caption(t("checks_from_upload", lang))
    st.dataframe(M["m2"], width="stretch")
    km = M["km"]
    fig = go.Figure()
    for i, (s, d) in enumerate(km.groupby("sector")):
        fig.add_trace(go.Scatter(x=d.service_years, y=d.survival, name=s, line=dict(color=PALETTE[i], width=2)))
    fig.update_layout(height=340, xaxis_title=t("checks_km_x", lang), yaxis_title=t("checks_km_y", lang),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    m5_end = S.m5.train_end(M["macro"])
    st.subheader(t("checks_m5", lang, y0=m5_end + 1, y1=m5_end + S.m5.TEST_YEARS))
    if "macro" in UP:
        st.caption(t("checks_from_upload", lang))
    st.dataframe(M["m5"].pivot(index="model", columns="target", values="MAPE_%"), width="stretch")
    st.caption(f"{t('checks_chosen', lang)}: {M['m5_choice']}")

# ---------- your data ----------
with TAB["tab_data"]:
    st.write(t("data_intro", lang))
    uploaded = {}
    for kind, title, what, req, opt, example, fname in UPLOADS:
        with st.container(border=True):
            st.markdown(f"**{t(title, lang)}**")
            st.caption(t(what, lang))
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            f = c1.file_uploader(t("data_upload_file", lang), type=["csv", "xlsx", "xls", "parquet"], key=f"up_{kind}")
            c2.download_button(t("data_template", lang), example(), file_name=fname, mime="text/csv", key=f"ex_{kind}")
            with st.expander(t("data_columns", lang)):
                st.table(upload.column_table(req, opt))
            if f is None:
                ss.get("parsed", {}).pop(kind, None)
            else:
                with st.spinner(t("data_checking", lang)):
                    df, errors, notes = parse_upload(kind, f)
                if errors:
                    st.error(t("data_errors", lang) + ":\n\n- " + "\n- ".join(errors))
                else:
                    st.success(t("data_ok", lang))
                    if notes:
                        st.info(t("data_notes", lang) + ":\n\n- " + "\n- ".join(notes))
                uploaded[kind] = (f, df, errors)

    ready = {k: df for k, (_, df, errors) in uploaded.items() if not errors}
    if not uploaded:
        st.caption(t("data_need_file", lang))
    elif len(ready) < len(uploaded):
        st.warning(t("data_fix_first", lang))
    else:
        st.caption(t("data_light", lang, rows=refit.MAX_ROWS, runs=refit.RUNS))
        if st.button(t("data_run", lang), type="primary"):
            with st.status(t("data_running", lang), expanded=True) as status:
                ss["data"] = run_uploads(ready, {k: f for k, (f, _, _) in uploaded.items()},
                                         lambda step: status.write(t(f"data_step_{step}", lang)))
                ss["custom"] = {k: (sc, S.run_scenario(ss["data"]["ctx"], sc)) for k, (sc, _) in ss["custom"].items()}
                status.update(label=t("data_finished", lang), state="complete")
            st.rerun()

    if D["source"] != "synthetic":
        st.divider()
        ups = D.get("uploads", {})
        st.success(t("data_using", lang, files=", ".join(ups.values()) or D["source"]))
        if D["m1"] is not None:
            st.write(t("data_done", lang, n=D["n_records"], f=D["n_flagged"], k=D["n_records"] - D["n_flagged"]))
        if "pensioners" in ups:
            st.write(t("data_rebuilt_pens", lang, n=len(D["ctx"].data["pensioners"])))
        if "history" in ups:
            st.write(t("data_rebuilt_history", lang, rows=D["rows_used"]))
        if "macro" in ups:
            st.write(t("data_rebuilt_macro", lang, choice=", ".join(f"{t('fm_' + k, lang)}: {FD.NAMES.get(v, v)}"
                                                                    for k, v in M["m5_choice"].items())))
        if D["m1"] is not None:
            recs = D["m1"]["records"]
            flagged = recs[recs["flag"]]
            reasons = pd.Series({
                "duplicate_match": int(m1.record_matching(recs).sum()),
                "iforest": int(recs["iforest_flag"].sum()),
                **{k: int(v) for k, v in m1.rule_flags(recs)[["past_retirement", "no_attendance", "off_scale_salary"]].sum().items()},
            }, name="records").rename(index=lambda k: check_label(k, lang))
            st.info(t("m1_note", lang))
            st.subheader(t("data_flag_reasons", lang))
            st.dataframe(reasons, width="stretch")
            st.subheader(t("data_flagged_rows", lang))
            st.dataframe(flagged.sort_values("anomaly_score", ascending=False).head(500), width="stretch")
        if st.button(t("data_reset", lang)):
            ss["data"] = synthetic_state()
            ss["custom"] = {k: (sc, S.run_scenario(ss["data"]["ctx"], sc)) for k, (sc, _) in ss["custom"].items()}
            st.rerun()

# ---------- export ----------
with TAB["tab_export"]:
    st.write(t("export_intro", lang))
    base = D["base"]
    scen = dict(C.SCENARIOS) | {k: sc for k, (sc, _) in ss["custom"].items()}
    source = ("Synthetic data shaped like HRMIS, Budget Law, NSSF-C and macro inputs"
              if D["source"] == "synthetic" else f"Uploaded files: {', '.join(UP.values()) or D['source']}; anything else synthetic")
    extra = {}
    if D["m1"] is not None:
        extra["M1 records to verify"] = D["m1"]["records"][D["m1"]["records"]["flag"]].drop(columns=["bank_account"], errors="ignore")
    for (src, k), r in ss.get("opt", {}).items():
        if src == D["source"]:
            extra[f"AI optimiser {k}"] = r["front"]
    xlsx = export.workbook(F, scen, D["runs"], source, breakdown.by(base, "ministry"), breakdown.by(base, "province"), extra)
    st.download_button(t("export_xlsx", lang), xlsx, file_name="fiscal_simulation_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.download_button(t("export_csv", lang), table.to_csv().encode(), file_name="scenario_summary.csv", mime="text/csv")
    st.caption(t("export_charts", lang))
