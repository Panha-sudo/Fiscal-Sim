"""M6 Dashboard: compare scenarios, build and save custom ones, break results down by
ministry and province, run the model on your own payroll file, and export to Excel.

    python -m fiscalsim.run          # once, writes outputs/dashboard_bundle.pkl
    streamlit run app.py
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fiscalsim import breakdown, export, upload
from fiscalsim import config as C
from fiscalsim import m1_data_quality as m1
from fiscalsim import simulate as S
from fiscalsim.i18n import LANGS, metric_label, scenario_name, t
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


def engine_ctx(stock0=None, pensioners=None):
    return SimpleNamespace(workforce=E["workforce"], stock0=E["stock0"] if stock0 is None else stock0,
                           data={"pensioners": E["pensioners"] if pensioners is None else pensioners},
                           paths=E["paths"])


def synthetic_state():
    return {"source": "synthetic", "runs": B["runs"], "ctx": engine_ctx(), "base": B["base_breakdown"],
            "frames": {k: B[k] for k in ("table", "fans", "workforce", "adequacy", "wage_bill_sector")},
            "m1": None}


ss = st.session_state
ss.setdefault("data", synthetic_state())
ss.setdefault("custom", {})  # name -> (Scenario, result)
D = ss["data"]

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
    "Noto Sans Khmer", "Leelawadee UI", sans-serif;}</style>""", unsafe_allow_html=True)


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


# ---------- header ----------
st.title(t("title", lang))
if D["source"] == "synthetic":
    st.caption(t("caption_synthetic", lang, runs=D["runs"]))
else:
    st.caption(t("caption_uploaded", lang, n=D["n_records"], flagged=D["n_flagged"], runs=D["runs"]))

tabs = st.tabs([t(k, lang) for k in ("tab_compare", "tab_build", "tab_adequacy", "tab_units", "tab_drivers",
                                     "tab_checks", "tab_data", "tab_export")])

# ---------- compare ----------
with tabs[0]:
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

# ---------- build ----------
with tabs[1]:
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

# ---------- adequacy and workforce ----------
with tabs[2]:
    ad = F["adequacy"].reset_index()
    st.subheader(t("adequacy_title", lang))
    st.plotly_chart(line_figure(ad, "avg_replacement", chosen, t("rr_axis", lang), 100), width="stretch")
    st.subheader(t("min_title", lang))
    st.plotly_chart(line_figure(ad, "share_on_minimum", chosen, t("min_axis", lang), 100), width="stretch")
    wf = F["workforce"].reset_index()
    st.subheader(t("headcount_title", lang))
    st.plotly_chart(line_figure(wf, "total", chosen, t("headcount_axis", lang)), width="stretch")

# ---------- ministries and provinces ----------
with tabs[3]:
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
with tabs[4]:
    sh = B["shap"]
    c1, c2 = st.columns(2)
    c1.subheader(t("drivers_policy", lang))
    c1.plotly_chart(hbar(sh["policy"], t("shap_axis_gdp", lang)), width="stretch")
    c2.subheader(t("drivers_unc", lang))
    c2.plotly_chart(hbar(sh["uncertainty"], t("shap_axis_gdp", lang)), width="stretch")
    st.subheader(t("drivers_exit", lang))
    st.plotly_chart(hbar(sh["exit"].head(10), t("shap_axis_logodds", lang)), width="stretch")

# ---------- checks ----------
with tabs[5]:
    st.subheader(t("checks_m1", lang))
    if D["source"] == "synthetic":
        st.dataframe(B["m1"], width="stretch")
        st.dataframe(B["m1_types"], width="stretch")
    else:
        st.caption("Precision and recall need known answers, which only the synthetic data has. "
                   "The Your data tab shows what M1 flagged in your file and why.")
    st.subheader(t("checks_m2", lang))
    st.dataframe(B["m2"], width="stretch")
    km = B["km"]
    fig = go.Figure()
    for i, (s, d) in enumerate(km.groupby("sector")):
        fig.add_trace(go.Scatter(x=d.service_years, y=d.survival, name=s, line=dict(color=PALETTE[i], width=2)))
    fig.update_layout(height=340, xaxis_title=t("checks_km_x", lang), yaxis_title=t("checks_km_y", lang),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.subheader(t("checks_m5", lang))
    st.dataframe(B["m5"].pivot(index="model", columns="target", values="MAPE_%"), width="stretch")
    st.caption(f"{t('checks_chosen', lang)}: {B['m5_choice']}")

# ---------- your data ----------
with tabs[6]:
    st.write(t("data_intro", lang))
    st.download_button(t("data_template", lang), upload.template_csv(B.get("hrmis_sample")),
                       file_name="hrmis_template.csv", mime="text/csv")
    with st.expander(t("data_required", lang) + " / " + t("data_optional", lang)):
        st.table(pd.DataFrame(
            [(c, d, "") for c, d in upload.REQUIRED.items()] + [(c, d, dflt) for c, (d, dflt) in upload.OPTIONAL.items()],
            columns=["column", "meaning", "default if missing"]))
    f_hr = st.file_uploader(t("data_upload_hrmis", lang), type=["csv", "xlsx", "xls"], key="up_hr")
    f_pn = st.file_uploader(t("data_upload_pens", lang), type=["csv", "xlsx", "xls"], key="up_pn")
    if f_hr is not None:
        recs, errors, notes = upload.prepare_hrmis(upload.read_table(f_hr.name, f_hr.getvalue()))
        pens, perr = (None, [])
        if f_pn is not None:
            pens, perr = upload.prepare_pensioners(upload.read_table(f_pn.name, f_pn.getvalue()))
        if errors or perr:
            st.error(t("data_errors", lang) + ":\n\n- " + "\n- ".join(errors + perr))
        else:
            if notes:
                st.info(t("data_notes", lang) + ":\n\n- " + "\n- ".join(notes))
            if st.button(t("data_run", lang), type="primary"):
                with st.spinner(t("data_running", lang)):
                    r1 = m1.run(recs)
                    ctx = engine_ctx(S.m2.base_stock(r1["clean"]), pens)
                    results = {k: S.run_scenario(ctx, sc) for k, sc in C.SCENARIOS.items()}
                    ss["data"] = {"source": f_hr.name, "runs": len(E["paths"]["cpi"]), "ctx": ctx,
                                  "base": breakdown.base_table(r1["records"]),
                                  "frames": S.results_frames(results), "m1": r1,
                                  "n_records": len(recs), "n_flagged": int(r1["records"]["flag"].sum())}
                    ss["custom"] = {k: (sc, S.run_scenario(ctx, sc)) for k, (sc, _) in ss["custom"].items()}
                st.rerun()
    if D["source"] != "synthetic":
        st.success(t("data_done", lang, n=D["n_records"], f=D["n_flagged"], k=D["n_records"] - D["n_flagged"]))
        recs = D["m1"]["records"]
        flagged = recs[recs["flag"]]
        reasons = pd.Series({
            "duplicate (name/birth date or national ID)": int(m1.record_matching(recs).sum()),
            "Isolation Forest anomaly": int(recs["iforest_flag"].sum()),
            **{k: int(v) for k, v in m1.rule_flags(recs)[["past_retirement", "no_attendance", "off_scale_salary"]].sum().items()},
        }, name="records")
        st.subheader(t("data_flag_reasons", lang))
        st.dataframe(reasons, width="stretch")
        st.subheader(t("data_flagged_rows", lang))
        st.dataframe(flagged.sort_values("anomaly_score", ascending=False).head(500), width="stretch")
        if st.button(t("data_reset", lang)):
            ss["data"] = synthetic_state()
            ss["custom"] = {k: (sc, S.run_scenario(ss["data"]["ctx"], sc)) for k, (sc, _) in ss["custom"].items()}
            st.rerun()

# ---------- export ----------
with tabs[7]:
    st.write(t("export_intro", lang))
    base = D["base"]
    scen = dict(C.SCENARIOS) | {k: sc for k, (sc, _) in ss["custom"].items()}
    source = ("Synthetic data shaped like HRMIS, Budget Law, NSSF-C and macro inputs"
              if D["source"] == "synthetic" else f"Uploaded payroll file {D['source']}")
    extra = {}
    if D["m1"] is not None:
        extra["M1 flagged records"] = D["m1"]["records"][D["m1"]["records"]["flag"]].drop(columns=["bank_account"], errors="ignore")
    xlsx = export.workbook(F, scen, D["runs"], source, breakdown.by(base, "ministry"), breakdown.by(base, "province"), extra)
    st.download_button(t("export_xlsx", lang), xlsx, file_name="fiscal_simulation_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.download_button(t("export_csv", lang), table.to_csv().encode(), file_name="scenario_summary.csv", mime="text/csv")
    st.caption(t("export_charts", lang))
