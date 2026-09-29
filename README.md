# AI fiscal simulation for Cambodia's civil service pay and pensions

This is a working prototype of the model in the thesis proposal *AI-Driven Fiscal Simulation for Civil
Service Salary and Pension Reform in Cambodia*. It builds modules M1 to M6 from the architecture
diagram and runs policy scenarios S0 to S6 with 10,000 Monte Carlo runs.

> **The data is synthetic.** HRMIS, payroll and NSSF-C microdata were not available, so
> `fiscalsim/synthetic.py` generates records with the same fields and structure (the proposal's
> synthetic fallback, section 4.2.1). Every calibration value sits in `fiscalsim/config.py`, and each one
> marked `# SOURCE:` should be replaced with the official figure before the results are used in the thesis.

## Quick start

```bash
pip install -r requirements-dev.txt     # or requirements.txt to skip Prophet and PyTorch
python -m fiscalsim.run                 # about 90 seconds: data, M1-M6, S0-S6 x 10,000 runs
streamlit run app.py                    # dashboard
pytest -q                               # 10 fast checks
```

`python -m fiscalsim.run --runs 1000 --regenerate` gives a quicker run and rebuilds the synthetic data.
The results are in `outputs/report.md`, the CSVs are in `outputs/`, and the charts are in `outputs/charts/`.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with your GitHub account and choose **Create app**, then **Deploy a public app from GitHub**.
2. Pick this repository, branch `main` and main file `app.py`. Under **Advanced settings**, choose Python 3.11.
3. Click **Deploy**. `requirements.txt` holds the pinned app dependencies. Prophet and PyTorch are left out to keep the build small, and the app doesn't need them.
4. Under the app's **Settings > Sharing**, limit who can view it before anyone uploads real payroll data.

The committed `outputs/dashboard_bundle.pkl` holds the published 10,000-run results, so the app starts straight away. If it's missing, the app builds it on first launch with 2,000 runs, which takes a few minutes. Rebuild it locally after changing the model (`python -m fiscalsim.run`) and commit the new file.

Uploaded files stay in the viewer's app session and are never written to disk or to the repository.

## Dashboard features

| Tab | What it does |
|---|---|
| Compare scenarios | Fan charts with 90% bands for S0 to S6 and any saved custom scenarios, plus the comparison table at 2036, 2046 or 2076 |
| Build a scenario | Set every policy lever, see the result live against S0, and save up to three custom scenarios to compare and export |
| AI reform optimiser | Searches retirement age, pension formula, accrual rate and contribution rises (optionally pay policy and hiring) for the best trade-offs between fiscal cost and pension adequacy, recommends the cheapest package meeting a replacement-rate target, and saves it as a custom scenario |
| Adequacy and workforce | Replacement rates, the share of retirees on the minimum pension, and headcount |
| Ministries and provinces | Base-year headcount, age, wage bill and M1 flags by ministry or province, with the projected wage bill under any scenario |
| What drives results | SHAP charts for policy levers, economic uncertainty and the exit model |
| Model checks | M1, M2 and M5 backtests and the Kaplan-Meier curves |
| Your data | Upload your own payroll extract (CSV or Excel) and, optionally, the pensioner register. M1 cleans it and every scenario re-runs on it. You can download an example file |
| Export | An Excel workbook with the summary, yearly bands, adequacy, workforce, sector, ministry and province tables, levers and flagged records, or the summary as a CSV |

The language switch in the sidebar changes the app between English and Khmer. The Khmer labels in `fiscalsim/i18n.py` are a first translation, so please check the terms against MEF, MCS and NSSF usage.

Ministry and province projections give each unit its base-year share of its sector's projected wage bill, because the engine projects by sector.

## How the modules map to the architecture

| Module | File | What it does | Baseline it is compared with |
|---|---|---|---|
| Inputs | `synthetic.py` | A 2012-2025 staff panel, a 2026 HRMIS extract with injected duplicates, records with unconfirmed identity or attendance, and off-scale salaries, the NSSF-C pensioner register, 1995-2025 macro series and a mortality table | |
| M1 Data quality | `m1_data_quality.py` | Record matching on name, birth date and national ID, plus Isolation Forest. Flagged records are set aside for verification before the base year is built. A flag is a prompt to check the record, not a finding of wrongdoing | Rule-based checks only |
| M2 Workforce | `m2_workforce.py` | Kaplan-Meier curves, a discrete-time logit hazard and an XGBoost exit model, with a Markov promotion matrix. It projects headcount by sector, framework, age and service | Cohort-ratio method |
| M3 Salary engine | `m3_salary.py` | Pay-scale rules (framework entry pay x service steps x salary index, plus allowances) under fixed, inflation-indexed or targeted raises | Fixed-growth assumption (S0) |
| M4 Pension and fiscal | `m4_pension.py` | A cohort actuarial model covering benefits, CPI or pay indexation, mortality, contributions, the fund balance, depletion and government top-up, across 10,000 paths | Deterministic central path (run 0) |
| M5 Macro forecast | `m5_macro.py` | Naive, 8-year mean, linear trend, ARIMA, Prophet and LSTM models are backtested on 2018-2025. The best one feeds the central path, which then converges to long-run anchors, and AR(1) shocks generate the Monte Carlo paths | Naive and linear trend |
| M6 Dashboard and SHAP | `m6_explain.py`, `app.py` | SHAP for the exit model, for Monte Carlo uncertainty drivers and for policy levers against uncertainty. The Streamlit dashboard compares scenarios and builds custom ones live | Static tables (`report.md`) |
| Reform optimiser | `optimize.py` | Surrogate-assisted multi-objective search: Gaussian-process surrogates of the simulator pick which reform packages to score next, and the Pareto front of cost against adequacy comes from simulator runs only. `python -m fiscalsim.run --grid-check` scores every package to check the search against brute force | Scoring every package in the grid |
| Policy scenarios | `config.py` | Levers for S0 to S6 as in proposal Table 5 | |

`simulate.py` wires M2 to M4 together. All scenarios share the same random paths, so differences between
scenarios come from policy rather than sampling noise. `run.py` is the command-line pipeline.

## Modelling choices to review

- **Current pension rule** is 80% of the final basic salary after 20 or more years of service (the ILO finding that 20 and 30 years pay the same), with a minimum pension that moves with pay. Under 20 years of service, the retiree gets a lump sum.
- **Accrual rule (S4, S5)** is 2% per year of service x the average basic salary over the last 10 years, revalued, with an 80% cap. Change `accrual_rate` or `cap_replacement` to test variants.
- **Contributions** are 6% from the employee and 12% from the employer, on basic salary. S5 adds 1 point to each side every 5 years, up to 3 points. These rates are placeholders.
- **Total cost** means the wage bill plus the government's pension cost, which is the employer contribution plus any top-up after the fund is exhausted.
- **The long-run anchors** are 4.5% real growth, 3% inflation and revenue at 24% of GDP. Forecasts converge to them over 15 years.
- **Not modelled yet**: survivor and disability pensions, pensions for early leavers, armed forces and police, and Khmer LLM summaries (an optional part of M6).

## Moving to real data

1. Replace `data/hrmis_2026.csv` with the aggregated or anonymised MCS extract, using the same columns. Replace `staff_history.parquet` with the yearly HRMIS snapshots.
2. Put the Budget Law pay scales, NSSF-C rules and NIS, NBC and IMF series into `config.py` and `data/macro_history.csv`.
3. Rerun `python -m fiscalsim.run` without `--regenerate`, then check the base-year wage bill against the Budget Law total. That check is the model-validity test in proposal Table 6.
