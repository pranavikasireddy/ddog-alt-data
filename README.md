# Datadog (DDOG) Alternative-Data Earnings Nowcast

Proposes five free alternative-data sources for nowcasting Datadog's quarterly
results, tests two of them against Datadog's KPIs with walk-forward
validation, and monitors the result on a live dashboard. Target quarter:
**2026Q3**. The finished write-up is `report/report.pdf` (`report/report.md`
is the source).

## Layout
```
templates/       cloud_revenue_template.csv, filled by hand from filings
src/             config.py, fetch_google_trends / fetch_npm_downloads / fetch_cloud_revenue, build_panel.py
analysis/        lead_lag.py, backtest.py, robustness.py, figures.py
dashboard/       app.py, a Streamlit prototype wired to the fitted model
report/          report.md + report.pdf + build_pdf.py + figures/
data/            ddog_kpis.csv, ddog_guidance.csv (targets); raw/ and
                 processed/ pulls are committed so analysis reruns offline
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline
```bash
# 1. signals (raw pulls are committed; rerun only to refresh)
python src/fetch_google_trends.py
python src/fetch_npm_downloads.py
python src/fetch_cloud_revenue.py       # reads templates/cloud_revenue_template.csv

# 2. assemble (reads data/ddog_kpis.csv and data/ddog_guidance.csv)
python src/build_panel.py

# 3. analysis
python analysis/lead_lag.py     # cross-correlation scan, both windows, all targets
python analysis/backtest.py     # walk-forward model vs. baselines
python analysis/robustness.py   # skill-score CI, Diebold-Mariano, BH, revenue-rule backtest, guidance beat
python analysis/figures.py      # the four report figures

# 4. report + dashboard
python report/build_pdf.py
streamlit run dashboard/app.py
```

## Sources

**Proposed and characterized (report Section 1):** npm downloads of
`@datadog/browser-rum`; npm downloads of `dd-trace`; Google Trends brand and
intent terms; hyperscaler cloud revenue (AWS, Google Cloud); hiring data
(open engineering roles -- characterized only; no free historical feed). A
sixth, AI-tooling adoption (`datadog-mcp-server` npm downloads + "LLM
observability" search interest), is characterized in Section 1.4 but has only
~4-6 quarters of history, so it is flagged for future quarters rather than
tested now.

**Tested (2 of the 5):**

| Signal | Result |
|---|---|
| `@datadog/browser-rum` downloads, lag 1, vs. $100k+ ARR customer-count growth | +15.4% RMSE skill over baseline in the longer window, but the 95% CI is [-32%, +38%] (Diebold-Mariano p = 0.24) and the signal correlates at every lag in that window -- largely co-trending, not statistically established |
| `dd-trace` downloads, lag 2, vs. revenue growth | real BH-significant lead-lag correlation, but never beats the baseline in the backtest |

Since no signal beats a naive baseline on revenue, the revenue nowcast is a
rule -- guidance midpoint x (1 + trailing mean beat) -- backtested walk-forward
over 9 quarters at 0.6% MAPE / $6.3M RMSE, ahead of a naive constant-growth
carry on error and on direction (report Section 2.3).

The other three proposed sources did not survive testing (Trends: sign flips
between windows; hyperscaler revenue: correlated at every lag, a shared
trend; hiring: no free historical data). During development a further ~13
signals were scanned across multiple targets/lags; none survived, and with
that many comparisons it is hypothesis generation, not evidence, so it is
excluded from this submission (report Section 1.3).

## Ground rules
Public or official-API data only, no material non-public information. Raw
pulls keep their fetch-date filenames since Google Trends re-normalises on
every pull. The one positive finding is stated with its confidence interval
and its co-trending caveat, not as a clean win.
