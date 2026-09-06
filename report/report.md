# Datadog (DDOG): Alternative-Data Earnings Nowcast

## Executive Summary

The question: can free, public alternative data help forecast Datadog's quarterly results before it reports? Five sources were proposed and characterized (Section 1); the two with the strongest case, npm download counts for Datadog's browser RUM SDK and for `dd-trace`, were tested with a walk-forward backtest against Datadog's KPIs.

**Revenue growth: no signal, but a clean rule.** Ratable revenue recognition makes quarterly revenue growth so smooth that "assume no change" is hard to beat, and neither signal beat it. The revenue call is instead anchored on **Datadog's own guidance**, which the company has come in above every one of the 13 quarters checked, by a tight +3.9% (sd 0.9pp). Backtested walk-forward over 9 quarters, "guidance midpoint x trailing mean beat" nowcasts revenue to **0.6% MAPE** and calls the direction of the growth change 7 times in 9, beating a naive constant-growth carry on both.

**$100k+ ARR customer-count growth: a weak signal, not a result.** RUM download growth lagged one quarter, added to the metric's own momentum, cut forecast error by 15.4% versus the baseline in the longer test window. But that figure is fragile: its 95% confidence interval is [-32%, +38%], and in that window the signal correlates with the target at *every* lag, the co-movement pattern that also ruled out cloud revenue. Read it as suggestive, not established.

**2026Q3 nowcast.** Revenue: ~$1,184M (guidance midpoint $1,140M plus the historical beat), *modestly ahead of guidance*. Customer-count growth: +22.5% vs. a +22.6% baseline, *in line*.

## 1. Data Selection and Rationale

### 1.1 npm downloads of `@datadog/browser-rum` (tested)

- **Definition & Origin.** Daily download counts for Datadog's Real User Monitoring SDK, from npm's official public API (`api.npmjs.org`). RUM is embedded in production web front-ends to monitor real user sessions and client-side performance.
- **Economic Rationale.** RUM is deployed by customer-facing, production-scale applications. When such an account scales up, it crosses the $100k ARR line before that shows up in reported revenue: the 10-K says customers who exceed their committed contract "are charged for their incremental usage," billed in arrears and recognized as delivered rather than ratably, so usage growth leads the revenue it produces. Management attributes recent ARR growth to "usage growth from existing customers" (CFO, Q1 2026 call). Figure 1 shows RUM downloads and customer-count growth moving together.
- **Data Characteristics.** Daily; ~1-day latency; history from 2016; breadth limited to one SDK for one product line.
- **Accessibility & Cost.** Free, unauthenticated API; npm registry terms of use.
- **Limitations & Mitigations.** Counts include CI re-installs and bots (inflate the level) -> use YoY change, not the raw count. A near-identical sibling signal, Logs SDK downloads, does *not* reproduce the backtest result -> the finding is specific to RUM, not "browser SDKs."

### 1.2 npm downloads of `dd-trace` (tested)

- **Definition & Origin.** Daily download counts for Datadog's Node.js APM tracer, same npm API. Each install instruments an application to send distributed traces, billed per host and per span.
- **Economic Rationale.** An install is a developer adding Datadog APM to a service. APM is metered per host (the 10-K: "a host is generally defined as a server"), so a newly instrumented service that reaches production adds billable hosts, with overage billed in arrears. Adoption should therefore show up in revenue one to two quarters after the install. APM is one of Datadog's largest product lines.
- **Data Characteristics.** Daily; ~1-day latency; history from 2016; breadth limited to the Node.js runtime (Python, Java, Go, .NET tracers not tested).
- **Accessibility & Cost.** Same free npm API.
- **Limitations & Mitigations.** Same CI/bot inflation, mitigated the same way. The lead-lag correlation with revenue growth is real and BH-significant, but did not convert to walk-forward skill (Section 2.3).

### 1.3 Three further sources: proposed and characterized, not carried into the backtest

| Dimension | Google Trends (brand + intent terms) | Hyperscaler cloud revenue (AWS, GCP) | Hiring data (open engineering roles at Datadog and at its customers) |
|---|---|---|---|
| Definition & Origin | Weekly 0-100 search-interest index, `pytrends` API, for "Datadog", "Datadog pricing", competitor comparisons | Quarterly AWS and Google Cloud segment revenue, from Amazon and Alphabet SEC filings | Time series of posting counts; free history via Wayback Machine snapshots, precise history via vendors (Revelio, LinkUp) |
| Economic Rationale | Search interest is the top of the adoption funnel and has been shown to nowcast contemporaneous economic activity (Choi & Varian 2012); usage-priced revenue follows adoption | The 10-K states revenue "is closely correlated with customers' cloud workloads"; management attributed the 2023 revenue slowdown directly to customers "optimizing" their cloud usage (2023 earnings calls), so hyperscaler consumption growth is the upstream driver | Hiring is a forward commitment; Datadog discloses total headcount annually in its 10-K (a hiring-intensity series in itself) and guides to continued go-to-market hiring on earnings calls, and "Datadog" as a required skill in other firms' postings signals market adoption |
| Data Characteristics | Weekly, ~2-day latency, 2018+, worldwide | Quarterly, ~4-week latency, long history, whole-market | Wayback: irregular, patchy; vendors: weekly, deep history |
| Accessibility & Cost | Free (unofficial API) | Free (filings) | Wayback free; vendors paid |
| Limitations & Mitigations | Index is re-normalized on every pull (vintage risk) -> use YoY change; **in testing, the sign flipped between windows for every term** | Only quarterly, so coincident not leading; **correlated with DDOG revenue at every lag, i.e. a shared macro trend, not a lead-lag relationship** | Wayback: the archived careers page is a client-rendered app backed by a live search index, so the HTML contains no listings -> **free historical data is infeasible; a paid vendor feed is the mitigation** |

During development, ~13 further signals (developer-forum activity, GitHub, insider filings, short interest, Terraform release cadence, Wikipedia pageviews) were also scanned across multiple targets and lags; none survived a two-window consistency check. That scan is not part of this submission -- with that many comparisons it is hypothesis generation, not evidence -- but it is the reason the multiple-testing framing in Section 2 is conservative.

### 1.4 AI-tooling adoption (proposed; not yet enough history to test)

The most economically relevant emerging driver is AI-workload observability: the FY2025 10-K attributes "approximately seven percentage points" of YoY revenue growth to AI-native customers, and management flagged MCP tool-call volume "up 22x since Q4 2025" (Q2 2026 call). Two free series track this adoption: npm downloads of `datadog-mcp-server` (roughly 1.8k rising to 16.7k quarterly through 2026Q2) and Google Trends for "LLM observability" (about 2 to about 85 between 2025Q1 and 2026Q2). Both move in the direction the disclosures imply, but this is **not carried into the analysis**: only ~4-6 quarters exist, all post-2024, and Datadog discloses no quarterly AI-ARR series to regress against. It is flagged as the first signal to test as history accrues (Section 5), not evidence now.

## 2. Statistical Analysis and Predictive Evidence

**Methods note.** Terms used below, with the bar each has to clear:

- **Walk-forward (expanding window).** At each quarter, the model is fit on all earlier quarters only, then predicts the next one. No future data enters any fit.
- **MAPE / RMSE.** Mean absolute percentage error and root-mean-square error of the point forecasts, over the walk-forward predictions.
- **Directional (accel/decel) hit rate.** Share of quarters where the forecast got the sign of the quarter-on-quarter change right.
- **Skill score.** `1 - RMSE_model / RMSE_baseline`. Positive means smaller error than the baseline; +15% means 15% less RMSE. Baseline is the best no-signal model (usually the random walk).
- **Bootstrap CI.** Resample the out-of-sample quarters with replacement 10,000 times (seed fixed), recompute the skill score each time; the reported interval is the 2.5th to 97.5th percentile. An interval spanning zero means the edge is not distinguishable from noise.
- **Diebold-Mariano.** Formal test of whether one forecast's squared errors are significantly smaller than another's, one-sided, with the Harvey-Leybourne-Newbold small-sample correction. p < 0.05 to reject "no difference."
- **Benjamini-Hochberg (BH).** Controls the false-discovery rate when many correlations are tested at once; q is the adjusted p-value, and q < 0.10 is the bar here.

### 2.1 Targets and windows

Four targets had enough consistently-disclosed history for a genuine test, each checked in a **primary window** (2023Q1+) and an independently-sampled **robustness window** (2021Q1+, verified to use different underlying quarters, not a superset artifact):

| Target | Primary n | Robustness n | Result |
|---|---|---|---|
| Revenue growth | 15 | 23 | no signal beat the baseline |
| $100k+ ARR customer-count growth | 14 | 18 | one signal: weak edge (2.3) |
| RPO growth | 10 | 12 | no signal held a consistent sign |
| Net new ARR (dollar change) | 14 | 21 | no signal held up across windows |

Five metrics named in the brief or common in SaaS analysis -- billings growth, total customer count growth, $1M+ ARR customer count growth, current RPO (cRPO) growth, 4+-product adoption growth -- had only 1-6 valid year-over-year data points each in freely-available disclosure and are reported as **inconclusive for lack of data**, distinct from "tested and found nothing."

**Net revenue retention (NRR)** deserves a fuller note, since the brief names it explicitly. Datadog stopped disclosing an exact NRR figure in 2023 and now gives only a qualitative range ("the low 120s%") that has barely moved for two years. There is therefore no usable quarterly variation in free data to correlate against. Cracking it would require paid data that observes the mechanism directly: card-panel spend by customer cohort, or product-level expansion telemetry from a vendor panel.

### 2.2 Lead-lag analysis

Each signal, using only data through quarter-end, was cross-correlated with each target at lags -2 to +2. Negative lags test whether the signal leads; positive lags are a control, since a real leading signal should not correlate as strongly with the target's *future* as with its past. Two signals cleared that check:

- **RUM downloads vs. customer-count growth.** Correlation peaks at lag -1 in both windows (r = +0.89 primary, +0.97 robustness; Figure 2). But in the robustness window the correlation stays high at *every* lag (0.64 to 0.97), including the implausible positive ones. That is the same co-movement pattern that ruled out cloud revenue: the two series mostly share the 2022-2024 growth arc rather than one cleanly leading the other.
- **`dd-trace` downloads vs. revenue growth.** Positive correlation in both windows (r = +0.27 to +0.89), consistent sign.

Both survive a Benjamini-Hochberg correction over the 20 correlations tested here (q < 1e-3). The development scan (Section 1.3) ran several hundred comparisons where BH leaves nothing, which is why it is not treated as evidence.

### 2.3 Modeling and out-of-sample validation

Each signal was tested in two model forms: OLS on the signal alone, and the signal added to an AR(1) on the target's own lag. Both ran against three no-signal baselines (random walk, seasonal-naive, AR(1)) in an expanding-window walk-forward: at each step the model sees only earlier quarters. The AR(1)+signal form is the fair test, since the target's own momentum is already a strong predictor. All models are scored on MAPE, RMSE, and a directional hit rate. Skill score is `1 - RMSE_model / RMSE_baseline` against the best baseline.

**Revenue growth, `npm_dd_trace_yoy` (lag 2).** This is the model the brief asks for: a proposed signal linked to quarterly revenue growth. `dd-trace` download growth, lagged two quarters, regressed on `revenue_yoy_pct`, walk-forward, both windows:

| Model | Primary skill | Robustness skill |
|---|---|---|
| OLS on `dd-trace` growth | -132.7% | -81.9% |
| AR(1) + `dd-trace` growth | -26.8% | -17.3% |

Every version loses to the naive baseline (negative skill = larger RMSE than a random walk). In the robustness window the AR(1)+signal model called one more quarter's direction right than the baseline (hit rate 0.87 vs. 0.80, one call in ~12), but that is not a magnitude edge and is too small to lean on. The lead-lag correlation from Section 2.2 is real; it does not convert into out-of-sample forecast skill. **Alt-data adds nothing to a revenue-growth forecast here.**

**Customer-count growth, `ar1 + npm_browser_rum_yoy` (lag 1), vs. the best baseline (random walk):**

| Window | OOS n | Baseline RMSE / MAPE / hit | Model RMSE / MAPE / hit | Skill (RMSE) |
|---|---|---|---|---|
| Primary (2023Q1+) | 8 | 1.91 / 10.5% / 0.57 | 3.55 / 16.0% / 0.57 | -85.9% |
| Robustness (2021Q1+) | 12 | 2.47 / 12.3% / 0.55 | 2.09 / 11.0% / 0.55 | **+15.4%** |

The +15.4% does not hold up to scrutiny. A 10,000-sample bootstrap puts the 95% CI at **[-32%, +38%]** (77% of draws positive); a Diebold-Mariano test gives one-sided p = **0.24** (and an uncorrected DM is optimistic, so the true p is higher). It also comes entirely from the longer window: the primary window scores -85.9%, because its walk-forward does not start until ~2024Q3 and so never tests the 2022-2024 arc where the model's advantage lies. Combined with the flat-lag correlation above, the evidence is one shared arc fit once, not twelve independent wins.

**Revenue nowcast: the guidance fallback.** With no signal beating the baseline on revenue, the delivered revenue nowcast is a rule rather than an alt-data model: next-quarter revenue = guidance midpoint x (1 + trailing mean beat), the mean beat re-estimated each quarter from only the beats known by then. Walk-forward over the 9 quarters where a trailing beat could be formed (2024Q1 to 2026Q2), scored on reported revenue in dollars and on the direction of the YoY-growth change:

| Revenue rule | MAPE | RMSE | Accel/decel hit |
|---|---|---|---|
| Guidance midpoint x trailing mean beat | 0.6% | $6.3M | 0.78 |
| Naive: same YoY growth as last quarter | 0.9% | $12.7M | 0.00 |

The guidance rule wins on all three. The naive carry's 0.00 direction hit is structural: it assumes growth never changes, so it cannot call the accelerations and decelerations the 2024-2026 window contains. This is the one clean out-of-sample result on the primary KPI, and it needs no alt-data.

![Figure 1: signal vs. KPI over time](fig1_signal_vs_kpi.png)

![Figure 2: lead-lag correlation by lag, both windows](fig2_lead_lag.png)

![Figure 3: walk-forward actual vs. predicted, robustness window](fig3_walk_forward.png)

### 2.4 2026Q3 nowcast (Figure 4)

**Revenue.** With no working signal, the call is the guidance rule backtested in Section 2.3 (0.6% MAPE). Datadog's Q3 2026 guidance midpoint is **$1,140M**, and actual revenue has come in above the midpoint in all 13 quarters checked (2023Q1 to 2026Q2; 2025Q2 could not be cleanly sourced and was left out rather than estimated), by **+3.9%** (sd 0.9pp, range +1.9% to +5.2%). Applying that beat gives **~$1,184M** (about $1,174M to $1,194M at +/-1 sd); the naive "same YoY growth" method gives ~$1,204M. Both sit above the midpoint: **tracking modestly ahead of guidance**.

**Customer-count growth.** Model nowcast **+22.5% YoY** vs. a **+22.6%** random-walk baseline (band +/-2.1 points): **in line**. On the year-ago base of 4,060, that implies roughly **4,970** customers with $100k+ ARR.

![Figure 4: dashboard nowcast panels, static rendering](fig4_dashboard.png)

## 3. Dashboard Design

A runnable Streamlit prototype (`dashboard/app.py`) with five panels, wired to the fitted model rather than static figures:

1. **Signal monitor** -- quarter-to-date YoY change for each tracked signal. Update cadence: npm downloads daily (~1-day lag), Google Trends weekly (~2-day lag), hyperscaler cloud revenue quarterly (~4 weeks after each filing). A full refresh is a same-day operation.
2. **Revenue nowcast** -- guidance midpoint, guidance-plus-historical-beat, and random-walk estimates, with the tracking-vs-guidance call.
3. **Customer-count nowcast** -- the model estimate, its baseline, and the bootstrap CI on its backtest skill, so the panel cannot overstate the edge.
4. **Signal synthesis** -- the intended combination method: inverse-variance weighting, where each signal's nowcast is weighted proportionally to 1 / its out-of-sample error variance. The worked table's weights currently collapse to one signal because only one survived screening.
5. **Backtest track record** -- the full MAPE / RMSE / hit-rate table for every model and baseline, selectable by window.

Interpretation guide: "tracking ahead" = nowcast above the reference (guidance midpoint for revenue, baseline for customer count) by more than one RMSE; "behind" = more than one RMSE below; "in line" otherwise.

## 4. Limitations

- The one alt-data signal that showed an edge (RUM downloads vs. customer-count growth) is not statistically established (CI [-32%, +38%], DM p = 0.24), rests on one 2022-2024 arc scored once, and comes from the window where the signal correlates at every lag (largely shared trend, not clean lead-lag). It should be read as a weak, not-yet-credible signal. The clean out-of-sample result is the revenue guidance rule (Section 2.3), which uses no alt-data.
- Sample sizes are small throughout (n = 8-23), a structural consequence of Datadog's short public history. Only the two tested pairs are treated as evidence, and even those only for the correlation, not the forecast.
- Google Trends is re-normalized on every pull (vintage risk); npm signals cover specific SDKs, not the whole business; three of the historical $100k-customer counts were read from a later quarter's YoY disclosure (noted per row in `data/ddog_kpis.csv`).
- Revenue guidance was gathered for 13 of the 14 relevant quarters (2025Q2 unverified); the beat statistic is tight and uniformly positive across them.
- All data is public or from official free APIs; no MNPI; data-source terms respected.

## 5. Further Work

Paid data would address the two binding constraints: a vendor feed for consistent quarterly NRR / cRPO / hiring, and historical analyst-consensus estimates so the benchmark is "beat the Street," not "beat a random walk." The priority new signal is AI-tooling adoption (Section 1.4): keep logging `datadog-mcp-server` npm downloads and "LLM observability" / "AI observability" search interest each quarter, and once ~8 quarters exist, test them against revenue growth and against the AI-native cohort's contribution to growth as newly disclosed each year. On method: a properly cross-validated multi-signal model once more history exists to fit one without overfitting.

---

*References for the alt-data nowcasting approach: Choi, H. & Varian, H. (2012), "Predicting the Present with Google Trends," Economic Record 88, 2-9. Kolanovic, M. & Krishnamachari, R. (2017), "Big Data and AI Strategies," J.P. Morgan Global Quantitative & Derivatives Strategy.*
