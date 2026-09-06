"""Streamlit dashboard: monitors alt-data signals ahead of a DDOG earnings print.

Panels:
  1. Signal monitor        - each signal's level and YoY change
  2. Revenue nowcast       - guidance midpoint + Datadog's stable historical
                             beat, with the tracking-vs-consensus call
  3. Customer-count nowcast - the alt-data model's estimate, its baseline, and
                             the bootstrap CI on its backtest skill
  4. Signal synthesis      - the intended multi-signal combination method
                             (inverse-variance weights), currently collapsing
                             to one signal because only one survived screening
  5. Backtest track record - walk-forward error metrics, both windows

The model (which signal, lag, target) lives in analysis/backtest.py and is
imported so this dashboard always reflects what that script tests.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from config import TARGET_QUARTER  # noqa: E402
from backtest import (  # noqa: E402
    MIN_TRAIN,
    MODELS,
    SIGNAL,
    SIGNAL_LAG,
    TARGET,
    WINDOWS,
    compute_metrics,
    predict_ar1_plus_signal,
    predict_random_walk,
    walk_forward,
)

PANEL_PATH = ROOT / "data" / "processed" / "quarterly_panel.csv"
MODEL_NAME = f"ar1+{SIGNAL}"

st.set_page_config(page_title="DDOG alt-data nowcast", layout="wide")
st.title("Datadog (DDOG): pre-earnings alt-data monitor")
st.caption("Signals are point-in-time (data through quarter-end only). Not investment advice.")

if not PANEL_PATH.exists():
    st.warning("No data/processed/quarterly_panel.csv found. Run src/build_panel.py first.")
    st.stop()

raw_panel = pd.read_csv(PANEL_PATH, index_col=0)
panel = raw_panel.copy()
panel.index = pd.PeriodIndex(panel.index, freq="Q")
panel[SIGNAL] = panel[SIGNAL].shift(SIGNAL_LAG)

target_quarter = st.sidebar.text_input("Quarter being nowcast", TARGET_QUARTER)
target_period = pd.Period(target_quarter, freq="Q")
signal_cols = [c for c in raw_panel.columns if c.endswith("_yoy")]
default_signals = [c for c in (SIGNAL, "npm_dd_trace_yoy", "gt_datadog_yoy") if c in signal_cols]
selected_signals = st.sidebar.multiselect("Signals", signal_cols, default=default_signals)

# --- 1. Signal monitor --------------------------------------------------------
st.subheader("1. Signal monitor")
columns = st.columns(max(1, len(selected_signals)))
for column, signal in zip(columns, selected_signals):
    series = raw_panel[signal].dropna()
    if series.empty:
        continue
    latest, prior = series.iloc[-1], series.iloc[-2] if len(series) > 1 else series.iloc[-1]
    column.metric(signal, f"{latest:,.1f}", f"{latest - prior:+.1f} vs. prior quarter", delta_color="off")
    column.line_chart(series)
st.caption(
    "Each value is the quarter-to-date YoY change. Update cadence: npm downloads "
    "daily (~1-day lag), Google Trends weekly (~2-day lag), hyperscaler cloud "
    "revenue quarterly (~4 weeks after each filing). Panels 2-5 recompute whenever "
    "src/build_panel.py is rerun; the KPI targets update once per quarter at earnings."
)

# --- 2. Revenue nowcast vs. guidance ----------------------------------------
st.subheader("2. Revenue nowcast (guidance-anchored)")
beat = panel["guidance_beat_pct"].dropna()
guide_mid = panel.loc[target_period, "revenue_guidance_mid_usd_m"] if target_period in panel.index else np.nan
if pd.isna(guide_mid) or beat.empty:
    st.info("No revenue guidance on file for this quarter.")
else:
    mean_beat = beat.mean()
    year_ago_rev = panel.loc[target_period - 4, "revenue_usd_m"]
    last_yoy = panel["revenue_yoy_pct"].dropna().iloc[-1]
    guide_plus_beat = guide_mid * (1 + mean_beat / 100)
    random_walk_rev = year_ago_rev * (1 + last_yoy / 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Datadog guidance midpoint", f"${guide_mid:,.0f}M")
    c2.metric("Guidance + historical beat", f"${guide_plus_beat:,.0f}M",
              f"{mean_beat:+.1f}% (mean of {len(beat)} qtrs, sd {beat.std(ddof=1):.1f}pp)", delta_color="off")
    c3.metric("Random-walk (last YoY held)", f"${random_walk_rev:,.0f}M")
    call = "tracking ahead of guidance" if guide_plus_beat > guide_mid * 1.01 else "in line with guidance"
    st.caption(
        f"No tested alt-data signal beat a naive baseline on revenue growth, so the revenue call is "
        f"anchored on guidance. Datadog has beaten its own revenue guidance midpoint every gathered "
        f"quarter (range +{beat.min():.1f}% to +{beat.max():.1f}%). Call for {target_quarter}: **{call}**."
    )

# --- 3. Customer-count nowcast (alt-data model) --------------------------------
st.subheader("3. Customer-count nowcast ($100k+ ARR customers, YoY %)")
train = panel[panel[TARGET].notna()]
if target_period not in panel.index or pd.isna(panel.loc[target_period, SIGNAL]):
    st.warning(f"No signal data available to nowcast {target_quarter}.")
else:
    test_row = panel.loc[target_period]
    nowcast = predict_ar1_plus_signal(train, test_row)
    baseline = predict_random_walk(train, test_row)

    rob = panel[panel.index >= WINDOWS["robustness"]]
    rob = rob[rob[TARGET].notna()]
    model_p = walk_forward(rob, predict_ar1_plus_signal).dropna()
    base_p = walk_forward(rob, predict_random_walk).dropna()
    e_model = (model_p["y_pred"] - model_p["y_true"]).to_numpy()
    e_base = (base_p["y_pred"] - base_p["y_true"]).to_numpy()
    band = float(np.sqrt(np.mean(e_model ** 2)))
    rng = np.random.default_rng(0)  # same seed/size as analysis/robustness.py
    n_oos = len(e_model)
    boot = np.array([
        1 - np.sqrt(np.mean(e_model[i] ** 2)) / np.sqrt(np.mean(e_base[i] ** 2))
        for i in (rng.integers(0, n_oos, n_oos) for _ in range(10_000))
    ])
    skill = 1 - band / np.sqrt(np.mean(e_base ** 2))

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Model nowcast ({MODEL_NAME})", f"{nowcast:+.1f}%")
    c2.metric("Random-walk baseline", f"{baseline:+.1f}%")
    c3.metric("Backtest skill vs. baseline", f"{skill:+.1%}",
              f"95% CI [{np.percentile(boot, 2.5):+.0%}, {np.percentile(boot, 97.5):+.0%}]", delta_color="off")
    if nowcast > baseline + band:
        call = "tracking ahead"
    elif nowcast < baseline - band:
        call = "tracking behind"
    else:
        call = "in line"
    st.caption(
        f"Nowcast {nowcast:.1f}% vs. baseline {baseline:.1f}% (band +/- {band:.1f}). Call: **{call}**. "
        f"The skill CI straddles zero and the signal correlates at every lag in this window -- "
        f"this edge is not statistically established (n={n_oos})."
    )
    history = panel[TARGET].dropna().to_frame("actual")
    history.loc[target_period, "nowcast"] = nowcast
    st.line_chart(history)

# --- 4. Signal synthesis ----------------------------------------------------
st.subheader("4. Signal synthesis")
st.markdown(
    "With multiple validated signals, each producing a nowcast with an out-of-sample "
    "error variance, they would be combined by **inverse-variance weighting**: each "
    "signal's weight is proportional to 1 / its error variance, normalised to sum to 1, "
    "so lower-error signals get more weight."
)
synth = pd.DataFrame({
    "signal": [f"{SIGNAL} -> {TARGET}", "npm_dd_trace_yoy -> revenue_yoy_pct"],
    "out-of-sample RMSE": [round(band, 2), float("nan")],
    "weight": [1.00, 0.00],
    "status": ["survived screening (customer-count target)",
               "did not beat baseline; weight 0"],
})
st.dataframe(synth, hide_index=True)
st.caption("Only one signal survived screening for the customer-count target, so the blend "
           "collapses to a single model. The second row shows where dd-trace would enter if "
           "its revenue backtest had cleared the baseline.")

# --- 5. Backtest track record ---------------------------------------------------
st.subheader("5. Backtest track record")
window_choice = st.selectbox("Window", list(WINDOWS.keys()), index=1)
window_df = panel[panel.index >= WINDOWS[window_choice]]
window_df = window_df[window_df[TARGET].notna()]
results = {name: compute_metrics(*[
    walk_forward(window_df, fn).dropna()[c].values for c in ("y_true", "y_pred")
]) for name, fn in MODELS.items()}
st.dataframe(pd.DataFrame(results).T.round(2))
st.caption(f"Target: {TARGET}. Window: {window_choice} ({WINDOWS[window_choice]} to latest). "
           f"First prediction after {MIN_TRAIN} training quarters. hit_rate compares the sign of "
           f"the predicted quarter-over-quarter change to the actual.")
