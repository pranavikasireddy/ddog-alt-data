"""Generate the three report figures into report/figures/.

  fig1_signal_vs_kpi.png   npm browser-RUM download growth (lagged 1q) against
                           $100k+ ARR customer-count growth, over time.
  fig2_lead_lag.png        correlation of that signal with the target by lag
                           -2..+2, primary and robustness windows side by side.
  fig3_walk_forward.png    walk-forward actual vs. predicted for the
                           ar1+signal model and the random-walk baseline,
                           robustness window.
  fig4_dashboard.png       static rendering of all five dashboard panels,
                           for readers who do not run Streamlit.

Usage:
    python analysis/figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PRIMARY_WINDOW_START, ROBUSTNESS_WINDOW_START, TARGET_QUARTER  # noqa: E402
from backtest import (  # noqa: E402
    MODELS,
    SIGNAL,
    SIGNAL_LAG,
    TARGET,
    compute_metrics,
    predict_ar1_plus_signal,
    predict_random_walk,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "quarterly_panel.csv"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    df = pd.read_csv(PANEL, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


def fig1_signal_vs_kpi(df: pd.DataFrame) -> None:
    d = df.copy()
    d["signal_lagged"] = d[SIGNAL].shift(SIGNAL_LAG)
    d = d[[TARGET, "signal_lagged"]].dropna()
    x = d.index.to_timestamp()

    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(x, d[TARGET], color="#1f4e79", marker="o", ms=3, label="$100k+ ARR customer-count growth (YoY %)")
    ax1.set_ylabel("customer-count growth (YoY %)", color="#1f4e79", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(x, d["signal_lagged"], color="#c55a11", marker="s", ms=3, ls="--",
             label="browser-RUM download growth, lagged 1q (YoY %)")
    ax2.set_ylabel("RUM download growth, lagged 1q (YoY %)", color="#c55a11", fontsize=8)
    ax2.tick_params(labelsize=7)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="upper right")
    ax1.set_title("Signal vs. KPI over time", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_signal_vs_kpi.png", dpi=110)
    plt.close(fig)


def fig2_lead_lag(df: pd.DataFrame) -> None:
    lags = list(range(-2, 3))
    windows = {"primary (2023Q1+)": PRIMARY_WINDOW_START,
               "robustness (2021Q1+)": ROBUSTNESS_WINDOW_START}
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    width = 0.38
    for i, (label, start) in enumerate(windows.items()):
        w = df[df.index >= start]
        rs = []
        for lag in lags:
            j = pd.concat([w[SIGNAL].shift(-lag), w[TARGET]], axis=1).dropna()
            rs.append(stats.pearsonr(j.iloc[:, 0], j.iloc[:, 1])[0])
        ax.bar(np.arange(len(lags)) + (i - 0.5) * width, rs, width, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([f"{l:+d}" for l in lags])
    ax.set_xlabel("lag (quarters; negative = signal leads)", fontsize=8)
    ax.set_ylabel("Pearson r", fontsize=8)
    ax.set_title("Lead-lag: RUM downloads vs. customer-count growth", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_lead_lag.png", dpi=110)
    plt.close(fig)


def fig3_walk_forward(df: pd.DataFrame) -> None:
    d = df.copy()
    d[SIGNAL] = d[SIGNAL].shift(SIGNAL_LAG)
    d = d[d[TARGET].notna()]
    d = d[d.index >= ROBUSTNESS_WINDOW_START]
    model = walk_forward(d, predict_ar1_plus_signal).dropna()
    base = walk_forward(d, predict_random_walk).dropna()
    x = model.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, model["y_true"], color="black", marker="o", ms=4, label="actual")
    ax.plot(x, model["y_pred"], color="#c55a11", marker="s", ms=3, ls="--", label="ar1 + RUM signal")
    ax.plot(x, base["y_pred"], color="#7f7f7f", marker="^", ms=3, ls=":", label="random-walk baseline")
    ax.set_ylabel("customer-count growth (YoY %)", fontsize=8)
    ax.set_title("Walk-forward: actual vs. predicted (robustness window)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_walk_forward.png", dpi=110)
    plt.close(fig)


def _mono(ax, lines, x=0.02, top=0.92, step=0.135, size=8):
    for i, line in enumerate(lines):
        ax.text(x, top - i * step, line, fontsize=size, family="monospace",
                va="top", transform=ax.transAxes)


def fig4_dashboard(df: pd.DataFrame) -> None:
    target = pd.Period(TARGET_QUARTER, freq="Q")

    # revenue panel
    beat = df["guidance_beat_pct"].dropna()
    guide = df.loc[target, "revenue_guidance_mid_usd_m"]
    year_ago = df.loc[target - 4, "revenue_usd_m"]
    last_yoy = df["revenue_yoy_pct"].dropna().iloc[-1]
    guide_beat = guide * (1 + beat.mean() / 100)
    rw_rev = year_ago * (1 + last_yoy / 100)

    # customer-count panel
    d = df.copy()
    d[SIGNAL] = d[SIGNAL].shift(SIGNAL_LAG)
    train = d[d[TARGET].notna()]
    nowcast = predict_ar1_plus_signal(train, d.loc[target])
    baseline = predict_random_walk(train, d.loc[target])

    # backtest panel (robustness window)
    rob = d[d[TARGET].notna()]
    rob = rob[rob.index >= ROBUSTNESS_WINDOW_START]
    results = {}
    for name, fn in MODELS.items():
        p = walk_forward(rob, fn).dropna()
        results[name] = compute_metrics(p["y_true"].values, p["y_pred"].values)

    # signal-monitor panel
    monitor = [(c, df[c].dropna()) for c in
               ("npm_browser_rum_yoy", "npm_dd_trace_yoy", "gt_datadog_yoy") if c in df]

    fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.0))
    for ax in axes.flat:
        ax.axis("off")
    (m, rev), (cust, synth), (bt, blank) = axes

    m.set_title("1. Signal monitor (latest YoY %, vs. prior quarter)", fontsize=8, weight="bold", loc="left")
    _mono(m, [f"{name:22s} {s.iloc[-1]:7.1f}   ({s.iloc[-1] - s.iloc[-2]:+.1f})"
              for name, s in monitor])

    rev.set_title(f"2. Revenue nowcast, {TARGET_QUARTER}", fontsize=8, weight="bold", loc="left")
    _mono(rev, [
        f"guidance midpoint       ${guide:,.0f}M",
        f"guidance + mean beat     ${guide_beat:,.0f}M ({beat.mean():+.1f}%)",
        f"random walk (last YoY)   ${rw_rev:,.0f}M",
        "call: tracking ahead of guidance",
        "(no alt-data edge; anchored on the stable beat)",
    ])

    cust.set_title(f"3. Customer-count nowcast, {TARGET_QUARTER}", fontsize=8, weight="bold", loc="left")
    _mono(cust, [
        f"model (ar1 + RUM signal)  {nowcast:+.1f}%",
        f"random-walk baseline      {baseline:+.1f}%",
        "backtest skill  +15.4%  (95% CI [-32%, +38%])",
        "call: in line",
        "(not established; n=12, signal co-trending)",
    ])

    synth.set_title("4. Signal synthesis", fontsize=8, weight="bold", loc="left")
    _mono(synth, [
        "method: inverse-variance weighting,",
        "  w_i proportional to 1 / OOS error variance_i",
        "",
        f"  {SIGNAL:22s}  weight 1.00",
        "  npm_dd_trace_yoy         weight 0.00 (failed screen)",
        "",
        "one signal survived, so the blend is one model",
    ], size=6.5)

    bt.set_title("5. Backtest track record (robustness window)", fontsize=8, weight="bold", loc="left")
    hdr = f"{'model':24s} {'MAPE':>6s} {'RMSE':>6s} {'hit':>5s}"
    rows = [hdr] + [
        f"{name[:24]:24s} {v['MAPE']:6.1f} {v['RMSE']:6.2f} {v['hit_rate']:5.2f}"
        for name, v in results.items()
    ]
    _mono(bt, rows, size=6.3)

    fig.suptitle("Dashboard (dashboard/app.py) -- static rendering of all five panels", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig4_dashboard.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if not PANEL.exists():
        raise SystemExit("Run src/build_panel.py first.")
    df = load()
    fig1_signal_vs_kpi(df)
    fig2_lead_lag(df)
    fig3_walk_forward(df)
    fig4_dashboard(df)
    print(f"wrote 4 figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
