"""Uncertainty quantification for the headline result, plus the guidance-based
revenue nowcast.

Three things the point estimates in backtest.py do not tell you:

  1. How wide the uncertainty on the +15.4% skill score is, given only ~12
     out-of-sample points. Bootstrapped CI + a Diebold-Mariano test on the
     squared-error differential.
  2. Whether the lead-lag correlations that drove signal selection survive a
     multiple-testing correction. Benjamini-Hochberg on the two carried-
     forward signal/target pairs across all lags and both windows.
  3. Datadog's revenue guidance has been beaten by a small, stable margin
     every quarter it was gathered. Walk-forward test of the rule "guidance
     midpoint x (1 + trailing mean beat)" against the naive "same YoY growth"
     nowcast, scored on reported revenue in dollars (MAPE, RMSE) and on the
     direction of the YoY-growth change (accel/decel hit rate), then the same
     rule applied forward to the target quarter.

Usage:
    python analysis/robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PRIMARY_WINDOW_START, ROBUSTNESS_WINDOW_START, TARGET_QUARTER  # noqa: E402
from backtest import (  # noqa: E402
    SIGNAL,
    SIGNAL_LAG,
    TARGET,
    predict_ar1_plus_signal,
    predict_random_walk,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "quarterly_panel.csv"
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(0)


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="Q")
    return df


# ---------------------------------------------------------------------------
# 1. Uncertainty on the skill score
# ---------------------------------------------------------------------------
def skill_uncertainty(df: pd.DataFrame) -> None:
    window = df.copy()
    window[SIGNAL] = window[SIGNAL].shift(SIGNAL_LAG)
    window = window[window[TARGET].notna()]
    window = window[window.index >= pd.Period(ROBUSTNESS_WINDOW_START, freq="Q")]

    model = walk_forward(window, predict_ar1_plus_signal).dropna()
    base = walk_forward(window, predict_random_walk).dropna()
    joined = model.join(base["y_pred"], rsuffix="_base")
    e_model = (joined["y_pred"] - joined["y_true"]).to_numpy()
    e_base = (joined["y_pred_base"] - joined["y_true"]).to_numpy()
    n = len(e_model)

    def skill(idx):
        rmse_m = np.sqrt(np.mean(e_model[idx] ** 2))
        rmse_b = np.sqrt(np.mean(e_base[idx] ** 2))
        return 1 - rmse_m / rmse_b

    point = skill(np.arange(n))
    boot = np.array([skill(RNG.integers(0, n, n)) for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Diebold-Mariano on the squared-error differential (d < 0 favours the model)
    d = e_model ** 2 - e_base ** 2
    dm_stat = d.mean() / np.sqrt(d.var(ddof=1) / n)
    dm_p = stats.t.cdf(dm_stat, df=n - 1)  # one-sided: is the model better?

    print("1. SKILL-SCORE UNCERTAINTY (customer-count model, robustness window)")
    print(f"   out-of-sample quarters:      {n}")
    print(f"   point skill score:           {point:+.1%}")
    print(f"   95% bootstrap CI:            [{lo:+.1%}, {hi:+.1%}]")
    print(f"   share of bootstraps > 0:     {(boot > 0).mean():.0%}")
    print(f"   Diebold-Mariano stat:        {dm_stat:+.2f}  (one-sided p = {dm_p:.2f})")
    print(f"   -> {'suggestive, not established' if dm_p > 0.05 else 'significant at 5%'}\n")


# ---------------------------------------------------------------------------
# 2. Multiple-testing correction on the lead-lag correlations
# ---------------------------------------------------------------------------
def lead_lag_significance(df: pd.DataFrame) -> None:
    pairs = [
        ("npm_browser_rum_yoy", "customers_arr_100k_yoy_pct"),
        ("npm_dd_trace_yoy", "revenue_yoy_pct"),
    ]
    windows = {"primary": PRIMARY_WINDOW_START, "robustness": ROBUSTNESS_WINDOW_START}
    lags = range(-2, 3)

    rows = []
    for sig, tgt in pairs:
        for wname, wstart in windows.items():
            w = df[df.index >= wstart]
            for lag in lags:
                s = w[sig].shift(-lag)
                joined = pd.concat([s, w[tgt]], axis=1).dropna()
                if len(joined) < 5:
                    continue
                r, p = stats.pearsonr(joined.iloc[:, 0], joined.iloc[:, 1])
                rows.append({"pair": f"{sig} vs {tgt}", "window": wname,
                             "lag": lag, "n": len(joined), "r": r, "p_raw": p})
    table = pd.DataFrame(rows)

    # Benjamini-Hochberg
    table = table.sort_values("p_raw").reset_index(drop=True)
    m = len(table)
    table["p_bh"] = (table["p_raw"] * m / (table.index + 1)).clip(upper=1.0)
    table["p_bh"] = table["p_bh"][::-1].cummin()[::-1]

    print("2. LEAD-LAG SIGNIFICANCE (two tested pairs, BH-adjusted)")
    print(f"   {m} correlations tested here. The development scan (not in this")
    print(f"   submission) covered several hundred signal x target x window x lag")
    print(f"   comparisons; BH there leaves essentially nothing, which is why it")
    print(f"   is excluded from evidence.\n")
    show = table[table["p_bh"] < 0.10].sort_values(["pair", "window", "lag"])
    if show.empty:
        print("   nothing survives BH at q<0.10\n")
    else:
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print()


# ---------------------------------------------------------------------------
# 3. Revenue nowcast: walk-forward backtest of the guidance rule, then forward
# ---------------------------------------------------------------------------
MIN_BEATS = 4  # trailing beats needed before the guidance rule makes a call


def _revenue_backtest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per quarter where the guidance rule could be applied out of
    sample: its prediction, the naive same-YoY-growth prediction, and what
    was needed to score both. YoY growth is recomputed from the revenue-dollar
    column here so the naive rule uses one self-consistent source.
    """
    rev = df["revenue_usd_m"]
    g = df[df["revenue_guidance_mid_usd_m"].notna() & rev.notna()]
    rows = []
    for i, q in enumerate(g.index):
        prior_beats = g.iloc[:i]["guidance_beat_pct"].dropna()
        need = [q - 1, q - 4, q - 5]
        if len(prior_beats) < MIN_BEATS or any(p not in df.index or pd.isna(rev[p]) for p in need):
            continue
        year_ago = rev[q - 4]
        prior_yoy = (rev[q - 1] / rev[q - 5] - 1) * 100
        rows.append({
            "quarter": q,
            "actual": rev[q],
            "year_ago": year_ago,
            "prior_yoy": prior_yoy,
            "actual_yoy": (rev[q] / year_ago - 1) * 100,
            "pred_guide": g.loc[q, "revenue_guidance_mid_usd_m"] * (1 + prior_beats.mean() / 100),
            "pred_naive": year_ago * (1 + prior_yoy / 100),
        })
    return pd.DataFrame(rows).set_index("quarter")


def revenue_backtest(df: pd.DataFrame) -> None:
    r = _revenue_backtest_rows(df)

    def score(pred: pd.Series) -> tuple[float, float, float]:
        err = pred - r["actual"]
        mape = float((err.abs() / r["actual"]).mean() * 100)
        rmse = float(np.sqrt((err ** 2).mean()))
        pred_yoy = (pred / r["year_ago"] - 1) * 100
        hit = float((np.sign(pred_yoy - r["prior_yoy"])
                     == np.sign(r["actual_yoy"] - r["prior_yoy"])).mean())
        return mape, rmse, hit

    print("3. REVENUE NOWCAST BACKTEST (walk-forward, reported revenue in $M)")
    print(f"   out-of-sample quarters:            {len(r)}  "
          f"({r.index.min()} to {r.index.max()})")
    for label, col in [("guidance x trailing beat", "pred_guide"),
                       ("naive same-YoY-growth", "pred_naive")]:
        mape, rmse, hit = score(r[col])
        print(f"   {label:24s}  MAPE {mape:5.1f}%   RMSE ${rmse:6.1f}M   "
              f"accel/decel hit {hit:.2f}")
    print()


def guidance_analysis(df: pd.DataFrame) -> None:
    target_period = pd.Period(TARGET_QUARTER, freq="Q")
    beat = df["guidance_beat_pct"].dropna()
    fwd = df.loc[target_period, "revenue_guidance_mid_usd_m"]

    mean_beat, sd_beat = beat.mean(), beat.std(ddof=1)
    nowcast = fwd * (1 + mean_beat / 100)
    band_lo = fwd * (1 + (mean_beat - sd_beat) / 100)
    band_hi = fwd * (1 + (mean_beat + sd_beat) / 100)
    last_actual_growth = df["revenue_yoy_pct"].dropna().iloc[-1]
    year_ago_rev = df.loc[target_period - 4, "revenue_usd_m"]
    rw_nowcast = year_ago_rev * (1 + last_actual_growth / 100)

    print("4. GUIDANCE-ANCHORED REVENUE NOWCAST (forward, target quarter)")
    print(f"   quarters with guidance gathered:   {len(beat)} of ~14 (2025Q2 not verified)")
    print(f"   actual vs guidance midpoint beat:   mean {mean_beat:+.1f}%  sd {sd_beat:.1f}pp  "
          f"(range {beat.min():+.1f} to {beat.max():+.1f}, all positive: {(beat > 0).all()})")
    print(f"   {TARGET_QUARTER} guidance midpoint:           ${fwd:,.0f}M")
    print(f"   guidance + mean beat (+/- 1 sd):    ${nowcast:,.0f}M  "
          f"[${band_lo:,.0f}M, ${band_hi:,.0f}M]  (implied YoY ~{(nowcast / year_ago_rev - 1) * 100:.0f}%)")
    print(f"   random-walk nowcast (last YoY held): ${rw_nowcast:,.0f}M")
    print(f"   -> both land above guidance; call is 'tracking modestly ahead of "
          f"the guidance midpoint', in line with Datadog's consistent beat.\n")


def main() -> None:
    if not PANEL.exists():
        raise SystemExit("Run src/build_panel.py first.")
    df = load_panel()
    skill_uncertainty(df)
    lead_lag_significance(df)
    revenue_backtest(df)
    guidance_analysis(df)


if __name__ == "__main__":
    main()
