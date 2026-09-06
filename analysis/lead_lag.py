"""Lead-lag analysis: does each signal move before a DDOG KPI, at the same
time, or after?

For each signal and each target in TARGETS (revenue growth, RPO growth,
$100k-ARR customer-count growth, NRR change, net new ARR), this computes the
signal-target correlation at lags -2 through +2 quarters. A negative lag means
the signal is shifted earlier -- i.e. it tests whether the signal leads the
target. A positive lag tests the reverse (target leading the signal), which is not
economically plausible and is included only as a check: if a signal
correlates just as strongly at a positive lag as at its best negative lag,
that is a sign the correlation comes from both series sharing a common trend
rather than from real predictive lead-lag structure, and should not be
reported as a leading indicator.

Runs on both the primary and robustness windows (see src/config.py) so a
candidate signal can be judged on whether the relationship holds in more than
one window, not just one.

This scans every _yoy signal column, which is a broader search than the two
signals actually reported on in the final write-up. Screening several
candidates and reporting the strongest is legitimate exploratory analysis,
but it should be disclosed as such: finding one correlation above a threshold
out of several tested is weaker evidence than a single a priori hypothesis
that succeeds, because chance alone produces some "hits" when enough signals
are tried against a 14-22 quarter sample.

Usage:
    python analysis/lead_lag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PRIMARY_WINDOW_START, ROBUSTNESS_WINDOW_START  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "quarterly_panel.csv"

# Revenue growth is smoothed by ratable GAAP recognition. RPO, the $100k-ARR
# customer count, NRR, and net new ARR are closer to bookings/usage and may
# carry more real quarter-to-quarter variance for a signal to actually track
# -- worth checking before concluding no signal works against any Datadog
# metric. Several other candidate targets (billings_yoy_pct, total customer
# count, $1M-ARR customer count, multi-product adoption, cRPO growth) are
# NOT included here: each has too few valid year-over-year pairs (1-6) in
# the data actually gathered (see data/ddog_kpis.csv) for a correlation to
# mean anything.
TARGETS = [
    "revenue_yoy_pct",
    "rpo_yoy_pct",
    "customers_arr_100k_yoy_pct",
    "dbnr_qoq_change",
    "net_new_arr_usd_m",
]
LAGS = range(-2, 3)
WINDOWS = {
    "primary": PRIMARY_WINDOW_START,
    "robustness": ROBUSTNESS_WINDOW_START,
}


def xcorr(panel: pd.DataFrame, signal_col: str, target_col: str) -> pd.Series:
    signal = panel[signal_col].astype(float)
    target = panel[target_col].astype(float)
    return pd.Series({lag: signal.shift(-lag).corr(target) for lag in LAGS})


def run_window(panel: pd.DataFrame, window_start: str, target: str, candidate_signals: list[str]) -> None:
    window = panel[panel.index >= window_start]
    n = window[target].notna().sum()
    print(f"window = {window_start} to latest, n = {n} quarters with {target} available\n")

    table = pd.DataFrame(
        {s: xcorr(window, s, target) for s in candidate_signals if s in window}
    ).T
    table = table.dropna(how="all")
    table.columns = [f"lag{c:+d}" for c in table.columns]
    print(table.round(2).to_string())

    print("\nStrongest lag per signal:")
    for signal_col in table.index:
        best_lag = table.loc[signal_col].abs().idxmax()
        print(f"  {signal_col:24s} {best_lag:>7s}  r = {table.loc[signal_col, best_lag]:+.2f}")


def main() -> None:
    if not PANEL.exists():
        raise SystemExit("Run src/build_panel.py first.")
    panel = pd.read_csv(PANEL, index_col=0)
    candidate_signals = [c for c in panel.columns if c.endswith("_yoy")]

    for target in TARGETS:
        print(f"===== target = {target} =====\n")
        for name, window_start in WINDOWS.items():
            print(f"--- {name} window ---")
            run_window(panel, window_start, target, candidate_signals)
            print()


if __name__ == "__main__":
    main()
