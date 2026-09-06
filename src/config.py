"""Shared configuration and helpers for the DDOG alt-data project.

Import this from every fetch/analysis script so paths, date ranges, and the
"save raw data with a fetch date" convention stay consistent.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
TEMPLATES = ROOT / "templates"

# ---------------------------------------------------------------------------
# Analysis window
# ---------------------------------------------------------------------------
# Pull signals from well before the first backtest quarter so we can compute
# trailing / YoY transforms with no missing values at the start.
SIGNAL_START = "2018-01-01"
SIGNAL_END = date.today().isoformat()

# Datadog reports on the calendar quarter. The quarter you are *nowcasting*
# right now (Sept 2026) is 2026Q3, which prints in early November 2026.
TARGET_QUARTER = "2026Q3"

# ---------------------------------------------------------------------------
# Backtest windows
# ---------------------------------------------------------------------------
# Datadog has passed through three distinct growth regimes: 2021 hypergrowth
# (COVID-era cloud migration, ~65-85% YoY), 2022-23 slowdown (customers cut
# cloud spend, growth fell to the mid-20s% YoY), and 2024-2026 reacceleration
# (AI-driven workloads, growth back up to the mid-30s% YoY). Pooling all of
# these into a single regression would let the earliest, most different regime
# dominate a model that is meant to predict the current one.
#
# PRIMARY_WINDOW_START defines the main backtest window: the most recent
# quarters, spanning a more consistent regime. analysis/lead_lag.py and
# analysis/backtest.py both use this as their default window.
PRIMARY_WINDOW_START = "2023Q1"

# ROBUSTNESS_WINDOW_START extends back through the earlier regimes, for a
# separate sensitivity check on whether the same relationship holds further
# back. It is not combined with the primary window in any reported result.
ROBUSTNESS_WINDOW_START = "2021Q1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_date() -> str:
    """Today's date as YYYY-MM-DD, for stamping raw-data filenames."""
    return date.today().isoformat()


def save_raw(df: pd.DataFrame, name: str) -> Path:
    """Write a raw pull to data/raw/<name>__fetched-YYYY-MM-DD.csv.

    Google Trends re-normalises its index on every pull, and some vendors
    revise history, so the fetch date is part of the data's identity. Each
    pull is written to a new file rather than overwriting the previous one.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / f"{name}__fetched-{fetch_date()}.csv"
    df.to_csv(path, index=False)
    print(f"  saved {len(df):>6,} rows -> {path.relative_to(ROOT)}")
    return path


def latest_raw(name_prefix: str) -> Path:
    """Return the most recent data/raw file whose name starts with name_prefix."""
    matches = sorted(DATA_RAW.glob(f"{name_prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No raw file matching '{name_prefix}*' in {DATA_RAW}. "
            f"Run the matching fetch_*.py script first."
        )
    return matches[-1]
