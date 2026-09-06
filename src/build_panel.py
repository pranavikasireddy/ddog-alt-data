"""Merge the DDOG target KPIs and the alt-data signals into one quarterly panel.

Output: data/processed/quarterly_panel.csv, one row per calendar quarter,
containing the reported DDOG KPIs plus each signal. Raw signals come in two
shapes:

  * sub-quarterly (daily/weekly) series -- Google Trends and npm downloads --
    aggregated to the quarter in three forms: <col>_qmean (mean over the
    quarter), <col>_qend (mean of the last two observations before quarter-end,
    approximating what was known at the time), and <col>_yoy (year-over-year
    percent change of <col>_qmean).

  * hyperscaler cloud revenue -- already quarterly, hand-filled from Amazon and
    Alphabet filings into templates/cloud_revenue_template.csv; level columns
    additionally get a <col>_yoy column.

Run the fetch_*.py scripts first. data/ddog_kpis.csv and data/ddog_guidance.csv
(the targets, filled from each quarter's 8-K earnings press release) must
already exist.

Usage:
    python src/build_panel.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, ROOT, latest_raw

KPI_CSV = ROOT / "data" / "ddog_kpis.csv"
GUIDANCE_CSV = ROOT / "data" / "ddog_guidance.csv"

# name -> date column, for signals with sub-quarterly raw data
SUB_QUARTERLY_SIGNALS = {
    "google_trends": "week",
    "npm_downloads": "day",
}


def load_targets() -> pd.DataFrame:
    if not KPI_CSV.exists():
        sys.exit(f"Missing {KPI_CSV}. It should list revenue_usd_m etc. per "
                  f"quarter, taken from each quarter's 8-K exhibit 99.1.")
    df = pd.read_csv(KPI_CSV)
    df["quarter"] = df["fiscal_quarter"].apply(lambda q: pd.Period(q, freq="Q"))
    df = df.set_index("quarter").sort_index()
    if "revenue_yoy_pct" not in df or df["revenue_yoy_pct"].isna().all():
        df["revenue_yoy_pct"] = df["revenue_usd_m"].pct_change(4) * 100
    # Alternate targets: revenue growth is smoothed by ratable GAAP recognition,
    # so RPO and $100k-ARR customer count (closer to bookings/usage) may carry
    # more real quarter-to-quarter signal for an alt-data predictor to catch.
    # Billings, total/$1M-ARR customer counts, cRPO and 4+-product adoption are
    # not derived here: each has too few consistently-disclosed quarters in
    # data/ddog_kpis.csv to correlate against (report Section 2.1).
    df["rpo_yoy_pct"] = df["rpo_usd_m"].pct_change(4) * 100
    df["customers_arr_100k_yoy_pct"] = df["customers_arr_ge_100k"].pct_change(4) * 100
    # NRR growth is already a rate, not a level, so its "growth" is the change
    # in percentage points quarter over quarter, not a percent change of a percent.
    df["dbnr_qoq_change"] = df["dbnr_rate_pct"].diff(1)
    # Net new ARR: the dollar change in annualized revenue quarter over
    # quarter. Unlike a percent-growth target, this does not mechanically
    # decelerate as the revenue base grows, so it can reveal a relationship
    # that percent growth targets obscure.
    df["net_new_arr_usd_m"] = df["revenue_usd_m"].diff(1) * 4

    # Datadog's own next-quarter revenue guidance midpoint, where gathered
    # (partial coverage -- see data/ddog_guidance.csv). guidance_beat_pct is
    # how far actual revenue came in above the midpoint; it is the closest
    # free proxy for "tracking ahead of / behind consensus".
    guidance = pd.read_csv(GUIDANCE_CSV)
    guidance["quarter"] = guidance["fiscal_quarter"].apply(lambda q: pd.Period(q, freq="Q"))
    guidance = guidance.set_index("quarter")[["revenue_guidance_mid_usd_m"]]
    df = df.join(guidance)
    df["guidance_beat_pct"] = (df["revenue_usd_m"] / df["revenue_guidance_mid_usd_m"] - 1) * 100
    return df


def aggregate_sub_quarterly(path, date_col: str) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=[date_col])
    raw[date_col] = raw[date_col].dt.tz_localize(None)  # some sources give tz-aware timestamps
    raw = raw.rename(columns={date_col: "date"}).set_index("date").sort_index()
    numeric = raw.select_dtypes(include=[np.number])
    # Some daily APIs pad the last few days with zeros because the data is not
    # in yet. Drop trailing all-zero rows so they do not drag the current
    # (still incomplete) quarter's average toward zero.
    while len(numeric) and (numeric.iloc[-1] == 0).all():
        numeric = numeric.iloc[:-1]
    quarter = numeric.index.to_period("Q")

    qmean = numeric.groupby(quarter).mean()
    qend = numeric.groupby(quarter).apply(lambda g: g.tail(2).mean())
    out = qmean.add_suffix("_qmean").join(qend.add_suffix("_qend"))
    for col in qmean.columns:
        out[f"{col}_yoy"] = qmean[col].pct_change(4) * 100
    out.index.name = "quarter"
    # The most recent quarter is still in progress at fetch time, so its
    # _qmean / _qend / _yoy are partial. analysis/backtest.py only ever uses
    # the signal at a >=1 quarter lag, so the live nowcast is not affected;
    # a lag of 0 against this column would be contaminated.
    return out


def load_cloud_revenue(path) -> pd.DataFrame:
    cloud = pd.read_csv(path)
    cloud["quarter"] = cloud["fiscal_quarter"].apply(lambda q: pd.Period(q, freq="Q"))
    cloud = cloud.set_index("quarter").select_dtypes(include=[np.number]).sort_index()
    # Revenue LEVEL columns (e.g. aws_rev_usd_m) trend upward on their own and
    # will look spuriously correlated with anything else that also trends
    # upward. Add YoY growth for each level column so it is compared
    # like-for-like with revenue growth rather than with a revenue level. The
    # *_growth_yoy_pct columns are already a growth rate as disclosed by the
    # company, so they are left as-is.
    for col in [c for c in cloud.columns if c.endswith("_usd_m")]:
        cloud[f"{col}_yoy"] = cloud[col].pct_change(4) * 100
    return cloud


def load_signals() -> pd.DataFrame:
    frames = []

    for name, date_col in SUB_QUARTERLY_SIGNALS.items():
        try:
            frames.append(aggregate_sub_quarterly(latest_raw(name), date_col))
            print(f"  + {name}")
        except FileNotFoundError:
            print(f"  - {name} (run src/fetch_{name}.py)")

    try:
        frames.append(load_cloud_revenue(latest_raw("cloud_revenue")))
        print("  + cloud_revenue")
    except FileNotFoundError:
        print("  - cloud_revenue (fill templates/cloud_revenue_template.csv, "
              "then run src/fetch_cloud_revenue.py)")

    if not frames:
        sys.exit("No signal files found. Run at least one fetch_*.py script first.")
    return frames[0].join(frames[1:], how="outer")


def main() -> None:
    print("Loading targets ...")
    targets = load_targets()
    print("Loading and aggregating signals ...")
    signals = load_signals()

    panel = targets.join(signals, how="left")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "quarterly_panel.csv"
    panel.to_csv(out_path)
    print(f"\nwrote {panel.shape[0]} quarters x {panel.shape[1]} columns to "
          f"{out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
