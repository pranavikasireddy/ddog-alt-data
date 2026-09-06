"""Load hyperscaler (AWS / Azure / GCP) quarterly revenue as a macro control.

Signal thesis: Datadog's own filings say revenue is "closely correlated with
customers' cloud workloads". AWS / Azure / Google Cloud revenue is the upstream
measure of how much cloud compute the world is running. It is only quarterly
(so coincident, not high-frequency) but it is a clean, filings-grade control
variable and a sanity check on the scrappier signals.

There is no clean, free, ToS-safe API for segment revenue, and scraping it well
is a time sink. So this is a MANUAL table you fill once from primary sources
(~20 min):

  * AWS revenue           -> Amazon 10-Q, "Segment Information" (AWS net sales)
  * Azure                 -> Microsoft only gives Azure *growth %*, not $$.
                             Use "Intelligent Cloud" segment revenue as the $ proxy,
                             and also record the reported Azure growth %.
  * Google Cloud revenue  -> Alphabet 10-Q, "Google Cloud" revenues line

Fill templates/cloud_revenue_template.csv, then this script just validates and
copies it into data/raw/ with a fetch date.

Usage:
    python src/fetch_cloud_revenue.py
"""
from __future__ import annotations

import sys

import pandas as pd

from config import TEMPLATES, save_raw

TEMPLATE = TEMPLATES / "cloud_revenue_template.csv"
REQUIRED = {"fiscal_quarter", "period_end", "aws_rev_usd_m",
            "msft_intelligent_cloud_usd_m", "azure_growth_yoy_pct",
            "gcp_rev_usd_m", "source_url"}


def load() -> pd.DataFrame:
    if not TEMPLATE.exists():
        sys.exit(f"Missing {TEMPLATE}. Fill it in first (see module docstring).")
    df = pd.read_csv(TEMPLATE)
    missing = REQUIRED - set(df.columns)
    if missing:
        sys.exit(f"{TEMPLATE} is missing columns: {sorted(missing)}")

    filled = df.dropna(subset=["aws_rev_usd_m"])
    if len(filled) < 8:
        print(f"WARNING: only {len(filled)} quarters have AWS revenue filled in. "
              f"Aim for 12+ before running the backtest.")
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df


if __name__ == "__main__":
    print(f"Loading {TEMPLATE.name} ...")
    out = load()
    print(out.tail(8).to_string(index=False))
    save_raw(out, "cloud_revenue")
