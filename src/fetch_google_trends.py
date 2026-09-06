"""Fetch Google Trends search-interest series for Datadog-related terms.

Signal thesis: search interest in "Datadog" (and buying-intent variants like
"datadog pricing") reflects the top of the adoption / evaluation funnel. In a
usage-priced model, adoption today shows up as billable usage over the
following 1-3 quarters.

Data notes / limitations (put these in the report):
  * Google Trends returns a 0-100 index that is *re-normalised on every pull*
    against the max value in the requested window. A pull today is NOT what you
    would have seen live in, say, 2023Q1 -> this is look-ahead / vintage risk.
    Mitigation: pull one long consistent window, keep the raw file, and treat
    levels as approximate; lean on YoY *changes* rather than absolute levels.
  * Weekly granularity for multi-year windows; daily only for <9-month windows.
  * Unofficial API (pytrends). Be gentle: few requests, backoff on 429.

Usage:
    python src/fetch_google_trends.py
"""
from __future__ import annotations

import time

import pandas as pd

from config import SIGNAL_START, SIGNAL_END, save_raw

KEYWORDS = [
    "Datadog",                  # brand / general interest
    "Datadog pricing",          # buying intent
    "Datadog login",            # proxy for active-user base (logins to the app)
    "Datadog vs New Relic",     # competitive evaluation
    "Datadog vs Splunk",        # competitive evaluation
    "Datadog careers",          # hiring interest, a proxy for the company's own growth
]

TIMEFRAME = f"{SIGNAL_START} {SIGNAL_END}"
GEO = ""  # worldwide


def fetch() -> pd.DataFrame:
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=1.0)

    frames = []
    for kw in KEYWORDS:
        print(f"  requesting: {kw!r}")
        pytrends.build_payload([kw], timeframe=TIMEFRAME, geo=GEO)
        s = pytrends.interest_over_time()
        if s.empty:
            print(f"    (no data for {kw!r})")
            continue
        s = s.drop(columns=[c for c in ("isPartial",) if c in s.columns])
        s = s.rename(columns={kw: _slug(kw)})
        frames.append(s)
        time.sleep(5)  # be polite to the unofficial endpoint

    if not frames:
        raise RuntimeError("Google Trends returned nothing for every keyword.")

    out = pd.concat(frames, axis=1)
    out.index.name = "week"
    return out.reset_index()


def _slug(kw: str) -> str:
    return "gt_" + kw.lower().replace(" ", "_")


if __name__ == "__main__":
    print("Fetching Google Trends ...")
    df = fetch()
    print(df.tail())
    save_raw(df, "google_trends")
