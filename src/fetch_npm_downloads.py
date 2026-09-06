"""Fetch npm download counts for Datadog's client libraries.

Signal thesis: every `npm install dd-trace` is a developer wiring Datadog APM
instrumentation into an application. Download volume is a bottom-up proxy for
the deployment footprint that Datadog bills against (hosts, services, spans).
It is genuinely high-frequency (daily) and has multi-year history.

Data notes / limitations:
  * npm counts include CI re-installs, mirrors, and bots -> level is inflated
    and noisy; build_panel.py aggregates to a quarterly mean and uses YoY
    change, not the raw level.
  * `dd-trace` (Node APM) is only one language/agent; not the whole business.
  * Public download stats are a documented, free API (api.npmjs.org), no key
    required. It caps each request at 18 months, so this loops by year.

Usage:
    python src/fetch_npm_downloads.py
"""
from __future__ import annotations

import datetime as _dt
import time

import pandas as pd
import requests

from config import SIGNAL_START, save_raw

PACKAGES = [
    "dd-trace",                 # Node.js APM tracer (tested vs. revenue growth)
    "@datadog/browser-rum",     # RUM browser SDK (tested vs. customer-count growth)
    "@datadog/browser-logs",    # Logs browser SDK (the sibling that does NOT reproduce)
]

API = "https://api.npmjs.org/downloads/range/{start}:{end}/{pkg}"


def _year_windows(start: str):
    start_d = _dt.date.fromisoformat(start)
    today = _dt.date.today()
    y = start_d.year
    while y <= today.year:
        w0 = max(start_d, _dt.date(y, 1, 1))
        w1 = min(today, _dt.date(y, 12, 31))
        if w0 <= w1:
            yield w0.isoformat(), w1.isoformat()
        y += 1


def fetch_one(pkg: str) -> pd.DataFrame:
    rows = []
    for w0, w1 in _year_windows(SIGNAL_START):
        url = API.format(start=w0, end=w1, pkg=pkg)
        r = requests.get(url, timeout=30)
        if r.status_code == 429:
            time.sleep(3)
            r = requests.get(url, timeout=30)
        if r.status_code == 404:
            print(f"    {pkg}: no data for {w0}..{w1}")
            continue
        r.raise_for_status()
        payload = r.json()
        for d in payload.get("downloads", []):
            rows.append({"day": d["day"], "package": pkg, "downloads": d["downloads"]})
    return pd.DataFrame(rows)


def fetch() -> pd.DataFrame:
    frames = []
    for pkg in PACKAGES:
        print(f"  requesting: {pkg}")
        frames.append(fetch_one(pkg))
    out = pd.concat(frames, ignore_index=True)
    out = out.pivot_table(index="day", columns="package", values="downloads").reset_index()
    out.columns = ["day"] + [f"npm_{c.replace('@datadog/', '').replace('-', '_')}"
                             for c in out.columns[1:]]
    return out


if __name__ == "__main__":
    print("Fetching npm downloads ...")
    df = fetch()
    print(df.tail())
    save_raw(df, "npm_downloads")
