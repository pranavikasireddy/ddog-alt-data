"""Predictive model and walk-forward backtest.

Compares regression models (a Datadog KPI predicted from a single alt-data
signal) against three baselines that use no external signal at all:

  random_walk     next quarter's value = last quarter's value
  seasonal_naive  next quarter's value = same quarter one year ago
  ar1             next quarter's value = linear function of last quarter's value
  ols[<signal>]   next quarter's value = linear function of the signal

All four are evaluated with an expanding-window walk-forward split: at each
step, the model is fit only on quarters strictly before the one being
predicted, then makes one out-of-sample prediction. This mirrors how the
model would actually be used in real time and avoids fitting on data that
would not yet have existed at the point of prediction.

An expanding window (rather than a fixed-size rolling window) is used because
the sample is already very small (see PRIMARY_WINDOW_START in src/config.py):
a rolling window would shrink the training set further and make the fitted
line unstable quarter to quarter.

Runs on both the primary and robustness windows (see src/config.py), so the
result can be checked for whether it holds up outside the window it was
found in.

Usage:
    python analysis/backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PRIMARY_WINDOW_START, ROBUSTNESS_WINDOW_START  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "quarterly_panel.csv"

# customers_arr_100k_yoy_pct (growth in the $100k+ ARR customer count) rather
# than revenue_yoy_pct: revenue is smoothed by ratable GAAP recognition, so it
# has little real quarter-to-quarter surprise for any signal to predict. ARR
# is a run-rate metric and moves more freely. Set back to "revenue_yoy_pct"
# to reproduce that (weaker) result instead -- see analysis/lead_lag.py for
# both.
TARGET = "customers_arr_100k_yoy_pct"
MIN_TRAIN = 6           # quarters used to fit the first prediction
WINDOWS = {
    "primary": PRIMARY_WINDOW_START,
    "robustness": ROBUSTNESS_WINDOW_START,
}

# SIGNAL and SIGNAL_LAG come from analysis/lead_lag.py: npm_browser_rum_yoy's
# correlation with customer-count growth peaks at lag-1 in BOTH the primary
# window (r=+0.89) and the robustness window (r=+0.97) -- the same sign and
# nearly the same lag in each, which is what makes it worth backtesting.
# npm_browser_logs_yoy looks similar on paper but does NOT reproduce this in
# the backtest (tested and rejected; see the report for the actual numbers).
# SIGNAL_LAG quarters are subtracted here so that at each prediction step,
# the model only sees the signal value from that many quarters before the
# quarter it is predicting -- consistent with the point-in-time rule used
# everywhere else.
SIGNAL = "npm_browser_rum_yoy"
SIGNAL_LAG = 1


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    mape = float(np.mean(np.abs(error / y_true)) * 100)
    rmse = float(np.sqrt(np.mean(error**2)))
    # Directional hit rate: did the predicted quarter-over-quarter change have
    # the same sign as the actual? np.diff (no prepend) starts the comparison
    # at the second point, so no model gets a free correct call from a
    # spurious 0 == 0 at the first element.
    hit_rate = float(np.mean(np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true))))
    return {"MAPE": mape, "RMSE": rmse, "hit_rate": hit_rate}


def walk_forward(df: pd.DataFrame, predict_fn) -> pd.DataFrame:
    rows = []
    for k in range(MIN_TRAIN, len(df)):
        train, test = df.iloc[:k], df.iloc[k]
        prediction = predict_fn(train, test)
        rows.append({"quarter": df.index[k], "y_true": test[TARGET], "y_pred": prediction})
    return pd.DataFrame(rows).set_index("quarter")


def predict_random_walk(train, test):
    return train[TARGET].iloc[-1]


def predict_seasonal_naive(train, test):
    return train[TARGET].iloc[-4] if len(train) >= 4 else train[TARGET].iloc[-1]


def predict_ar1(train, test):
    y = train[TARGET].dropna().values
    if len(y) < 4:
        return y[-1]
    slope, intercept = np.polyfit(y[:-1], y[1:], 1)
    return intercept + slope * y[-1]


def predict_ols_signal(train, test):
    fit_data = train[[TARGET, SIGNAL]].dropna()
    if len(fit_data) < 4 or pd.isna(test[SIGNAL]):
        return train[TARGET].iloc[-1]
    slope, intercept = np.polyfit(fit_data[SIGNAL].values, fit_data[TARGET].values, 1)
    return intercept + slope * test[SIGNAL]


def predict_ar1_plus_signal(train, test):
    """AR1 augmented with the signal, to test whether the signal adds
    anything beyond what the target's own recent momentum already explains.
    predict_ols_signal above ignores momentum entirely, which is an unfair
    test given how strong the random_walk and ar1 baselines already are.
    """
    fit_data = train[[TARGET, SIGNAL]].copy()
    fit_data["growth_lag1"] = fit_data[TARGET].shift(1)
    fit_data = fit_data.dropna()
    if len(fit_data) < 5 or pd.isna(test[SIGNAL]):
        return predict_ar1(train, test)
    predictors = np.column_stack([
        np.ones(len(fit_data)), fit_data["growth_lag1"].values, fit_data[SIGNAL].values
    ])
    intercept, momentum_coef, signal_coef = np.linalg.lstsq(
        predictors, fit_data[TARGET].values, rcond=None
    )[0]
    return intercept + momentum_coef * train[TARGET].iloc[-1] + signal_coef * test[SIGNAL]


MODELS = {
    "random_walk": predict_random_walk,
    "seasonal_naive": predict_seasonal_naive,
    "ar1": predict_ar1,
    f"ols[{SIGNAL}]": predict_ols_signal,
    f"ar1+{SIGNAL}": predict_ar1_plus_signal,
}


def run_window(df: pd.DataFrame, window_start: str) -> None:
    window = df[df.index >= pd.Period(window_start, freq="Q")]
    print(f"window = {window_start} to latest, n = {len(window)}, "
          f"first prediction at quarter {MIN_TRAIN}\n")

    results = {}
    for name, predict_fn in MODELS.items():
        predictions = walk_forward(window, predict_fn).dropna()
        results[name] = compute_metrics(predictions["y_true"].values, predictions["y_pred"].values)

    results_table = pd.DataFrame(results).T
    print(results_table.round(2).to_string())

    baseline_rmse = results_table.loc[["random_walk", "seasonal_naive", "ar1"], "RMSE"].min()
    print(f"\nbest baseline RMSE = {baseline_rmse:.2f}")
    for model_name in (f"ols[{SIGNAL}]", f"ar1+{SIGNAL}"):
        skill_score = 1 - results_table.loc[model_name, "RMSE"] / baseline_rmse
        print(f"skill score of {model_name} vs. best baseline = {skill_score:+.2%}")


def main() -> None:
    if not PANEL.exists():
        raise SystemExit("Run src/build_panel.py first.")
    df = pd.read_csv(PANEL, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="Q")
    df[SIGNAL] = df[SIGNAL].shift(SIGNAL_LAG)  # shift before windowing/trimming
    df = df[df[TARGET].notna()]

    print(f"target = {TARGET}, signal = {SIGNAL} (lagged {SIGNAL_LAG}q)\n")
    for name, window_start in WINDOWS.items():
        print(f"--- {name} window ---")
        run_window(df, window_start)
        print()


if __name__ == "__main__":
    main()
