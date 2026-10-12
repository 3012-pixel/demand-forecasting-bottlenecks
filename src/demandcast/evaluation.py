"""Accuracy metrics and rolling-origin backtesting.

Backtest design (mimics a weekly planning cycle): starting at ``test_start``,
refit every ``step`` days on all data before the origin, forecast the next
``horizon`` days, and move on. No information after the origin is used.
Metrics are computed on open days only (closed days are 0 by construction).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from demandcast.business_calendar import closed_mask
from demandcast.models import Forecaster


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error (%), ignoring zero actuals."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    nz = a != 0
    return float(np.mean(np.abs((a[nz] - f[nz]) / a[nz])) * 100)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE (%), in [0, 200]. Pairs where both are 0 count as 0 error."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = np.abs(a) + np.abs(f)
    ratio = np.divide(2 * np.abs(f - a), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(np.mean(ratio) * 100)


def forecast_with_calendar(model: Forecaster, dates: pd.DatetimeIndex) -> pd.Series:
    """Model forecast with planned-closure days forced to zero."""
    pred = model.predict(dates)
    pred[closed_mask(dates).to_numpy()] = 0.0
    return pred


def rolling_backtest(
    models: Iterable[Forecaster],
    series: pd.Series,
    test_start: str | pd.Timestamp,
    horizon: int = 7,
    step: int = 7,
) -> pd.DataFrame:
    """Rolling-origin backtest.

    Args:
        series: daily units with NaN on closed days (see ``data.to_series``).

    Returns:
        Long DataFrame: ``origin, date, horizon_day, model, forecast, actual, is_open``.
    """
    test_start = pd.Timestamp(test_start)
    end = series.index.max()
    rows = []
    origin = test_start
    while origin <= end:
        history = series[series.index < origin]
        dates = pd.date_range(origin, min(origin + pd.Timedelta(days=horizon - 1), end), freq="D")
        actual = series.reindex(dates)
        for m in models:
            m.fit(history)
            fc = forecast_with_calendar(m, dates)
            rows.append(pd.DataFrame({
                "origin": origin,
                "date": dates,
                "horizon_day": np.arange(1, len(dates) + 1),
                "model": m.name,
                "forecast": fc.to_numpy(),
                "actual": actual.fillna(0).to_numpy(),
                "is_open": actual.notna().to_numpy(),
            }))
        origin += pd.Timedelta(days=step)
    return pd.concat(rows, ignore_index=True)


def score(backtest: pd.DataFrame, baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Per-model accuracy on open days, plus MAE skill vs the baseline.

    ``mae_skill_vs_baseline_pct`` > 0 means the model beats the baseline.
    """
    bt = backtest[backtest["is_open"]]
    rows = []
    for name, g in bt.groupby("model", sort=False):
        rows.append({
            "model": name,
            "n_days": len(g),
            "mae": mae(g["actual"], g["forecast"]),
            "rmse": rmse(g["actual"], g["forecast"]),
            "mape_pct": mape(g["actual"], g["forecast"]),
            "smape_pct": smape(g["actual"], g["forecast"]),
            "bias_pct": float((g["forecast"].sum() / g["actual"].sum() - 1) * 100),
        })
    df = pd.DataFrame(rows)
    if baseline in set(df["model"]):
        base = float(df.loc[df["model"] == baseline, "mae"].iloc[0])
        df["mae_skill_vs_baseline_pct"] = (1 - df["mae"] / base) * 100
    return df.sort_values("mae", ignore_index=True)


def error_breakdown(backtest: pd.DataFrame, by: str) -> pd.DataFrame:
    """MAE and sMAPE by a grouping column (e.g. weekday, week) for failure analysis."""
    bt = backtest[backtest["is_open"]].copy()
    if by == "weekday":
        bt["weekday"] = bt["date"].dt.day_name().str[:3]
    elif by == "week":
        bt["week"] = bt["date"].dt.to_period("W-SUN").dt.start_time
    out = bt.groupby([by, "model"]).apply(
        lambda g: pd.Series({"mae": mae(g["actual"], g["forecast"]),
                             "smape_pct": smape(g["actual"], g["forecast"])}),
        include_groups=False,
    )
    return out.reset_index()
