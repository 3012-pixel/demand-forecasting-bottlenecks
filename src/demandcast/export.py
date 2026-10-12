"""Power BI / Excel-ready exports: tidy CSVs in a star-schema layout.

Load the folder in Power BI (Get Data > Folder or Text/CSV) and relate
every ``fact_*`` table to ``dim_date`` on ``date``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def dim_date(dates: pd.Series | pd.DatetimeIndex, open_flags: pd.Series | None = None) -> pd.DataFrame:
    """Date dimension table."""
    d = pd.DatetimeIndex(pd.to_datetime(dates)).unique().sort_values()
    df = pd.DataFrame({"date": d})
    df["year"] = d.year
    df["month"] = d.month
    df["month_name"] = d.month_name().str[:3]
    df["iso_week"] = d.isocalendar().week.to_numpy()
    df["weekday"] = d.day_name().str[:3]
    df["weekday_num"] = d.dayofweek + 1
    if open_flags is not None:
        df["is_open"] = df["date"].map(open_flags).fillna(True).astype(bool)
    return df


def export_powerbi(out_dir: str | Path, daily: pd.DataFrame, backtest: pd.DataFrame,
                   metrics: pd.DataFrame, bottlenecks: pd.DataFrame,
                   future: pd.DataFrame | None = None) -> list[Path]:
    """Write the star-schema CSVs; return paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dates = pd.concat([daily["date"], backtest["date"]] + ([future["date"]] if future is not None else []))
    from demandcast.business_calendar import open_mask  # local import avoids a cycle at import time

    flags = open_mask(pd.DatetimeIndex(all_dates.unique()))
    tables = {
        "dim_date.csv": dim_date(all_dates, flags),
        "fact_daily_actuals.csv": daily[["date", "units", "orders", "is_open"]],
        "fact_backtest_forecasts.csv": backtest,
        "model_metrics.csv": metrics,
        "fact_bottlenecks.csv": bottlenecks,
    }
    if future is not None:
        tables["fact_future_forecast.csv"] = future
    paths = []
    for name, df in tables.items():
        p = out_dir / name
        df.to_csv(p, index=False, date_format="%Y-%m-%d")
        paths.append(p)
    return paths
