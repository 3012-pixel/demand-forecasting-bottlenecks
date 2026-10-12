"""Exploratory analysis: trend, annual seasonality, weekday effects, outliers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from demandcast.plotting import BLUE, GREY, ORANGE, RED, style_axes, thousands  # noqa: E402


def weekday_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean/median units by weekday on open days, and index vs overall mean."""
    o = daily[daily["is_open"]].copy()
    o["weekday"] = o["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    prof = o.groupby("weekday")["units"].agg(["count", "mean", "median"]).reindex(order).dropna()
    prof["index_vs_mean"] = prof["mean"] / o["units"].mean()
    return prof.reset_index()


def weekly_totals(daily: pd.DataFrame) -> pd.Series:
    """Units per week (weeks ending Sunday), excluding partial first/last weeks."""
    w = daily.set_index("date")["units"].resample("W-SUN").sum()
    return w.iloc[1:-1]


def yoy_growth(daily: pd.DataFrame, start_md: str = "09-01", end_md: str = "11-30") -> dict[str, float]:
    """Compare the same season across years (trend check without extrapolating)."""
    out: dict[str, float] = {}
    d = daily.set_index("date")["units"]
    for year in sorted(set(d.index.year)):
        window = d[f"{year}-{start_md}":f"{year}-{end_md}"]
        if len(window) > 80:
            out[str(year)] = float(window.sum())
    years = sorted(out)
    if len(years) >= 2:
        out["growth_pct"] = (out[years[-1]] / out[years[-2]] - 1) * 100
    return out


def detect_outliers(daily: pd.DataFrame, high: float = 2.5, low: float = 0.4, window_weeks: int = 3) -> pd.DataFrame:
    """Flag open days far from the median of the same weekday in +/- ``window_weeks``.

    A transparent, robust rule. Returns the flagged days with ``ratio`` to baseline.
    """
    o = daily[daily["is_open"]].set_index("date")["units"]
    rows = []
    for d, v in o.items():
        neighbours = [d + pd.Timedelta(weeks=k) for k in range(-window_weeks, window_weeks + 1) if k != 0]
        base = o.reindex(neighbours).dropna()
        if len(base) < 2:
            continue
        ratio = v / base.median()
        if ratio > high or ratio < low:
            rows.append({"date": d, "units": v, "baseline_median": base.median(), "ratio": ratio,
                         "type": "spike" if ratio > high else "dip"})
    return pd.DataFrame(rows, columns=["date", "units", "baseline_median", "ratio", "type"])


def eda_summary(daily: pd.DataFrame, high: float = 2.5, low: float = 0.4) -> dict:
    """Headline numbers used in the README."""
    o = daily[daily["is_open"]]
    out = detect_outliers(daily, high, low)
    cv_daily = float(o["units"].std() / o["units"].mean())
    return {
        "date_range": f"{daily['date'].min():%Y-%m-%d} to {daily['date'].max():%Y-%m-%d}",
        "calendar_days": int(len(daily)),
        "open_days": int(len(o)),
        "mean_units_open_day": float(o["units"].mean()),
        "median_units_open_day": float(o["units"].median()),
        "p90_units_open_day": float(o["units"].quantile(0.9)),
        "max_units_day": float(o["units"].max()),
        "coefficient_of_variation": cv_daily,
        "autumn_volume": yoy_growth(daily),
        "n_spikes": int((out["type"] == "spike").sum()) if len(out) else 0,
        "n_dips": int((out["type"] == "dip").sum()) if len(out) else 0,
    }


def plot_eda(daily: pd.DataFrame, out_dir: str | Path, high: float = 2.5, low: float = 0.4) -> list[Path]:
    """Save the EDA charts; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Weekly volume with trend line and year-over-year overlay
    w = weekly_totals(daily)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(w.index, w.values, color=BLUE, linewidth=1.2, label="Weekly units")
    ax.plot(w.index, w.rolling(8, center=True).mean(), color=ORANGE, linewidth=2, label="8-week rolling mean")
    ax.set_title("Weekly units shipped: trend and annual peak")
    ax.yaxis.set_major_formatter(thousands())
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    ax = axes[1]
    iso = w.index.isocalendar()  # group by ISO year so week 52/53 never wraps to week 1
    for year, colour in zip(sorted(set(iso["year"])), [GREY, BLUE, ORANGE]):
        wy = w[(iso["year"] == year).to_numpy()]
        if len(wy) < 4:
            continue
        ax.plot(wy.index.isocalendar().week, wy.values, color=colour, label=str(year), linewidth=1.5)
    ax.set_title("Same weeks, year over year")
    ax.set_xlabel("ISO week")
    ax.yaxis.set_major_formatter(thousands())
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    paths.append(out_dir / "eda_trend_seasonality.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    # 2. Weekday profile
    prof = weekday_profile(daily)
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.bar(prof["weekday"].str[:3], prof["mean"], color=BLUE)
    ax.axhline(daily.loc[daily["is_open"], "units"].mean(), color=GREY, linestyle="--", linewidth=1,
               label="Open-day mean")
    ax.set_title("Average units by weekday (open days; Saturday = closed)")
    ax.yaxis.set_major_formatter(thousands())
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    paths.append(out_dir / "eda_weekday.png")
    fig.savefig(paths[-1])
    plt.close(fig)

    # 3. Daily series with outliers
    outl = detect_outliers(daily, high, low)
    o = daily[daily["is_open"]]
    fig, ax = plt.subplots(figsize=(13, 3.8), dpi=150)
    ax.plot(o["date"], o["units"], color=BLUE, linewidth=0.7, label="Daily units (open days)")
    if len(outl):
        sp = outl[outl["type"] == "spike"]
        dp = outl[outl["type"] == "dip"]
        ax.scatter(sp["date"], sp["units"], color=RED, s=22, zorder=3, label=f"Spike (>{high}x local norm)")
        ax.scatter(dp["date"], dp["units"], color=ORANGE, s=22, zorder=3, label=f"Dip (<{low}x local norm)")
    ax.set_title("Daily volume and outlier days")
    ax.yaxis.set_major_formatter(thousands())
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    paths.append(out_dir / "eda_outliers.png")
    fig.savefig(paths[-1])
    plt.close(fig)
    return paths

