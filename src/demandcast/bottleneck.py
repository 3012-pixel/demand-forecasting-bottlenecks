"""Volume-spike (bottleneck) detection against a daily capacity threshold.

A day is *flagged* when its forecast exceeds capacity. In a backtest we also
know the actual, so each day gets an outcome:

- ``hit``          forecast flagged and actual exceeded capacity
- ``false_alarm``  flagged, but actual stayed within capacity
- ``missed``       not flagged, but actual exceeded capacity
- ``ok``           neither
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from demandcast.plotting import BLUE, GREY, ORANGE, RED, style_axes, thousands  # noqa: E402


def capacity_threshold(history: pd.Series, units_per_day: float | None = None, percentile: float = 90) -> float:
    """Daily capacity: explicit value, else a percentile of open-day history."""
    if units_per_day is not None:
        if units_per_day <= 0:
            raise ValueError("units_per_day must be positive")
        return float(units_per_day)
    if not 0 < percentile < 100:
        raise ValueError("percentile must be in (0, 100)")
    return float(history.dropna().quantile(percentile / 100))


def flag_spikes(forecasts: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Add capacity, utilisation, flag and (if actuals exist) outcome columns.

    Args:
        forecasts: columns ``date``, ``forecast`` and optionally ``actual``.
    """
    df = forecasts.copy()
    df["capacity"] = capacity
    df["forecast_utilisation_pct"] = df["forecast"] / capacity * 100
    df["flagged"] = df["forecast"] > capacity
    df["excess_units_forecast"] = (df["forecast"] - capacity).clip(lower=0)
    if "actual" in df:
        exceeded = df["actual"] > capacity
        df["actual_exceeded"] = exceeded
        df["outcome"] = "ok"
        df.loc[df["flagged"] & exceeded, "outcome"] = "hit"
        df.loc[df["flagged"] & ~exceeded, "outcome"] = "false_alarm"
        df.loc[~df["flagged"] & exceeded, "outcome"] = "missed"
    return df


def spike_skill(flagged: pd.DataFrame) -> dict[str, float]:
    """Precision/recall of spike flags (needs an ``outcome`` column)."""
    counts = flagged["outcome"].value_counts()
    hit, fa, miss = (int(counts.get(k, 0)) for k in ("hit", "false_alarm", "missed"))
    return {
        "hits": hit,
        "false_alarms": fa,
        "missed": miss,
        "precision": hit / (hit + fa) if hit + fa else float("nan"),
        "recall": hit / (hit + miss) if hit + miss else float("nan"),
    }


def plot_bottlenecks(report: pd.DataFrame, path: str | Path, title: str) -> Path:
    """Chart forecast vs actual vs capacity, marking flagged days."""
    path = Path(path)
    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=150)
    ax.plot(report["date"], report["forecast"], color=BLUE, linewidth=1.4, label="Forecast")
    if "actual" in report:
        ax.plot(report["date"], report["actual"], color=GREY, linewidth=1, alpha=0.9, label="Actual")
    cap = float(report["capacity"].iloc[0])
    ax.axhline(cap, color=RED, linestyle="--", linewidth=1.2, label=f"Capacity ({cap:,.0f} units/day)")
    if "outcome" in report:
        for outcome, colour, marker in [("hit", RED, "o"), ("false_alarm", ORANGE, "^"), ("missed", "black", "x")]:
            sub = report[report["outcome"] == outcome]
            y = sub["forecast"] if outcome != "missed" else sub["actual"]
            ax.scatter(sub["date"], y, color=colour, marker=marker, s=36, zorder=3,
                       label=f"{outcome.replace('_', ' ')} ({len(sub)})")
    else:
        sub = report[report["flagged"]]
        ax.scatter(sub["date"], sub["forecast"], color=RED, s=36, zorder=3, label=f"flagged ({len(sub)})")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(thousands())
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
