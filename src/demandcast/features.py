"""Calendar features for the regression model (known in advance, no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def days_to_christmas(dates: pd.DatetimeIndex) -> np.ndarray:
    """Days until the next 25 December (0 on Christmas Day)."""
    out = []
    for d in dates:
        xmas = pd.Timestamp(year=d.year, month=12, day=25)
        if d > xmas:
            xmas = pd.Timestamp(year=d.year + 1, month=12, day=25)
        out.append((xmas - d).days)
    return np.asarray(out, dtype=float)


def calendar_features(dates: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    """Build the design matrix.

    Features:
        - weekday dummies (Monday is the reference level)
        - month dummies (January is the reference level)
        - linear trend in years since ``origin``
        - ``xmas_ramp``: max(0, 1 - days_to_christmas / 120), a 0->1 ramp
          capturing the autumn peak build-up
    """
    dates = pd.DatetimeIndex(dates)
    X = pd.DataFrame(index=dates)
    for i, name in enumerate(WEEKDAYS[1:], start=1):
        X[f"dow_{name}"] = (dates.dayofweek == i).astype(float)
    for m in range(2, 13):
        X[f"month_{m:02d}"] = (dates.month == m).astype(float)
    X["trend_years"] = (dates - origin).days / 365.25
    X["xmas_ramp"] = np.clip(1 - days_to_christmas(dates) / 120, 0, None)
    return X
