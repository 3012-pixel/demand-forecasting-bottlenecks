"""Operating calendar for the warehouse: which days are planned closures.

In the source data the retailer does not ship on Saturdays, UK (England)
bank holidays, Easter Sunday or over a Christmas shutdown (24 Dec - 3 Jan).
These closures are known in advance, so every model forecasts zero on them
and they are excluded from accuracy metrics.
"""

from __future__ import annotations

import holidays
import pandas as pd
from dateutil.easter import easter

SATURDAY = 5


def closed_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """Return a boolean Series (indexed by ``dates``) that is True on closed days."""
    dates = pd.DatetimeIndex(dates).normalize()
    years = sorted(set(dates.year))
    bank = holidays.country_holidays("GB", subdiv="ENG", years=years)
    easter_sundays = {pd.Timestamp(easter(y)) for y in years}

    closed = []
    for d in dates:
        christmas_shutdown = (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 3)
        closed.append(
            d.dayofweek == SATURDAY
            or d.date() in bank
            or d in easter_sundays
            or christmas_shutdown
        )
    return pd.Series(closed, index=dates, name="is_closed")


def open_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """Inverse of :func:`closed_mask`."""
    return (~closed_mask(dates)).rename("is_open")
