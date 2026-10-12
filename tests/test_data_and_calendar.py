"""Tests for cleaning, aggregation and the operating calendar."""

from __future__ import annotations

import pandas as pd

from demandcast.business_calendar import closed_mask
from demandcast.data import aggregate_daily, clean_transactions, to_series


def test_cleaning_rules(synthetic_lines) -> None:
    clean, report = clean_transactions(synthetic_lines)
    assert set(clean["invoice"]) == {"100", "105", "106"}
    assert report.duplicate_rows_removed == 1
    assert report.cancellation_rows_removed == 1
    assert report.cancelled_originals_removed == 1
    assert report.non_product_rows_removed == 1
    assert report.non_positive_rows_removed == 1
    assert report.clean_rows == 3


def test_aggregate_daily_fills_gaps_and_drops_partial_day(synthetic_lines) -> None:
    clean, _ = clean_transactions(synthetic_lines)
    daily = aggregate_daily(clean, drop_partial_last_day=False)
    assert list(daily["date"].dt.strftime("%Y-%m-%d")) == ["2022-03-01", "2022-03-02", "2022-03-03"]
    assert list(daily["units"]) == [10, 0, 11]
    # Last (partial) trading day 03-03 dropped by default; the calendar then ends at the
    # last remaining day with data (03-01), since 03-02 had no transactions.
    trimmed = aggregate_daily(clean)
    assert trimmed["date"].max() == pd.Timestamp("2022-03-01")


def test_calendar_known_closures() -> None:
    dates = pd.to_datetime(["2011-04-29", "2011-12-10", "2011-12-27", "2012-01-03",
                            "2011-04-24", "2011-12-05", "2011-12-11"])
    closed = closed_mask(pd.DatetimeIndex(dates)).tolist()
    # Royal wedding, Saturday, Christmas shutdown x2, Easter Sunday -> closed; Monday, Sunday -> open
    assert closed == [True, True, True, True, True, False, False]


def test_to_series_marks_closed_as_nan(synthetic_daily) -> None:
    s = to_series(synthetic_daily)
    assert s.index.freqstr == "D"
    assert s[~synthetic_daily.set_index("date")["is_open"]].isna().all()
    assert s.dropna().gt(0).all()
