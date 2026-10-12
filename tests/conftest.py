"""Shared fixtures. All test data here is SYNTHETIC and generated in code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandcast.business_calendar import open_mask

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_daily() -> pd.DataFrame:
    """Two years of synthetic daily volume with a weekday pattern, a Q4 bump and noise."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-04", "2023-12-15", freq="D")
    weekday_factor = np.array([1.1, 1.0, 1.0, 1.2, 0.9, 0.0, 0.5])[dates.dayofweek]
    q4 = np.where(dates.month >= 9, 1.5, 1.0)
    units = 10_000 * weekday_factor * q4 * rng.lognormal(0, 0.1, len(dates))
    is_open = open_mask(dates).to_numpy()
    units = np.where(is_open, units, 0).round()
    return pd.DataFrame({"date": dates, "units": units, "orders": (units / 50).round(), "is_open": is_open})


@pytest.fixture
def synthetic_lines() -> pd.DataFrame:
    """Tiny synthetic transaction table covering each cleaning rule."""
    rows = [
        # invoice, stock_code, qty, date, customer
        ("100", "85123A", 10, "2022-03-01 09:00", 1.0),
        ("100", "85123A", 10, "2022-03-01 09:00", 1.0),   # exact duplicate -> removed
        ("101", "22001", 5, "2022-03-01 10:00", 2.0),     # later cancelled -> removed
        ("C102", "22001", -5, "2022-03-02 10:00", 2.0),   # cancellation -> removed
        ("103", "POST", 1, "2022-03-02 11:00", 3.0),      # non-product -> removed
        ("104", "22002", -3, "2022-03-02 12:00", None),   # non-positive, not a C invoice -> removed
        ("105", "22003", 7, "2022-03-03 12:00", None),    # kept (no customer id is fine)
        ("106", "22004", 4, "2022-03-03 13:00", 4.0),     # kept
    ]
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "quantity", "invoice_date", "customer_id"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["description"] = "synthetic"
    df["price"] = 1.0
    df["country"] = "United Kingdom"
    return df
