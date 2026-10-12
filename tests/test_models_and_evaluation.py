"""Tests for metrics, models and the rolling backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandcast.data import to_series
from demandcast.evaluation import mae, mape, rmse, rolling_backtest, score, smape
from demandcast.features import calendar_features, days_to_christmas
from demandcast.models import CalendarRegression, Sarima, SeasonalNaive, build_models


def test_metrics_hand_values() -> None:
    a, f = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert mae(a, f) == pytest.approx(15)
    assert rmse(a, f) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(a, f) == pytest.approx((10 + 10) / 2)
    assert smape(a, f) == pytest.approx(np.mean([20 / 210, 40 / 380]) * 100)


def test_mape_ignores_zero_actuals_and_smape_handles_double_zero() -> None:
    assert mape(np.array([0.0, 100.0]), np.array([5.0, 90.0])) == pytest.approx(10)
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0


def test_seasonal_naive_skips_closed_days() -> None:
    idx = pd.date_range("2023-01-02", periods=21, freq="D")  # starts Monday
    s = pd.Series(np.arange(21, dtype=float), index=idx)
    s.iloc[14] = np.nan  # Monday of week 3 closed (e.g. bank holiday)
    m = SeasonalNaive().fit(s)
    pred = m.predict(pd.date_range("2023-01-23", periods=2, freq="D"))  # next Monday, Tuesday
    assert pred.iloc[0] == 7.0  # falls back two weeks to an open Monday
    assert pred.iloc[1] == 15.0


def test_seasonal_naive_long_horizon_uses_history() -> None:
    idx = pd.date_range("2023-01-02", periods=14, freq="D")
    s = pd.Series(np.arange(14, dtype=float), index=idx)
    pred = SeasonalNaive().fit(s).predict(pd.date_range("2023-01-30", periods=1, freq="D"))
    assert pred.iloc[0] == 7.0  # 3 weeks after last Monday in history


def test_calendar_features_no_leakage_and_shape() -> None:
    dates = pd.date_range("2023-12-20", periods=10, freq="D")
    X = calendar_features(dates, pd.Timestamp("2023-01-01"))
    assert X.shape == (10, 6 + 11 + 2)
    assert days_to_christmas(pd.DatetimeIndex(["2023-12-25", "2023-12-26"])).tolist() == [0, 365]
    assert X["xmas_ramp"].between(0, 1).all()


def test_regression_learns_weekday_pattern(synthetic_daily) -> None:
    s = to_series(synthetic_daily)
    m = CalendarRegression().fit(s[s.index < "2023-06-01"])
    pred = m.predict(pd.date_range("2023-06-05", periods=7, freq="D"))  # Mon..Sun
    assert pred.iloc[3] > pred.iloc[6]  # Thursday > Sunday
    assert (pred >= 0).all()


def test_sarima_forecasts_positive_after_history(synthetic_daily) -> None:
    s = to_series(synthetic_daily)
    m = Sarima().fit(s[s.index < "2023-06-01"])
    pred = m.predict(pd.date_range("2023-06-01", periods=7, freq="D"))
    open_days = pred.drop(pred.index[pred.index.dayofweek == 5])
    assert np.isfinite(open_days).all() and (open_days > 0).all()
    with pytest.raises(ValueError):
        m.predict(pd.date_range("2023-05-01", periods=3, freq="D"))


def test_backtest_no_lookahead_and_closed_days_zero(synthetic_daily) -> None:
    s = to_series(synthetic_daily)
    models = [SeasonalNaive(), CalendarRegression()]
    bt = rolling_backtest(models, s, "2023-11-01", horizon=7, step=7)
    assert set(bt["model"]) == {"seasonal_naive", "linear_regression"}
    assert (bt["date"] >= bt["origin"]).all()
    assert (bt["date"] < bt["origin"] + pd.Timedelta(days=7)).all()
    assert (bt.loc[~bt["is_open"], "forecast"] == 0).all()
    # each date forecast exactly once per model when horizon == step
    assert not bt.duplicated(["date", "model"]).any()


def test_score_skill_vs_baseline(synthetic_daily) -> None:
    s = to_series(synthetic_daily)
    bt = rolling_backtest([SeasonalNaive(), CalendarRegression()], s, "2023-10-01")
    sc = score(bt).set_index("model")
    assert sc.loc["seasonal_naive", "mae_skill_vs_baseline_pct"] == pytest.approx(0)
    assert {"mae", "rmse", "mape_pct", "smape_pct", "bias_pct"} <= set(sc.columns)


def test_build_models_from_config() -> None:
    cfg = {"models": ["seasonal_naive", "sarima"], "sarima": {"order": [1, 0, 0], "seasonal_order": [0, 1, 1, 7]}}
    models = build_models(cfg)
    assert [m.name for m in models] == ["seasonal_naive", "sarima"]
    assert models[1].seasonal_order == (0, 1, 1, 7)
    with pytest.raises(ValueError):
        build_models({"models": ["prophet"]})
