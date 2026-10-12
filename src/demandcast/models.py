"""Forecasting models with a common ``fit(history)`` / ``predict(dates)`` interface.

``history`` is a daily Series with a DatetimeIndex (freq "D") and NaN on
planned-closure days. Predictions are returned for the requested dates;
the caller (``evaluation.forecast_with_calendar``) forces closed days to 0.

APIs used (verified against installed versions / official docs):
- ``statsmodels.tsa.statespace.sarimax.SARIMAX(endog, order, seasonal_order, ...)``,
  ``.fit(disp=False)``, ``results.get_forecast(steps).predicted_mean``.
  SARIMAX handles NaN in ``endog`` via the Kalman filter.
- ``sklearn.linear_model.LinearRegression().fit(X, y).predict(X)``.
"""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

from demandcast.features import calendar_features


class Forecaster(Protocol):
    """Minimal forecasting interface."""

    name: str

    def fit(self, history: pd.Series) -> "Forecaster": ...

    def predict(self, dates: pd.DatetimeIndex) -> pd.Series: ...


class SeasonalNaive:
    """Same weekday last week (skipping back further if that day was closed).

    The standard benchmark: any model worth deploying must beat it.
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 7, max_lookback_seasons: int = 8) -> None:
        self.season = season
        self.max_lookback_seasons = max_lookback_seasons
        self._history: pd.Series | None = None

    def fit(self, history: pd.Series) -> "SeasonalNaive":
        self._history = history
        return self

    def predict(self, dates: pd.DatetimeIndex) -> pd.Series:
        assert self._history is not None, "call fit() first"
        h = self._history
        end = h.index.max()
        out = []
        for d in dates:
            value = np.nan
            # step back whole seasons until we land inside history on an open day
            k = max(1, int(np.ceil((d - end).days / self.season)))
            for j in range(k, k + self.max_lookback_seasons):
                candidate = d - pd.Timedelta(days=self.season * j)
                v = h.get(candidate, np.nan)
                if not np.isnan(v):
                    value = v
                    break
            out.append(value)
        return pd.Series(out, index=dates, name=self.name)


class CalendarRegression:
    """OLS on calendar features (weekday, month, trend, Christmas run-up).

    Fitted on log(units) by default because volume is right-skewed and the
    weekday pattern is multiplicative.
    """

    name = "linear_regression"

    def __init__(self, log_target: bool = True) -> None:
        self.log_target = log_target
        self._model = LinearRegression()
        self._origin: pd.Timestamp | None = None
        self._smear = 1.0

    def fit(self, history: pd.Series) -> "CalendarRegression":
        y = history.dropna()
        self._origin = history.index.min()
        X = calendar_features(y.index, self._origin)
        target = np.log(y) if self.log_target else y
        self._model.fit(X, target)
        if self.log_target:  # Duan smearing correction for log back-transform bias
            resid = target - self._model.predict(X)
            self._smear = float(np.mean(np.exp(resid)))
        return self

    def predict(self, dates: pd.DatetimeIndex) -> pd.Series:
        assert self._origin is not None, "call fit() first"
        pred = self._model.predict(calendar_features(dates, self._origin))
        if self.log_target:
            pred = np.exp(pred) * self._smear
        return pd.Series(np.clip(pred, 0, None), index=dates, name=self.name)


class Sarima:
    """Seasonal ARIMA with weekly seasonality, fitted on log(units).

    Closed days are NaN and handled by the state-space Kalman filter, so the
    weekly cycle stays aligned on the calendar.
    """

    name = "sarima"

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0),
                 seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 7),
                 log_target: bool = True) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.log_target = log_target
        self._result = None
        self._end: pd.Timestamp | None = None

    def fit(self, history: pd.Series) -> "Sarima":
        y = np.log(history) if self.log_target else history
        model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on refits
            self._result = model.fit(disp=False)
        self._end = history.index.max()
        return self

    def predict(self, dates: pd.DatetimeIndex) -> pd.Series:
        assert self._result is not None and self._end is not None, "call fit() first"
        steps = int((dates.max() - self._end).days)
        if steps < 1:
            raise ValueError("SARIMA can only forecast dates after the training period")
        fc = self._result.get_forecast(steps=steps)
        mean = pd.Series(np.asarray(fc.predicted_mean),
                         index=pd.date_range(self._end + pd.Timedelta(days=1), periods=steps, freq="D"))
        if self.log_target:
            # lognormal mean correction using the forecast variance
            # Saturdays are never observed, so their forecast variance is huge and
            # exp() overflows; those days are forced to 0 by the calendar anyway.
            var = pd.Series(np.asarray(fc.var_pred_mean), index=mean.index)
            with np.errstate(over="ignore"):
                mean = np.exp(mean + var / 2)
        return mean.reindex(dates).clip(lower=0).rename(self.name)


def build_models(cfg: dict) -> list[Forecaster]:
    """Instantiate the models listed in the config."""
    registry = {
        "seasonal_naive": lambda: SeasonalNaive(),
        "linear_regression": lambda: CalendarRegression(**cfg.get("linear_regression", {})),
        "sarima": lambda: Sarima(**{k: tuple(v) if isinstance(v, list) else v
                                    for k, v in cfg.get("sarima", {}).items()}),
    }
    unknown = set(cfg["models"]) - set(registry)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    return [registry[m]() for m in cfg["models"]]
