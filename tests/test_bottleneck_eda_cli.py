"""Tests for bottleneck flags, EDA helpers, exports and the CLI."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from demandcast.bottleneck import capacity_threshold, flag_spikes, spike_skill
from demandcast.cli import main
from demandcast.eda import detect_outliers, weekday_profile


def test_capacity_threshold_explicit_and_percentile() -> None:
    s = pd.Series([10.0, 20.0, 30.0, float("nan")])
    assert capacity_threshold(s, 25) == 25
    assert capacity_threshold(s, None, 50) == 20
    with pytest.raises(ValueError):
        capacity_threshold(s, -1)
    with pytest.raises(ValueError):
        capacity_threshold(s, None, 100)


def test_flag_outcomes_and_skill() -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=4),
        "forecast": [120.0, 120.0, 80.0, 80.0],
        "actual": [130.0, 90.0, 110.0, 90.0],
    })
    out = flag_spikes(df, capacity=100)
    assert out["outcome"].tolist() == ["hit", "false_alarm", "missed", "ok"]
    assert out["excess_units_forecast"].tolist() == [20, 20, 0, 0]
    sk = spike_skill(out)
    assert sk["precision"] == 0.5 and sk["recall"] == 0.5


def test_flag_without_actuals() -> None:
    out = flag_spikes(pd.DataFrame({"date": pd.date_range("2023-01-02", periods=2), "forecast": [50.0, 150.0]}), 100)
    assert "outcome" not in out and out["flagged"].tolist() == [False, True]


def test_detect_outliers_finds_injected_spike(synthetic_daily) -> None:
    d = synthetic_daily.copy()
    spike_day = d.index[(d["date"] == "2023-03-09")][0]  # a Thursday
    d.loc[spike_day, "units"] *= 5
    out = detect_outliers(d)
    assert pd.Timestamp("2023-03-09") in set(out.loc[out["type"] == "spike", "date"])


def test_weekday_profile_excludes_closed(synthetic_daily) -> None:
    prof = weekday_profile(synthetic_daily)
    assert "Saturday" not in set(prof["weekday"])
    assert prof.set_index("weekday").loc["Thursday", "mean"] > prof.set_index("weekday").loc["Sunday", "mean"]


@pytest.fixture
def cli_config(tmp_path, synthetic_daily):
    csv = tmp_path / "daily.csv"
    synthetic_daily.to_csv(csv, index=False)
    cfg = {
        "data": {"raw_dir": str(tmp_path / "raw"), "daily_csv": str(csv)},
        "split": {"test_start": "2023-11-01"},
        "backtest": {"horizon_days": 7, "step_days": 7},
        "models": ["seasonal_naive", "linear_regression", "sarima"],
        "linear_regression": {"log_target": True},
        "sarima": {"order": [1, 0, 0], "seasonal_order": [0, 1, 1, 7]},
        "capacity": {"units_per_day": None, "percentile": 90},
        "forecast": {"model": "linear_regression", "days": 14},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_cli_end_to_end(cli_config, tmp_path, capsys) -> None:
    out = tmp_path / "out"
    assert main(["--config", str(cli_config), "--out", str(out), "all"]) == 0
    for name in ["model_metrics.csv", "bottleneck_report.csv", "future_bottleneck_report.csv",
                 "bottleneck_backtest.png", "eda_weekday.png", "powerbi/dim_date.csv",
                 "powerbi/fact_backtest_forecasts.csv"]:
        assert (out / name).exists(), name
    report = pd.read_csv(out / "bottleneck_report.csv")
    assert {"date", "forecast", "actual", "capacity", "flagged", "outcome"} <= set(report.columns)
    assert "Winner by MAE" in capsys.readouterr().out


def test_cli_prepare_missing_raw(cli_config, tmp_path) -> None:
    assert main(["--config", str(cli_config), "--out", str(tmp_path), "prepare"]) == 2
