"""Command-line interface.

Examples::

    demandcast prepare --download     # fetch UCI data and build data/processed/daily_units.csv
    demandcast eda
    demandcast backtest               # model comparison + bottleneck report + Power BI export
    demandcast forecast --capacity 30000
    demandcast all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from demandcast.bottleneck import capacity_threshold, flag_spikes, plot_bottlenecks, spike_skill  # noqa: E402
from demandcast.data import build_daily_dataset, download_raw, load_daily, to_series  # noqa: E402
from demandcast.eda import detect_outliers, eda_summary, plot_eda, weekday_profile  # noqa: E402
from demandcast.evaluation import error_breakdown, forecast_with_calendar, rolling_backtest, score  # noqa: E402
from demandcast.export import export_powerbi  # noqa: E402
from demandcast.models import build_models  # noqa: E402
from demandcast.plotting import MODEL_COLOURS, style_axes, thousands  # noqa: E402


def load_config(path: str | Path) -> dict:
    """Read the YAML config."""
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _out_dir(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_prepare(args: argparse.Namespace, cfg: dict) -> int:
    """Build the processed daily dataset from the raw UCI workbook."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    xlsx = raw_dir / "online_retail_II.xlsx"
    if args.download:
        xlsx = download_raw(raw_dir)
    if not xlsx.exists():
        print(f"error: {xlsx} not found. Run with --download or see data/README.md", file=sys.stderr)
        return 2
    print("Reading workbook (about 1 minute)...")
    daily, report = build_daily_dataset(xlsx, cfg["data"]["daily_csv"])
    for k, v in report.__dict__.items():
        print(f"  {k:30s} {v:>10,}")
    print(f"Wrote {cfg['data']['daily_csv']} ({len(daily)} days)")
    return 0


def cmd_eda(args: argparse.Namespace, cfg: dict) -> int:
    """Run EDA: summary stats, outlier list and charts."""
    out = _out_dir(args)
    daily = load_daily(cfg["data"]["daily_csv"])
    e = cfg.get("eda", {})
    hi, lo = e.get("outlier_ratio_high", 2.5), e.get("outlier_ratio_low", 0.4)
    summary = eda_summary(daily, hi, lo)
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    detect_outliers(daily, hi, lo).to_csv(out / "eda_outliers.csv", index=False)
    prof = weekday_profile(daily)
    prof.to_csv(out / "eda_weekday_profile.csv", index=False)
    paths = plot_eda(daily, out, hi, lo)
    print(json.dumps(summary, indent=2))
    print(prof.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("Charts:", ", ".join(str(p) for p in paths))
    return 0


def _plot_backtest(bt: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=150)
    first = bt[bt["model"] == bt["model"].iloc[0]]
    ax.plot(first["date"], first["actual"], color="black", linewidth=1.3, label="Actual")
    for name, g in bt.groupby("model", sort=False):
        ax.plot(g["date"], g["forecast"], color=MODEL_COLOURS.get(name), linewidth=1, alpha=0.9, label=name)
    ax.set_title("Rolling 7-day-ahead forecasts vs actual (test period)")
    ax.yaxis.set_major_formatter(thousands())
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def cmd_backtest(args: argparse.Namespace, cfg: dict) -> int:
    """Rolling backtest, metrics, failure analysis, bottleneck report, Power BI export."""
    out = _out_dir(args)
    daily = load_daily(cfg["data"]["daily_csv"])
    series = to_series(daily)
    test_start = pd.Timestamp(cfg["split"]["test_start"])
    bcfg = cfg["backtest"]
    models = build_models(cfg)
    print(f"Backtesting {', '.join(m.name for m in models)} from {test_start:%Y-%m-%d} "
          f"(horizon {bcfg['horizon_days']}d, refit every {bcfg['step_days']}d)...")
    bt = rolling_backtest(models, series, test_start, bcfg["horizon_days"], bcfg["step_days"])
    metrics = score(bt)
    winner = str(metrics.iloc[0]["model"])

    bt.to_csv(out / "backtest_forecasts.csv", index=False)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    error_breakdown(bt, "weekday").to_csv(out / "error_by_weekday.csv", index=False)
    weekly = error_breakdown(bt, "week")
    weekly.to_csv(out / "error_by_week.csv", index=False)
    _plot_backtest(bt, out / "backtest_forecast_vs_actual.png")

    capacity = capacity_threshold(series[series.index < test_start], args.capacity
                                  if args.capacity is not None else cfg["capacity"].get("units_per_day"),
                                  cfg["capacity"].get("percentile", 90))
    wb = bt[bt["model"] == winner][["date", "forecast", "actual", "is_open"]]
    report = flag_spikes(wb, capacity)
    report.to_csv(out / "bottleneck_report.csv", index=False)
    plot_bottlenecks(report, out / "bottleneck_backtest.png",
                     f"Capacity check, test period ({winner} forecasts, one week ahead)")
    skill = spike_skill(report[report["is_open"]])
    baseline_report = flag_spikes(bt[bt["model"] == "seasonal_naive"][["date", "forecast", "actual", "is_open"]],
                                  capacity)
    base_skill = spike_skill(baseline_report[baseline_report["is_open"]])

    export_powerbi(out / "powerbi", daily, bt, metrics, report)

    print("\n=== Accuracy on open days (lower is better) ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print(f"\nWinner by MAE: {winner}")
    print("\nWorst weeks for the winner (sMAPE):")
    ww = weekly[weekly["model"] == winner].nlargest(3, "smape_pct")
    print(ww.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print(f"\n=== Bottlenecks: capacity {capacity:,.0f} units/day ===")
    print(f"  {winner:16s}: {skill}")
    print(f"  {'seasonal_naive':16s}: {base_skill}")
    print(f"\nOutputs in {out}/ (bottleneck_report.csv, charts, powerbi/)")
    return 0


def cmd_forecast(args: argparse.Namespace, cfg: dict) -> int:
    """Fit on all history and forecast forward, flagging capacity breaches."""
    out = _out_dir(args)
    daily = load_daily(cfg["data"]["daily_csv"])
    series = to_series(daily)
    fcfg = cfg.get("forecast", {})
    name = args.model or fcfg.get("model", "sarima")
    model = next(m for m in build_models({**cfg, "models": [name]}))
    model.fit(series)
    days = args.days or fcfg.get("days", 28)
    dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=days, freq="D")
    fc = forecast_with_calendar(model, dates)
    capacity = capacity_threshold(series, args.capacity if args.capacity is not None
                                  else cfg["capacity"].get("units_per_day"), cfg["capacity"].get("percentile", 90))
    report = flag_spikes(pd.DataFrame({"date": dates, "forecast": fc.to_numpy()}), capacity)
    report.to_csv(out / "future_bottleneck_report.csv", index=False)
    plot_bottlenecks(report, out / "bottleneck_future.png", f"Next {days} days: {name} forecast vs capacity")
    print(report.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print(f"\n{int(report['flagged'].sum())} day(s) flagged above {capacity:,.0f} units/day")
    return 0


def cmd_all(args: argparse.Namespace, cfg: dict) -> int:
    """EDA + backtest + forward forecast."""
    for fn in (cmd_eda, cmd_backtest, cmd_forecast):
        code = fn(args, cfg)
        if code:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    load_dotenv()
    default_out = os.getenv("DEMANDCAST_OUTPUT_DIR", "outputs")
    p = argparse.ArgumentParser(prog="demandcast", description="Demand forecasting and bottleneck detection.")
    p.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    p.add_argument("--out", default=default_out, help=f"output folder (default: {default_out})")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prepare", help="build the daily dataset from the raw UCI workbook")
    pr.add_argument("--download", action="store_true", help="download the raw data from UCI first (~45 MB)")
    pr.set_defaults(func=cmd_prepare)

    sub.add_parser("eda", help="exploratory analysis and charts").set_defaults(func=cmd_eda)

    for name, fn, help_ in [("backtest", cmd_backtest, "compare models and build the bottleneck report"),
                            ("forecast", cmd_forecast, "forecast forward and flag capacity breaches"),
                            ("all", cmd_all, "eda + backtest + forecast")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--capacity", type=float, help="daily capacity in units (overrides config)")
        sp.add_argument("--model", help="model for the forward forecast (forecast/all)")
        sp.add_argument("--days", type=int, help="forward forecast length in days")
        sp.set_defaults(func=fn)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        return args.func(args, cfg)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
