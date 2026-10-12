"""Shared chart styling."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

BLUE = "#2a6f97"
ORANGE = "#e07a1f"
RED = "#c0392b"
GREEN = "#2e8b57"
GREY = "#8a8a8a"
MODEL_COLOURS = {"seasonal_naive": GREY, "linear_regression": ORANGE, "sarima": BLUE}


def thousands() -> FuncFormatter:
    """Axis formatter: 12,000 -> '12k'."""
    return FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k")


def style_axes(ax: Axes) -> None:
    """Remove top/right spines."""
    ax.spines[["top", "right"]].set_visible(False)
