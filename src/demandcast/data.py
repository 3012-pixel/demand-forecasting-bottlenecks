"""Load, clean and aggregate the UCI Online Retail II dataset.

Dataset: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D - licensed CC BY 4.0.

The raw workbook has two sheets ("Year 2009-2010", "Year 2010-2011") that
overlap for 1-9 December 2010. We aggregate cleaned line items to *daily
units shipped*, used here as a proxy for warehouse pick/pack volume.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from demandcast.business_calendar import open_mask

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}
PRODUCT_CODE = r"^\d{5}[A-Za-z]*$"  # e.g. 85123A; excludes POST, DOT, M, BANK CHARGES, gift vouchers...


@dataclass(frozen=True)
class CleaningReport:
    """Row counts removed at each cleaning step (for transparency in the README)."""

    raw_rows: int
    overlap_rows_removed: int
    duplicate_rows_removed: int
    cancellation_rows_removed: int
    cancelled_originals_removed: int
    non_positive_rows_removed: int
    non_product_rows_removed: int
    clean_rows: int


def download_raw(dest_dir: str | Path) -> Path:
    """Download and extract the UCI zip into ``dest_dir``; return the xlsx path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "online_retail_II.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(UCI_URL, zip_path)  # noqa: S310 - fixed https URL
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return dest_dir / "online_retail_II.xlsx"


def load_transactions(xlsx_path: str | Path) -> tuple[pd.DataFrame, int]:
    """Read both sheets, drop the Dec-2010 overlap, and standardise columns.

    Returns:
        (transactions, number of overlap rows removed)
    """
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    frames = [df.rename(columns=COLUMNS) for _, df in sorted(sheets.items())]
    removed = 0
    for i in range(1, len(frames)):
        start = frames[i]["invoice_date"].min()
        prev = frames[i - 1]
        overlap = prev["invoice_date"] >= start
        removed += int(overlap.sum())
        frames[i - 1] = prev[~overlap]
    return pd.concat(frames, ignore_index=True), removed


def _drop_cancelled_originals(sales: pd.DataFrame, cancels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove original order lines that were later cancelled.

    A cancellation (invoice starting with "C") is matched one-to-one to an
    order line with the same customer, stock code and absolute quantity.
    Cancelled orders are assumed never to have been picked.
    """
    keys = ["customer_id", "stock_code", "quantity"]
    c = cancels.dropna(subset=["customer_id"]).assign(quantity=lambda d: -d["quantity"])
    c = c.assign(_n=c.groupby(keys).cumcount())[keys + ["_n"]].assign(_cancelled=True)
    s = sales.assign(_n=sales.groupby(keys, dropna=False).cumcount())
    merged = s.merge(c, on=keys + ["_n"], how="left")
    keep = merged["_cancelled"].isna().to_numpy()
    return sales[keep], int((~keep).sum())


def clean_transactions(df: pd.DataFrame, overlap_removed: int = 0) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the cleaning rules and return (clean lines, report)."""
    raw_rows = len(df) + overlap_removed
    df = df.copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates()
    dups = before - len(df)

    is_cancel = df["invoice"].str.startswith("C")
    cancels, sales = df[is_cancel], df[~is_cancel]
    sales, cancelled_originals = _drop_cancelled_originals(sales, cancels)

    before = len(sales)
    sales = sales[sales["quantity"] > 0]
    non_positive = before - len(sales)

    before = len(sales)
    sales = sales[sales["stock_code"].str.match(PRODUCT_CODE)]
    non_product = before - len(sales)

    report = CleaningReport(
        raw_rows=raw_rows,
        overlap_rows_removed=overlap_removed,
        duplicate_rows_removed=dups,
        cancellation_rows_removed=int(is_cancel.sum()),
        cancelled_originals_removed=cancelled_originals,
        non_positive_rows_removed=non_positive,
        non_product_rows_removed=non_product,
        clean_rows=len(sales),
    )
    return sales.reset_index(drop=True), report


def aggregate_daily(lines: pd.DataFrame, drop_partial_last_day: bool = True) -> pd.DataFrame:
    """Aggregate clean lines to a complete daily calendar.

    Returns columns ``date``, ``units``, ``orders``, ``is_open``. Days with
    no transactions get 0 units. The source extract stops mid-day on its
    final date (9 Dec 2011, 12:50), so that partial day is dropped by default.
    """
    day = lines["invoice_date"].dt.normalize()
    g = lines.groupby(day).agg(units=("quantity", "sum"), orders=("invoice", "nunique"))
    if drop_partial_last_day:
        g = g.iloc[:-1]
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D", name="date")
    g = g.reindex(idx, fill_value=0)
    g["is_open"] = open_mask(idx).to_numpy()
    return g.reset_index()


def build_daily_dataset(xlsx_path: str | Path, out_csv: str | Path) -> tuple[pd.DataFrame, CleaningReport]:
    """Full pipeline: raw workbook -> cleaned -> daily CSV."""
    raw, overlap = load_transactions(xlsx_path)
    clean, report = clean_transactions(raw, overlap)
    daily = aggregate_daily(clean)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_csv, index=False)
    return daily, report


def load_daily(csv_path: str | Path) -> pd.DataFrame:
    """Load the processed daily dataset."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["is_open"] = df["is_open"].astype(bool)
    return df


def to_series(daily: pd.DataFrame) -> pd.Series:
    """Daily units as a Series with NaN on closed days (what models train on)."""
    s = daily.set_index("date")["units"].astype(float)
    s[~daily.set_index("date")["is_open"]] = float("nan")
    s.index = pd.DatetimeIndex(s.index, freq="D")
    return s
