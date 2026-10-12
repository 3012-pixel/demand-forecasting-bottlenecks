# Data

## Source and licence

**Online Retail II**, UCI Machine Learning Repository.
Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

Licence: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This is real transaction data from a UK-based online retailer of giftware (many of its customers are wholesalers), covering 1 Dec 2009 to 9 Dec 2011.

## What is in this repo

| Path | Contents | In git? |
|------|----------|---------|
| `raw/online_retail_II.xlsx` | Original workbook (~45 MB). Download with `demandcast prepare --download` | No (`.gitignore`) |
| `processed/daily_units.csv` | **Derived** daily aggregate: `date, units, orders, is_open` | Yes (CC BY 4.0 permits redistribution of adapted material with attribution) |

## Changes made to the source (CC BY 4.0 "indicate if changes were made")

The daily file is an aggregation of cleaned line items. The raw data was changed as follows:

1. The two sheets overlap for 1–9 Dec 2010. These rows were kept once (22,523 overlap rows dropped).
2. Exact duplicate rows were removed (11,812).
3. Cancellation lines (invoice prefix `C`, 19,104) were removed, together with the original order line each one cancels (7,067). Matching is one-to-one on customer, stock code and quantity.
4. Non-positive quantities that are not cancellations (stock adjustments, 3,393) were removed.
5. Non-product codes (`POST`, `DOT`, `M`, `BANK CHARGES`, gift vouchers and similar, 4,546) were removed.
6. The last date (9 Dec 2011) was dropped because the extract stops at 12:50 that day.
7. Lines were aggregated to **units per calendar day**, and an `is_open` flag was added from the operating calendar (Saturdays, England bank holidays, Easter Sunday and a 24 Dec–3 Jan shutdown are closed).

Result: 998,926 clean lines over 738 calendar days (602 open).

"Units shipped per day" is used as a **proxy for warehouse pick/pack workload**. The dataset contains no warehouse capacity data. The capacity threshold in `config.yaml` is an assumed planning parameter.
