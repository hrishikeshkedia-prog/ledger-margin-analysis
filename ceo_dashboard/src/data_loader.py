"""
Swappable data loader.

Purpose
-------
Every calculation module in this project reads canonical column names
defined in `schema.py` (e.g. `schema.UNIT_PRICE`). This module is the ONLY
place that touches raw file headers. To plug in a real CSV export (a real
Shopify/POS orders export, a real ad-spend report, a real WMS inventory
export) instead of the synthetic data:

    1. Open the export and note its column headers.
    2. Copy the relevant DEFAULT_*_MAPPING dict below.
    3. Edit the KEYS (raw headers) to match the real file's headers.
       Leave the VALUES (canonical names) unchanged.
    4. Pass your new mapping dict into `load_orders(path, mapping=...)`
       (or the equivalent loader function).

No analysis code changes. If a real file is missing a column the engine
needs, `schema.validate_columns` raises immediately with the missing
column's name, rather than letting a KPI silently compute on absent data.
"""

from __future__ import annotations

import pandas as pd

from . import schema

DATE_COLUMNS_BY_TABLE = {
    "orders": [schema.ORDER_DATE, schema.RETURN_DATE],
    "marketing_spend": [],
    "inventory_snapshots": [],
}


def _load_and_map(path: str, mapping: dict, expected_columns: list, table_name: str,
                   date_columns: list | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # Only rename columns that are present in the mapping AND the file --
    # a real export may have extra columns we don't need, which is fine.
    rename = {raw_col: canon_col for raw_col, canon_col in mapping.items() if raw_col in raw.columns}
    df = raw.rename(columns=rename)
    schema.validate_columns(df, expected_columns, table_name)
    df = df[expected_columns].copy()
    for col in (date_columns or []):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_orders(path: str, mapping: dict | None = None) -> pd.DataFrame:
    mapping = mapping or schema.DEFAULT_ORDERS_MAPPING
    return _load_and_map(path, mapping, schema.ORDERS_SCHEMA, "orders",
                          date_columns=DATE_COLUMNS_BY_TABLE["orders"])


def load_marketing_spend(path: str, mapping: dict | None = None) -> pd.DataFrame:
    mapping = mapping or schema.DEFAULT_MARKETING_MAPPING
    return _load_and_map(path, mapping, schema.MARKETING_SCHEMA, "marketing_spend")


def load_inventory_snapshots(path: str, mapping: dict | None = None) -> pd.DataFrame:
    mapping = mapping or schema.DEFAULT_INVENTORY_MAPPING
    return _load_and_map(path, mapping, schema.INVENTORY_SCHEMA, "inventory_snapshots")


def load_all(data_dir: str = "data/synthetic", mappings: dict | None = None) -> dict[str, pd.DataFrame]:
    """Load all three tables from a directory of CSVs named
    orders.csv / marketing_spend.csv / inventory_snapshots.csv.

    `mappings`, if given, is a dict like
    {"orders": {...}, "marketing_spend": {...}, "inventory_snapshots": {...}}
    letting the caller override any subset of the default (identity)
    mappings -- e.g. when only the orders export has different headers.
    """
    import os

    mappings = mappings or {}
    return {
        "orders": load_orders(os.path.join(data_dir, "orders.csv"),
                               mappings.get("orders")),
        "marketing_spend": load_marketing_spend(os.path.join(data_dir, "marketing_spend.csv"),
                                                  mappings.get("marketing_spend")),
        "inventory_snapshots": load_inventory_snapshots(os.path.join(data_dir, "inventory_snapshots.csv"),
                                                          mappings.get("inventory_snapshots")),
    }
