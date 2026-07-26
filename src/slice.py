"""Grouping utilities for margin already computed by margin.py.

Every function here expects a DataFrame that already has
margin.ALLOCATED_COST / margin.MARGIN / margin.MARGIN_PCT columns (i.e.
the output of margin.compute_margin), and returns a tidy DataFrame -
one row per group, plain columns, no MultiIndex.

Groups are kept even when the grouping column is null (dropna=False):
a customer_segment or origin_id that's missing in the source data is a
real, visible bucket, not something to silently drop.

margin_pct on a group is recomputed from the group's summed margin and
revenue, not averaged from per-transaction margin_pct - averaging
percentages across transactions of very different size would misstate
the group's actual margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import margin, schema


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            revenue=(schema.REVENUE, "sum"),
            allocated_cost=(margin.ALLOCATED_COST, "sum"),
            margin=(margin.MARGIN, "sum"),
            txn_count=(schema.TXN_ID, "count"),
        )
        .reset_index()
    )
    safe_revenue = grouped["revenue"].replace(0, np.nan)
    grouped[margin.MARGIN_PCT] = grouped["margin"] / safe_revenue
    return grouped


def by_customer(df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(df, [schema.CUSTOMER_ID])


def by_route(df: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(df, [schema.ORIGIN_ID, schema.DESTINATION_ID])


def by_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[schema.PERIOD_MONTH] = margin.period_month(df[schema.TXN_DATE])
    return _aggregate(df, [schema.PERIOD_MONTH])


def by_two_way(df: pd.DataFrame, dim_a: str, dim_b: str) -> pd.DataFrame:
    """Cross any two columns already present on df - e.g. customer_id x
    origin_id, or origin_id x destination_id x-referenced against
    period_month after calling by_month's date-bucketing yourself.
    """
    return _aggregate(df, [dim_a, dim_b])


def concentration(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = schema.REVENUE,
    top_n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """What share of value_col's total sits in the top_n groups of
    group_col. ascending=False ranks the largest contributors first
    (e.g. revenue concentration); ascending=True ranks the smallest/most
    negative first (e.g. which few groups hold most of the loss).

    If value_col has mixed signs across groups, cumulative_share_of_total
    can behave non-intuitively (e.g. exceed 1 or go negative) - filter df
    to one sign first (e.g. only groups with negative margin) if that's
    what you're trying to isolate.
    """
    totals = df.groupby(group_col, dropna=False)[value_col].sum().sort_values(ascending=ascending)
    total_value = totals.sum()
    top = totals.head(top_n)

    result = top.reset_index()
    result.columns = [group_col, value_col]
    result["cumulative_" + value_col] = top.cumsum().values
    result["cumulative_share_of_total"] = (top.cumsum() / total_value).values
    return result
