"""Per-transaction margin computation with pluggable allocation strategies.

Real cost data is incomplete: ``direct_cost`` is missing on some lines,
and a large share of true cost - driver salaries, EMIs, rent, office -
never appears on a line at all. It only exists as the period overhead
pool (``schema.OVERHEAD_SCHEMA``). Allocation strategies decide how
that unattributed pool gets spread across transactions; they do not
change what's known - direct_cost is always used as-is when present.

Every strategy shares the same signature so they can be swapped and
compared:

    strategy(ledger_df: pd.DataFrame, overhead_df: pd.DataFrame) -> pd.DataFrame

and returns ledger_df with three added columns: ALLOCATED_COST, MARGIN,
MARGIN_PCT. These are derived output columns, not part of the canonical
input schema, so they're defined here rather than in schema.py.

This module computes margin on whatever rows it is given. Deciding
whether to exclude non-"sale" txn_type rows (credit_note, cancelled)
before allocating is left to the caller.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src import schema

ALLOCATED_COST = "allocated_cost"
MARGIN = "margin"
MARGIN_PCT = "margin_pct"

Allocator = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def _require_datetime(df: pd.DataFrame, col: str) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        raise TypeError(
            f"{col} must already be parsed to datetime64 (run clean.normalise_dates first), "
            f"got {df[col].dtype}"
        )


def _period_month(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp()


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df[MARGIN] = df[schema.REVENUE] - df[ALLOCATED_COST]
    safe_revenue = df[schema.REVENUE].replace(0, np.nan)
    df[MARGIN_PCT] = df[MARGIN] / safe_revenue
    return df


def direct_cost_only(ledger_df: pd.DataFrame, overhead_df: pd.DataFrame) -> pd.DataFrame:
    """Floor strategy: allocated_cost is exactly direct_cost. The
    overhead pool is ignored entirely. Rows with missing direct_cost get
    a null allocated_cost and null margin - this strategy never guesses.
    """
    df = ledger_df.copy()
    df[ALLOCATED_COST] = df[schema.DIRECT_COST]
    return _finalize(df)


def _allocate_overhead(df: pd.DataFrame, overhead_df: pd.DataFrame, weight_col: str) -> pd.Series:
    """Spread each period's total overhead pool across df's rows for
    that period, in proportion to each row's share of weight_col. A
    period with zero (or negative) total weight cannot be allocated by
    share and yields null for every row in that period, rather than a
    divide-by-zero or a silently wrong number.
    """
    period = _period_month(df[schema.TXN_DATE])
    period_total_weight = df.groupby(period)[weight_col].transform("sum")
    safe_weight = period_total_weight.where(period_total_weight > 0, np.nan)
    weight_share = df[weight_col] / safe_weight

    overhead_by_period = overhead_df.groupby(schema.PERIOD_MONTH)[schema.AMOUNT].sum()
    period_overhead = period.map(overhead_by_period)

    return weight_share * period_overhead


def proportional_by_revenue(ledger_df: pd.DataFrame, overhead_df: pd.DataFrame) -> pd.DataFrame:
    """Adds each transaction's revenue-share of its period's overhead
    pool on top of any known direct_cost. Missing direct_cost is
    treated as 0 for the known-cost component - the point of this
    strategy is to estimate the unattributable piece, not the
    attributable one, so a missing direct_cost is not itself imputed.
    """
    df = ledger_df.copy()
    _require_datetime(df, schema.TXN_DATE)
    allocated_overhead = _allocate_overhead(df, overhead_df, weight_col=schema.REVENUE)
    df[ALLOCATED_COST] = df[schema.DIRECT_COST].fillna(0.0) + allocated_overhead
    return _finalize(df)


def per_unit_by_quantity(ledger_df: pd.DataFrame, overhead_df: pd.DataFrame) -> pd.DataFrame:
    """Same as proportional_by_revenue, but spreads each period's
    overhead pool by quantity share instead of revenue share - useful
    when cost drivers track volume (tonnes, trips) rather than price.
    """
    df = ledger_df.copy()
    _require_datetime(df, schema.TXN_DATE)
    allocated_overhead = _allocate_overhead(df, overhead_df, weight_col=schema.QUANTITY)
    df[ALLOCATED_COST] = df[schema.DIRECT_COST].fillna(0.0) + allocated_overhead
    return _finalize(df)


STRATEGIES: dict[str, Allocator] = {
    "direct_cost_only": direct_cost_only,
    "proportional_by_revenue": proportional_by_revenue,
    "per_unit_by_quantity": per_unit_by_quantity,
}


def compute_margin(ledger_df: pd.DataFrame, overhead_df: pd.DataFrame, strategy: str | Allocator) -> pd.DataFrame:
    """Look up strategy by name in STRATEGIES (or call it directly if
    it's already an Allocator) and run it.
    """
    fn = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    return fn(ledger_df, overhead_df)
