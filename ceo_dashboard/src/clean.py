"""
Cleaning pipeline for the orders table.

Philosophy: every step records what it found and what it did about it in a
decision log, returned alongside the cleaned data. Nothing is silently
imputed or silently dropped -- a CEO dashboard that quietly guesses at
missing costs is worse than one that visibly excludes what it can't trust.

`clean_orders(df)` runs four named steps and returns `(cleaned_df,
decision_log)`:

1. `normalise_categories`   - strips whitespace, title-cases `category` so
                               "TOPS  " and "Tops" aren't counted separately.
2. `deduplicate_order_lines` - drops exact-duplicate rows (same order_line_id
                               appearing more than once), keeping the first.
3. `flag_missing_cogs`      - rows with a null `unit_cogs` can't have a
                               margin computed. Policy: never impute a cost
                               figure. These rows are kept (so revenue/return
                               reporting still includes them) but flagged
                               `valid_economics=False`, and every KPI in
                               `core.py` that touches cost excludes flagged
                               rows explicitly.
4. `flag_invalid_economics` - rows with a non-positive quantity or a
                               non-positive unit_price are not real sales
                               (a mis-tagged free-gift line, a negative-qty
                               data bug). Policy: flag them the same way as
                               missing COGS, never silently coerce the value.
"""

from __future__ import annotations

import pandas as pd

from . import schema


def normalise_categories(df: pd.DataFrame, log: list) -> pd.DataFrame:
    before = df[schema.CATEGORY].copy()
    cleaned = df[schema.CATEGORY].str.strip().str.title()
    n_changed = int((before != cleaned).sum())
    df = df.copy()
    df[schema.CATEGORY] = cleaned
    log.append({
        "step": "normalise_categories",
        "rows_affected": n_changed,
        "rule_applied": "strip whitespace, title-case",
    })
    return df


def deduplicate_order_lines(df: pd.DataFrame, log: list) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=schema.ORDER_LINE_ID, keep="first")
    n_removed = n_before - len(df)
    log.append({
        "step": "deduplicate_order_lines",
        "rows_affected": n_removed,
        "rule_applied": "drop exact duplicate order_line_id, keep first occurrence",
    })
    return df


def flag_missing_cogs(df: pd.DataFrame, log: list) -> pd.DataFrame:
    df = df.copy()
    missing = df[schema.UNIT_COGS].isna()
    if "valid_economics" not in df.columns:
        df["valid_economics"] = True
    df.loc[missing, "valid_economics"] = False
    log.append({
        "step": "flag_missing_cogs",
        "rows_affected": int(missing.sum()),
        "rule_applied": "unit_cogs is never imputed; rows flagged valid_economics=False "
                         "and excluded from every cost/margin KPI, kept for revenue/return reporting",
    })
    return df


def flag_invalid_economics(df: pd.DataFrame, log: list) -> pd.DataFrame:
    df = df.copy()
    if "valid_economics" not in df.columns:
        df["valid_economics"] = True
    invalid = (df[schema.QUANTITY] <= 0) | (df[schema.UNIT_PRICE] <= 0)
    df.loc[invalid, "valid_economics"] = False
    log.append({
        "step": "flag_invalid_economics",
        "rows_affected": int(invalid.sum()),
        "rule_applied": "quantity<=0 or unit_price<=0 is not a real sale; "
                         "rows flagged valid_economics=False, values never coerced",
    })
    return df


def clean_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline. Returns (cleaned_df, decision_log_df).

    `cleaned_df` always has a `valid_economics` boolean column: True unless
    a step above flagged the row. Every KPI function in `core.py` that
    depends on cost/margin filters to `valid_economics == True` explicitly
    -- revenue-only views (e.g. gross order counts) may still use all rows.
    """
    log: list = []
    out = df
    out = normalise_categories(out, log)
    out = deduplicate_order_lines(out, log)
    out = flag_missing_cogs(out, log)
    out = flag_invalid_economics(out, log)
    return out.reset_index(drop=True), pd.DataFrame(log)
