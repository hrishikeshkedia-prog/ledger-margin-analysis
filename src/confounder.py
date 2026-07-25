"""Helpers for trying to kill a finding, not confirm one.

Each function here takes an apparent pattern (a gap between two groups,
a period-over-period change, a shift in a metric) and gives you a way
to check whether it survives an obvious alternative explanation:
- is it actually explained by mix (Simpson's paradox)?
- is it actually one entity dominating the numbers?
- is it actually a volume/mix shift rather than a real rate change?
- is it actually a payment-timing artifact rather than a real change
  in the underlying business?

None of these functions render a verdict - they return a comparison
table. You decide whether the apparent pattern holds up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import margin, schema


def stratified_comparison(
    df: pd.DataFrame,
    fixed_col: str,
    compare_col: str,
    group_a,
    group_b,
    value_col: str = margin.MARGIN,
    revenue_col: str = schema.REVENUE,
) -> pd.DataFrame:
    """Compare value_col between compare_col == group_a and group_b,
    both overall (an OVERALL row) and within each stratum of fixed_col.
    A gap that's large OVERALL but shrinks or flips within every
    stratum is explained by an uneven mix of fixed_col between group_a
    and group_b, not by group_a/group_b themselves.

    rate = sum(value_col) / sum(revenue_col) per group, e.g. margin_pct
    when value_col=margin.MARGIN (the default).
    """
    subset = df[df[compare_col].isin([group_a, group_b])]

    def _rates(frame: pd.DataFrame) -> pd.Series:
        g = frame.groupby(compare_col, dropna=False)
        value = g[value_col].sum()
        revenue = g[revenue_col].sum()
        return value / revenue.replace(0, np.nan)

    def _counts(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby(compare_col, dropna=False)[schema.TXN_ID].count()

    rows = []
    strata = [("OVERALL", subset)] + list(subset.groupby(fixed_col, dropna=False))
    for label, stratum_df in strata:
        rates = _rates(stratum_df)
        counts = _counts(stratum_df)
        rate_a = rates.get(group_a, np.nan)
        rate_b = rates.get(group_b, np.nan)
        rows.append(
            {
                fixed_col: label,
                f"rate_{group_a}": rate_a,
                f"txn_count_{group_a}": int(counts.get(group_a, 0)),
                f"rate_{group_b}": rate_b,
                f"txn_count_{group_b}": int(counts.get(group_b, 0)),
                "gap": rate_a - rate_b,
            }
        )
    return pd.DataFrame(rows)


def period_over_period_excluding(
    df: pd.DataFrame,
    entity_col: str,
    entity_value,
    period_a,
    period_b,
    value_col: str = schema.REVENUE,
    period_col: str | None = None,
) -> pd.DataFrame:
    """Compare value_col between period_a and period_b, both including
    and excluding rows where entity_col == entity_value. If excluding
    that one entity flattens or reverses the change, the entity was
    driving the apparent pattern rather than something broader.

    period_col defaults to a month bucket derived from txn_date; pass
    period_a/period_b as values comparable to whatever period_col
    actually holds (e.g. pd.Timestamp("2024-03-01") for the default).
    """
    df = df.copy()
    if period_col is None:
        df[schema.PERIOD_MONTH] = margin.period_month(df[schema.TXN_DATE])
        period_col = schema.PERIOD_MONTH

    subset = df[df[period_col].isin([period_a, period_b])]
    excluded = subset[subset[entity_col] != entity_value]

    rows = []
    for scenario, frame in [("including_entity", subset), ("excluding_entity", excluded)]:
        totals = frame.groupby(period_col, dropna=False)[value_col].sum()
        val_a = totals.get(period_a, np.nan)
        val_b = totals.get(period_b, np.nan)
        rows.append(
            {
                "scenario": scenario,
                "period_a_value": val_a,
                "period_b_value": val_b,
                "change": val_b - val_a,
                "pct_change": (val_b - val_a) / val_a if val_a else np.nan,
            }
        )
    return pd.DataFrame(rows)


def mix_shift_decomposition(
    df: pd.DataFrame,
    period_col: str,
    period_a,
    period_b,
    group_col: str,
) -> pd.DataFrame:
    """Split the change in total margin between period_a and period_b
    into a mix/volume effect (revenue shifting between groups) and a
    rate effect (each group's own margin_pct changing), using each
    group's own margin_pct in each period rather than one blended rate.

    Returns one row per group_col value plus a TOTAL row with
    actual_margin_change (computed directly, not from the decomposition)
    alongside total_effect and their residual - for a group with ~zero
    net revenue but nonzero margin (e.g. dominated by credit notes),
    the decomposition can leave a small residual; check that group
    directly if residual is not ~0.
    """
    def _group_stats(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        g = frame.groupby(group_col, dropna=False)
        revenue = g[schema.REVENUE].sum()
        margin_sum = g[margin.MARGIN].sum()
        rate = margin_sum / revenue.replace(0, np.nan)
        return revenue, rate

    frame_a = df[df[period_col] == period_a]
    frame_b = df[df[period_col] == period_b]
    revenue_a, rate_a = _group_stats(frame_a)
    revenue_b, rate_b = _group_stats(frame_b)

    all_groups = revenue_a.index.union(revenue_b.index)
    revenue_a = revenue_a.reindex(all_groups, fill_value=0.0)
    revenue_b = revenue_b.reindex(all_groups, fill_value=0.0)
    rate_a = rate_a.reindex(all_groups)
    rate_b = rate_b.reindex(all_groups)

    # Safe rates for arithmetic only (0 where the group had no revenue
    # in that period) - a group with no revenue contributes nothing to
    # rate_effect/mix_effect regardless of an undefined rate. Displayed
    # margin_pct columns keep the real (possibly null) rate.
    rate_a_safe = rate_a.fillna(0.0)
    rate_b_safe = rate_b.fillna(0.0)

    rate_effect = revenue_a * (rate_b_safe - rate_a_safe)
    mix_effect = (revenue_b - revenue_a) * rate_b_safe
    total_effect = rate_effect + mix_effect

    result = pd.DataFrame(
        {
            group_col: all_groups,
            "revenue_a": revenue_a.values,
            "revenue_b": revenue_b.values,
            "margin_pct_a": rate_a.values,
            "margin_pct_b": rate_b.values,
            "rate_effect": rate_effect.values,
            "mix_effect": mix_effect.values,
            "total_effect": total_effect.values,
        }
    )

    actual_margin_a = frame_a[margin.MARGIN].sum()
    actual_margin_b = frame_b[margin.MARGIN].sum()
    actual_change = actual_margin_b - actual_margin_a
    total_row = {
        group_col: "TOTAL",
        "revenue_a": revenue_a.sum(),
        "revenue_b": revenue_b.sum(),
        "margin_pct_a": np.nan,
        "margin_pct_b": np.nan,
        "rate_effect": rate_effect.sum(),
        "mix_effect": mix_effect.sum(),
        "total_effect": total_effect.sum(),
        "actual_margin_change": actual_change,
        "residual": actual_change - total_effect.sum(),
    }
    return pd.concat([result, pd.DataFrame([total_row])], ignore_index=True)


def accrual_vs_cash_view(df: pd.DataFrame, value_col: str = schema.REVENUE) -> pd.DataFrame:
    """Compare a period's totals booked by txn_date (accrual - when the
    sale happened) against the same value booked by payment_date (cash -
    when it actually settled). If a period looks bad on one view but
    not the other, the apparent dip may be a payment-timing artifact
    rather than a real change in the business.

    Rows with a null payment_date (not yet settled) are excluded from
    the cash view rather than counted as zero, and their count per
    accrual period is reported separately so the exclusion is visible.
    """
    accrual_period = margin.period_month(df[schema.TXN_DATE])
    accrual_total = df.groupby(accrual_period, dropna=False)[value_col].sum().rename("accrual_total")

    has_payment = df[schema.PAYMENT_DATE].notna()
    cash_period = margin.period_month(df.loc[has_payment, schema.PAYMENT_DATE])
    cash_total = df.loc[has_payment].groupby(cash_period, dropna=False)[value_col].sum().rename("cash_total")

    unsettled_count = accrual_period[~has_payment].value_counts().rename("unsettled_txn_count")

    result = pd.concat([accrual_total, cash_total, unsettled_count], axis=1).sort_index()
    result["unsettled_txn_count"] = result["unsettled_txn_count"].fillna(0).astype(int)
    return result.rename_axis(schema.PERIOD_MONTH).reset_index()
