"""
D2C Fashion industry module (Layer 2).

Built ON TOP of `core.py` (Layer 1's universal engine) -- every function
here calls `core.add_line_economics` or another `core` function to get
revenue/cost columns, then adds fashion-specific logic (returns as a
margin driver, CAC, cohorts, sell-through). Nothing in `core.py` is edited
or duplicated here.

The config boundary: every function that encodes a business judgment call
(what counts as "high return", how long a customer is assumed to keep
buying, what counts as dead stock) takes `config=` as an argument,
defaulting to `fashion_config.FASHION_CONFIG`. No threshold is hardcoded
inline -- see `fashion_config.py` for what "porting to another industry
means writing a new config" actually looks like in practice, and the
notebook for a live demonstration that passing a different config changes
these functions' output with zero code changes here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core, schema
from .fashion_config import FASHION_CONFIG

WEEKS_PER_MONTH = 4.345  # 365.25 / 7 / 12 -- used to convert monthly sell rate to weekly


# ---------------------------------------------------------------------------
# Returns rate
# ---------------------------------------------------------------------------

def returns_rate_by_category(cleaned_orders_df: pd.DataFrame, config: dict = FASHION_CONFIG) -> pd.DataFrame:
    """Returns rate by category, in both units and value.

    Formula: units_return_rate = returned units / gross units sold;
    value_return_rate = returned (gross) revenue / gross revenue. Both
    computed per category over the full window.
    Business question: which categories are most exposed to returns, in
    volume terms (units coming back, a fulfillment/restocking burden) and
    value terms (revenue given back, a margin burden)? A category can be
    high in one and not the other (e.g. cheap, bulky, frequently-returned
    accessories vs. expensive, rarely-returned outerwear).
    """
    econ = core.add_line_economics(cleaned_orders_df)
    grouped = econ.groupby(schema.CATEGORY).apply(
        lambda g: pd.Series({
            "gross_units": g[schema.QUANTITY].sum(),
            "returned_units": g.loc[g[schema.IS_RETURN], schema.QUANTITY].sum(),
            "gross_revenue": g["gross_revenue_line"].sum(),
            "returned_revenue": g.loc[g[schema.IS_RETURN], "gross_revenue_line"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    grouped["units_return_rate"] = grouped["returned_units"] / grouped["gross_units"]
    grouped["value_return_rate"] = grouped["returned_revenue"] / grouped["gross_revenue"]
    grouped["high_return_flag"] = grouped["value_return_rate"] >= config["high_return_rate_threshold"]
    return grouped.sort_values("value_return_rate", ascending=False).reset_index(drop=True)


def returns_rate_by_sku(cleaned_orders_df: pd.DataFrame, config: dict = FASHION_CONFIG) -> pd.DataFrame:
    """Same as `returns_rate_by_category`, at SKU grain. See that
    docstring for the formula -- identical, just grouped by SKU instead of
    category, which is what actually lets you find the two or three
    specific products driving a category's return-rate problem."""
    econ = core.add_line_economics(cleaned_orders_df)
    grouped = econ.groupby([schema.SKU_ID, schema.CATEGORY]).apply(
        lambda g: pd.Series({
            "gross_units": g[schema.QUANTITY].sum(),
            "returned_units": g.loc[g[schema.IS_RETURN], schema.QUANTITY].sum(),
            "gross_revenue": g["gross_revenue_line"].sum(),
            "returned_revenue": g.loc[g[schema.IS_RETURN], "gross_revenue_line"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    grouped["units_return_rate"] = grouped["returned_units"] / grouped["gross_units"]
    grouped["value_return_rate"] = grouped["returned_revenue"] / grouped["gross_revenue"]
    grouped["high_return_flag"] = grouped["value_return_rate"] >= config["high_return_rate_threshold"]
    return grouped.sort_values("value_return_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CAC, ROAS, MER, LTV:CAC
# ---------------------------------------------------------------------------

def _first_orders(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Each customer's first-ever order (their acquisition order) -- the
    channel and month on THIS row is what "true CAC" attributes a new
    customer to, as opposed to every order they ever place.
    """
    econ = core.add_line_economics(cleaned_orders_df)
    orders_only = econ.drop_duplicates(subset=schema.ORDER_ID)
    return orders_only.sort_values(schema.ORDER_DATE).drop_duplicates(subset=schema.CUSTOMER_ID, keep="first")


def cac_by_channel(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly TRUE Customer Acquisition Cost by channel.

    Formula: True CAC = total spend on a channel in a month / number of
    customers whose FIRST-EVER order that month was attributed to that
    channel. This is deliberately NOT spend / all orders credited to the
    channel (a cost-per-order figure) -- a Stage 2/3 diagnostic confirmed
    that proxy understates true CAC by 20-30%, since it also credits spend
    to repeat-customer orders that channel had nothing to do with
    acquiring.
    Business question: what does it actually cost, in acquisition spend,
    to win one new customer through this channel this month?
    """
    new_cust = (_first_orders(cleaned_orders_df)
                .groupby(["month", schema.MARKETING_CHANNEL]).size()
                .reset_index(name="new_customers")
                .rename(columns={schema.MARKETING_CHANNEL: "channel"}))
    spend = marketing_df.rename(columns={schema.SPEND_CHANNEL: "channel", schema.SPEND_MONTH: "month"})
    merged = spend.merge(new_cust, on=["month", "channel"], how="left")
    merged["new_customers"] = merged["new_customers"].fillna(0)
    merged["true_cac"] = merged[schema.SPEND_AMOUNT] / merged["new_customers"].replace(0, np.nan)
    return merged


def cac_summary_by_channel(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame) -> pd.DataFrame:
    """Full-window (not monthly) true CAC by channel -- sums spend and new
    customers across all 12 months first, then divides, which is the
    correct way to combine a ratio across periods (never average of
    monthly ratios). See `cac_by_channel` for the formula."""
    monthly = cac_by_channel(cleaned_orders_df, marketing_df)
    summary = monthly.groupby("channel").agg(
        total_spend=(schema.SPEND_AMOUNT, "sum"),
        total_new_customers=("new_customers", "sum"),
    ).reset_index()
    summary["true_cac"] = summary["total_spend"] / summary["total_new_customers"].replace(0, np.nan)
    return summary


def roas_by_channel(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly Return on Ad Spend (ROAS) by channel.

    Formula: ROAS = Net Revenue from orders attributed to a channel that
    month / Spend on that channel that month.
    Business question: for every rupee spent on this channel, how many
    rupees of revenue did it drive? Unlike CAC, ROAS doesn't distinguish
    new vs. repeat customers -- it answers "was this spend worth it in
    revenue terms," where CAC answers "what did it cost to grow the
    customer base."
    """
    econ = core.add_line_economics(cleaned_orders_df)
    rev = (econ.groupby(["month", schema.MARKETING_CHANNEL])["net_revenue_line"].sum()
           .reset_index().rename(columns={schema.MARKETING_CHANNEL: "channel",
                                           "net_revenue_line": "net_revenue_attributed"}))
    spend = marketing_df.rename(columns={schema.SPEND_CHANNEL: "channel", schema.SPEND_MONTH: "month"})
    merged = spend.merge(rev, on=["month", "channel"], how="left")
    merged["net_revenue_attributed"] = merged["net_revenue_attributed"].fillna(0)
    merged["roas"] = merged["net_revenue_attributed"] / merged[schema.SPEND_AMOUNT].replace(0, np.nan)
    return merged


def mer_trend(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly Marketing Efficiency Ratio (MER) -- the blended, company-wide
    version of ROAS.

    Formula: MER = Total Net Revenue (ALL channels, including organic) /
    Total Marketing Spend (all paid channels), same month.
    Business question: blended across the whole business, how many rupees
    of revenue come in per rupee of marketing spend? MER doesn't try to
    attribute organic/repeat revenue to any channel -- it's the blunter,
    top-line efficiency number a board or investor typically asks for.
    """
    rev = core.revenue_trend(cleaned_orders_df)[["month", "net_revenue"]]
    spend = (marketing_df.groupby(schema.SPEND_MONTH)[schema.SPEND_AMOUNT].sum()
             .reset_index().rename(columns={schema.SPEND_MONTH: "month", schema.SPEND_AMOUNT: "total_spend"}))
    merged = rev.merge(spend, on="month", how="left")
    merged["mer"] = merged["net_revenue"] / merged["total_spend"].replace(0, np.nan)
    return merged


def ltv_cac_ratio(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame,
                   config: dict = FASHION_CONFIG) -> pd.DataFrame:
    """LTV:CAC ratio by channel.

    Formula: LTV = (company-wide average Contribution Margin per Order,
    from `core.contribution_margin_trend`) x
    `config['ltv_assumed_customer_lifetime_orders']`. LTV is built from
    CONTRIBUTION MARGIN, not revenue -- what a customer is actually worth
    to the business is the profit they generate, not the top line.
    `ltv_assumed_customer_lifetime_orders` is a stated ASSUMPTION (see
    `fashion_config.py`) because 12 months of data cannot reveal a
    customer's true multi-year lifetime order count. LTV:CAC ratio =
    LTV / True CAC (per channel, from `cac_summary_by_channel`).
    Business question: for each channel, is what a typical customer is
    worth over their lifetime worth several times what it cost to acquire
    them? (A common rule of thumb: healthy is >=3x.) Note LTV itself is a
    single blended, company-wide figure compared against each channel's
    own CAC -- this dataset lacks the per-customer, per-channel repeat-
    purchase granularity to fit channel-specific retention curves, which
    is the standard simplification early-stage D2C teams use for this
    exact metric.
    """
    cm = core.contribution_margin_trend(cleaned_orders_df)
    avg_contribution_margin_per_order = cm["contribution_margin_abs"].sum() / cm["n_orders"].sum()
    ltv = avg_contribution_margin_per_order * config["ltv_assumed_customer_lifetime_orders"]

    summary = cac_summary_by_channel(cleaned_orders_df, marketing_df)
    summary["avg_contribution_margin_per_order"] = avg_contribution_margin_per_order
    summary["ltv_assumed_customer_lifetime_orders"] = config["ltv_assumed_customer_lifetime_orders"]
    summary["ltv"] = ltv
    summary["ltv_cac_ratio"] = ltv / summary["true_cac"]
    return summary


# ---------------------------------------------------------------------------
# AOV, repeat rate, cohorts
# ---------------------------------------------------------------------------

def aov_trend(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly Average Order Value (AOV).

    Formula: AOV = Gross Revenue (at checkout, before any later return) /
    number of distinct orders that month. Gross, not net, is used
    deliberately -- AOV is a basket-size-at-checkout metric; whether an
    item is later returned isn't known at the moment of purchase.
    Business question: is the average basket size at checkout growing,
    shrinking, or holding steady month to month?
    """
    econ = core.add_line_economics(cleaned_orders_df)
    monthly = econ.groupby("month").agg(
        gross_revenue=("gross_revenue_line", "sum"),
        n_orders=(schema.ORDER_ID, "nunique"),
    ).reset_index()
    monthly["aov"] = monthly["gross_revenue"] / monthly["n_orders"]
    return monthly


def repeat_purchase_rate(cleaned_orders_df: pd.DataFrame, config: dict = FASHION_CONFIG) -> dict:
    """Repeat-purchase rate over the full 12-month window.

    Formula: Repeat Purchase Rate = (customers with >=
    `config['repeat_purchase_min_orders']` distinct orders in the window)
    / (all customers with >= 1 order in the window).
    Business question: of everyone who bought at least once, what share
    came back for a second order? This is the cleanest available signal of
    whether the PRODUCT itself is working, independent of marketing --
    customers vote with a second order.
    """
    orders_only = cleaned_orders_df.drop_duplicates(subset=schema.ORDER_ID)
    order_counts = orders_only.groupby(schema.CUSTOMER_ID).size()
    total_customers = len(order_counts)
    repeat_customers = int((order_counts >= config["repeat_purchase_min_orders"]).sum())
    return {
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_purchase_rate": repeat_customers / total_customers,
    }


def cohort_retention_table(cleaned_orders_df: pd.DataFrame, config: dict = FASHION_CONFIG) -> pd.DataFrame:
    """Cohort retention: for customers acquired in a given month (their
    cohort), what % are still placing orders N months later?

    Formula: cohort_month = the calendar month of a customer's FIRST-ever
    order. For each cohort and each month-offset N (0, 1, 2, ...): active
    customers = distinct customers from that cohort who placed >=1 order in
    (cohort_month + N); retention_pct = active customers / cohort size.
    Offset 0 is always 100% by construction (everyone in a cohort is
    active in their own acquisition month). Reported out to
    `config['cohort_window_months']`.
    Business question: does retention decay quickly or slowly after
    acquisition, and does it differ by WHEN a customer was acquired (e.g.
    do customers acquired during a big discount-driven sale month churn
    faster than customers acquired at full price)?
    """
    econ = core.add_line_economics(cleaned_orders_df)
    orders_only = econ.drop_duplicates(subset=schema.ORDER_ID)[[schema.CUSTOMER_ID, schema.ORDER_DATE, "month"]]

    first = orders_only.sort_values(schema.ORDER_DATE).drop_duplicates(subset=schema.CUSTOMER_ID, keep="first")
    cohort_month_by_customer = first.set_index(schema.CUSTOMER_ID)["month"]

    customer_months = orders_only.drop_duplicates(subset=[schema.CUSTOMER_ID, "month"]).copy()
    customer_months["cohort_month"] = customer_months[schema.CUSTOMER_ID].map(cohort_month_by_customer)

    activity_ordinal = pd.PeriodIndex(customer_months["month"], freq="M").asi8
    cohort_ordinal = pd.PeriodIndex(customer_months["cohort_month"], freq="M").asi8
    customer_months["offset"] = activity_ordinal - cohort_ordinal

    cohort_sizes = first.groupby("month")[schema.CUSTOMER_ID].nunique()

    retention = (customer_months.groupby(["cohort_month", "offset"])[schema.CUSTOMER_ID].nunique()
                 .reset_index(name="active_customers"))
    retention["cohort_size"] = retention["cohort_month"].map(cohort_sizes)
    retention["retention_pct"] = retention["active_customers"] / retention["cohort_size"]

    max_offset = config["cohort_window_months"]
    retention = retention[(retention["offset"] >= 0) & (retention["offset"] <= max_offset)]
    table = retention.pivot(index="cohort_month", columns="offset", values="retention_pct")
    return table.sort_index()


# ---------------------------------------------------------------------------
# Sell-through, weeks of cover, markdown, dead stock
# ---------------------------------------------------------------------------

def sell_through_rate_monthly(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly sell-through rate, per SKU.

    Formula: Sell-Through Rate = units_sold that month / units_available
    that month, where units_available = beginning_inventory +
    units_received (everything that COULD have been sold that month).
    Business question: of everything we had ready to sell this month, what
    fraction actually sold? The classic fashion-retail efficiency metric --
    distinct from raw units sold, since a SKU can sell a lot in absolute
    terms while still selling through a low % of a much larger stock pile.
    """
    df = inventory_df.copy()
    df["units_available"] = df[schema.INV_BEGIN] + df[schema.INV_RECEIVED]
    df["sell_through_rate"] = df[schema.INV_SOLD] / df["units_available"].replace(0, np.nan)
    return df[[schema.INV_SKU_ID, schema.INV_MONTH, "units_available", schema.INV_SOLD, "sell_through_rate"]]


def cumulative_sell_through(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Lifetime (since-launch) sell-through per SKU, plus how long it's
    been available.

    Formula: cumulative_units_available = the SKU's very first month's
    beginning_inventory (its initial stocking) + units_received summed
    over every month it's appeared in the inventory table.
    cumulative_units_sold = units_sold summed the same way.
    cumulative_sell_through = cumulative_units_sold /
    cumulative_units_available. months_available = number of months the
    SKU has appeared in the inventory table at all (a proxy for months
    since launch, since a SKU's rows only start from its launch month).
    Business question: over its whole life so far, what fraction of
    everything this SKU ever had in stock has it actually sold? Feeds
    directly into the dead-stock definition below.
    """
    df = inventory_df.sort_values([schema.INV_SKU_ID, schema.INV_MONTH])
    grouped = df.groupby(schema.INV_SKU_ID)
    initial_stock = grouped[schema.INV_BEGIN].first()
    total_received = grouped[schema.INV_RECEIVED].sum()
    total_sold = grouped[schema.INV_SOLD].sum()
    months_available = grouped.size()

    out = pd.DataFrame({
        "initial_stock": initial_stock,
        "total_received": total_received,
        "total_sold": total_sold,
        "months_available": months_available,
    })
    out["cumulative_units_available"] = out["initial_stock"] + out["total_received"]
    out["cumulative_sell_through"] = out["total_sold"] / out["cumulative_units_available"].replace(0, np.nan)
    return out.reset_index()


def weeks_of_cover(inventory_df: pd.DataFrame, config: dict = FASHION_CONFIG) -> pd.DataFrame:
    """Monthly weeks-of-cover per SKU, with a stockout/healthy/overstock
    classification.

    Formula: weekly_sell_rate = units_sold that month / weeks-per-month
    (~4.345). Weeks of Cover = ending_inventory / weekly_sell_rate. If a
    SKU sold zero units that month but still has stock on hand, weeks of
    cover is set to infinity (there's no recent sell rate to divide by --
    this itself is a signal, not a missing value). If it has neither stock
    nor sales, weeks of cover is undefined (NaN).
    Classification uses `config['weeks_of_cover_healthy_range']`: below
    the low end = stockout risk, above the high end (or infinite) =
    overstock risk, in between = healthy.
    Business question: at the current sell rate, how many weeks until this
    SKU runs out of stock -- or, if that number is very high, how many
    weeks of dead capital is sitting in the warehouse for it?
    """
    df = inventory_df.copy()
    weekly_rate = df[schema.INV_SOLD] / WEEKS_PER_MONTH
    df["weeks_of_cover"] = np.where(
        weekly_rate > 0, df[schema.INV_END] / weekly_rate.replace(0, np.nan),
        np.where(df[schema.INV_END] > 0, np.inf, np.nan),
    )
    low, high = config["weeks_of_cover_healthy_range"]
    df["cover_status"] = np.where(
        df["weeks_of_cover"].isna(), "No stock, no recent sales",
        np.where(df["weeks_of_cover"] < low, "Stockout risk",
                 np.where(df["weeks_of_cover"] > high, "Overstock risk", "Healthy")),
    )
    return df[[schema.INV_SKU_ID, schema.INV_MONTH, schema.INV_END, schema.INV_SOLD, "weeks_of_cover", "cover_status"]]


def markdown_pct(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly markdown exposure: what share of revenue was sold at a
    discount, and how deep.

    Formula: a line is "discounted" if unit_price < list_price.
    pct_revenue_discounted = gross revenue from discounted lines / total
    gross revenue, per month. avg_discount_depth = mean((list_price -
    unit_price) / list_price), among discounted lines only.
    Business question: how much of what we sell is full-price vs.
    marked-down, and when discounting happens, how deep does it typically
    go? Heavy, deep markdown activity is a margin drag that a simple
    revenue trend hides completely.
    """
    econ = core.add_line_economics(cleaned_orders_df)
    econ["is_discounted"] = econ[schema.UNIT_PRICE] < econ[schema.LIST_PRICE]
    econ["discount_depth"] = np.where(
        econ["is_discounted"],
        (econ[schema.LIST_PRICE] - econ[schema.UNIT_PRICE]) / econ[schema.LIST_PRICE],
        np.nan,
    )

    monthly = econ.groupby("month").apply(
        lambda g: pd.Series({
            "gross_revenue": g["gross_revenue_line"].sum(),
            "discounted_revenue": g.loc[g["is_discounted"], "gross_revenue_line"].sum(),
            "avg_discount_depth": g.loc[g["is_discounted"], "discount_depth"].mean(),
        }),
        include_groups=False,
    ).reset_index()
    monthly["pct_revenue_discounted"] = monthly["discounted_revenue"] / monthly["gross_revenue"]
    return monthly


def dead_stock_pct(inventory_df: pd.DataFrame, config: dict = FASHION_CONFIG) -> dict:
    """What share of the catalog is dead stock: available long enough to
    have proven itself, but barely selling.

    Formula: a SKU is "dead stock" if `months_available` (see
    `cumulative_sell_through`) >= `config['dead_stock_min_months_available']`
    (so a brand-new arrival isn't unfairly flagged) AND
    `cumulative_sell_through` <= `config['dead_stock_max_cumulative_sell_through']`.
    dead_stock_pct is reported against ELIGIBLE SKUs only (those old enough
    to judge), not the whole catalog, since including too-young SKUs in
    the denominator would understate the real problem among SKUs old
    enough to have proven themselves.
    Business question: of the products that have had a fair chance to
    sell, what share are essentially dead capital sitting in the
    warehouse?
    """
    cum = cumulative_sell_through(inventory_df)
    eligible = cum[cum["months_available"] >= config["dead_stock_min_months_available"]].copy()
    eligible["is_dead_stock"] = eligible["cumulative_sell_through"] <= config["dead_stock_max_cumulative_sell_through"]

    n_eligible = len(eligible)
    n_dead = int(eligible["is_dead_stock"].sum())
    return {
        "n_eligible_skus": n_eligible,
        "n_dead_stock_skus": n_dead,
        "dead_stock_pct_of_eligible": n_dead / n_eligible if n_eligible else np.nan,
        "dead_stock_sku_ids": eligible.loc[eligible["is_dead_stock"], schema.INV_SKU_ID].tolist(),
    }
