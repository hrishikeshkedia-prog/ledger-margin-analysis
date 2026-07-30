"""
Universal Core KPI engine (Layer 1).

Every function here works for ANY goods-selling business -- nothing in
this file knows what "fashion" means. It takes the cleaned `orders` table
(the output of `clean.clean_orders`), optionally `inventory_snapshots` and
`opex`, and computes revenue, margin, COGS/opex breakdowns, inventory
efficiency, the working-capital cycle, and SKU concentration.

Industry-specific logic (returns-as-margin-driver, CAC, cohorts,
sell-through) lives one layer up, in `fashion_config.py` + the Stage 3
module, and is built ON TOP of these functions -- it never modifies them.
Porting this dashboard to a different goods business (FMCG, auto parts)
should mean writing a new industry module, never editing this file.

No black boxes: every function's docstring states its formula and the
business question it answers.

Data contract: functions here expect the OUTPUT of `clean.clean_orders`
(a `valid_economics` boolean column may be present). Rows flagged
`valid_economics=False` -- untrustworthy quantity/price, or missing cost
data -- are excluded from every revenue and cost computation. This is
enforced once, in `add_line_economics`, so no individual KPI function has
to re-implement the exclusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# ---------------------------------------------------------------------------
# Working-capital assumptions
# ---------------------------------------------------------------------------
# DSO and DPO cannot be DERIVED from this dataset: there is no
# accounts-receivable sub-ledger (D2C orders are paid upfront via a payment
# gateway, not invoiced on credit) and no accounts-payable sub-ledger (no
# record of when supplier invoices are actually settled). Both are therefore
# modeled as explicit, stated ASSUMPTIONS rather than computed figures:
#   - dso_days: the payment gateway's settlement lag (money in hand to money
#     in the bank), typically 1-3 days for an Indian payment gateway. This is
#     NOT customer credit -- D2C customers pay at checkout.
#   - dpo_days: the brand's negotiated supplier payment terms (assumed
#     net-30, common in fashion sourcing).
# DIO, by contrast, IS derived from actual inventory data (see
# `inventory_turns`). Keeping the assumed inputs separate and labeled is the
# point: a CEO reading CCC should know which two of its three inputs are
# assumptions, not measurements.
WORKING_CAPITAL_ASSUMPTIONS = {"dso_days": 2.0, "dpo_days": 30.0}


def _filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    if "valid_economics" in df.columns:
        return df[df["valid_economics"]].copy()
    return df.copy()


def add_line_economics(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-line economics columns to a cleaned orders table. Every other
    function in this module calls this first, so it's the single place the
    return/cost model is defined.

    Columns added:
    - `month`: order_date's calendar month (YYYY-MM), the grain most KPIs
      trend on.
    - `gross_revenue_line` = unit_price * quantity, REGARDLESS of return.
      "What did we ring up at checkout."
    - `net_revenue_line` = gross_revenue_line, or 0 if the line was
      returned. "What we actually keep" -- a return means the sale is
      commercially reversed.
    - `net_cogs_line` = unit_cogs * quantity, or 0 if returned. Assumption:
      a returned item is restocked in sellable condition, so its cost is
      recovered along with reversing the revenue. (Fashion-specific nuance
      -- damaged/unsellable returns, markdown on returned stock -- is
      handled in the Stage 3 fashion module, not here.)
    - `variable_cost_line` = shipping_cost + payment_gateway_fee. These are
      NOT reversed on a return: the business already paid to ship the item
      and already paid the payment gateway's fee on the original charge.
      This is what makes returns a real margin drag, not a wash.
    - `gross_margin_line` = net_revenue_line - net_cogs_line.
    - `contribution_margin_line` = gross_margin_line - variable_cost_line.
      "What's left after the product itself and the direct cost of
      fulfilling the order, before any shared/marketing/overhead cost."

    Rows with `valid_economics == False` (see `clean.py`) are dropped here,
    once, so every downstream KPI is automatically computed on trustworthy
    data only.
    """
    df = _filter_valid(cleaned_orders_df)
    df["month"] = df[schema.ORDER_DATE].dt.to_period("M").astype(str)

    df["gross_revenue_line"] = df[schema.UNIT_PRICE] * df[schema.QUANTITY]
    df["net_revenue_line"] = np.where(df[schema.IS_RETURN], 0.0, df["gross_revenue_line"])
    df["net_cogs_line"] = np.where(df[schema.IS_RETURN], 0.0,
                                    df[schema.UNIT_COGS] * df[schema.QUANTITY])
    df["variable_cost_line"] = df[schema.SHIPPING_COST] + df[schema.PAYMENT_FEE]
    df["gross_margin_line"] = df["net_revenue_line"] - df["net_cogs_line"]
    df["contribution_margin_line"] = df["gross_margin_line"] - df["variable_cost_line"]
    return df


# ---------------------------------------------------------------------------
# Revenue and margin trends
# ---------------------------------------------------------------------------

def revenue_trend(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly gross vs. net revenue.

    Formula: gross_revenue = sum(unit_price * quantity) for all lines;
    net_revenue = the same, excluding returned lines; returned_revenue =
    gross - net.
    Business question: is the top line growing, and how much of gross
    revenue is being given back through returns?
    """
    df = add_line_economics(cleaned_orders_df)
    monthly = df.groupby("month").agg(
        gross_revenue=("gross_revenue_line", "sum"),
        net_revenue=("net_revenue_line", "sum"),
    ).reset_index()
    monthly["returned_revenue"] = monthly["gross_revenue"] - monthly["net_revenue"]
    monthly["return_rate_value"] = monthly["returned_revenue"] / monthly["gross_revenue"]
    return monthly


def gross_margin_trend(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly Gross Margin % — the classic "how much of every rupee of
    revenue is left after the cost of the product itself" number.

    Formula: Gross Margin % = (Net Revenue - Net COGS) / Net Revenue.
    Business question: is the core product economics (price vs. cost to
    make/source it) healthy and improving, independent of fulfillment,
    marketing, or overhead costs?
    """
    df = add_line_economics(cleaned_orders_df)
    monthly = df.groupby("month").agg(
        net_revenue=("net_revenue_line", "sum"),
        net_cogs=("net_cogs_line", "sum"),
    ).reset_index()
    monthly["gross_margin_abs"] = monthly["net_revenue"] - monthly["net_cogs"]
    monthly["gross_margin_pct"] = monthly["gross_margin_abs"] / monthly["net_revenue"]
    return monthly


def contribution_margin_trend(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly Contribution Margin, absolute and per order.

    Formula: Contribution Margin = Net Revenue - Net COGS - Variable
    Fulfillment Costs (shipping + payment gateway fees). Contribution
    Margin per Order = Contribution Margin / number of distinct orders that
    month.
    Business question: after making/sourcing the product AND fulfilling the
    order, how much is left to cover marketing and overhead, per rupee of
    revenue and per order placed? This is the number that answers "is each
    additional order worth taking" before marketing cost is even considered
    -- marketing-inclusive economics is a Layer 2 (fashion CAC) concern,
    layered on top of this in Stage 3/5.
    """
    df = add_line_economics(cleaned_orders_df)
    monthly = df.groupby("month").agg(
        contribution_margin_abs=("contribution_margin_line", "sum"),
        n_orders=(schema.ORDER_ID, "nunique"),
    ).reset_index()
    monthly["contribution_margin_per_order"] = (
        monthly["contribution_margin_abs"] / monthly["n_orders"]
    )
    return monthly


def returns_margin_bridge(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Makes the cost of returns visible as its own line, rather than
    leaving it implicit inside net revenue/COGS.

    Formula: `gross_contribution_margin` = (unit_price*qty - unit_cogs*qty)
    summed over EVERY line (as if nothing were ever returned) minus
    variable fulfillment cost (shipping + payment fee, which is charged
    either way). `returns_margin_impact` = the negative of
    (unit_price*qty - unit_cogs*qty) summed over RETURNED lines only --
    exactly the margin given back when a sale reverses.
    `net_contribution_margin` = gross_contribution_margin +
    returns_margin_impact, which reconciles exactly to
    `contribution_margin_trend`'s output (same number, different lens).
    Business question: precisely how many rupees of margin does this
    business give back to returns each month, isolated from product
    economics or fulfillment cost? Returns are a margin driver in any
    goods business, not a fashion-only concept -- Stage 3 goes further and
    breaks this down by category/SKU for the fashion-specific "returns are
    routine, not exceptional" story.
    """
    df = add_line_economics(cleaned_orders_df)
    df["line_margin_if_kept"] = df["gross_revenue_line"] - df[schema.UNIT_COGS] * df[schema.QUANTITY]

    monthly = df.groupby("month").agg(
        line_margin_if_kept_total=("line_margin_if_kept", "sum"),
        variable_cost_total=("variable_cost_line", "sum"),
    ).reset_index()
    monthly["gross_contribution_margin"] = (
        monthly["line_margin_if_kept_total"] - monthly["variable_cost_total"]
    )

    returned = df[df[schema.IS_RETURN]]
    returns_impact = returned.groupby("month")["line_margin_if_kept"].sum().rename("returns_margin_impact") * -1
    monthly = monthly.merge(returns_impact, on="month", how="left")
    monthly["returns_margin_impact"] = monthly["returns_margin_impact"].fillna(0.0)

    monthly["net_contribution_margin"] = (
        monthly["gross_contribution_margin"] + monthly["returns_margin_impact"]
    )
    return monthly[["month", "gross_contribution_margin", "returns_margin_impact", "net_contribution_margin"]]


# ---------------------------------------------------------------------------
# Cost breakdowns
# ---------------------------------------------------------------------------

def cogs_breakdown(cleaned_orders_df: pd.DataFrame, by: str = schema.CATEGORY) -> pd.DataFrame:
    """COGS breakdown by a grouping column (default: product category).

    Formula: sum(net_cogs_line) grouped by `by`, plus each group's % of
    total COGS.
    Business question: where is the cost of goods actually concentrated --
    which categories/SKUs are driving the biggest slice of what we spend
    making or sourcing product?
    """
    df = add_line_economics(cleaned_orders_df)
    grouped = df.groupby(by).agg(net_cogs=("net_cogs_line", "sum")).reset_index()
    grouped["pct_of_total_cogs"] = grouped["net_cogs"] / grouped["net_cogs"].sum()
    return grouped.sort_values("net_cogs", ascending=False).reset_index(drop=True)


def opex_breakdown(opex_df: pd.DataFrame) -> pd.DataFrame:
    """Operating expense breakdown by category, over the full window.

    Formula: sum(amount) grouped by expense_category, plus each category's
    % of total opex.
    Business question: where does the fixed overhead that isn't tied to any
    single order actually go (salaries vs. rent vs. tools vs. warehousing)?
    """
    grouped = opex_df.groupby(schema.OPEX_CATEGORY).agg(
        total_amount=(schema.OPEX_AMOUNT, "sum")
    ).reset_index()
    grouped["pct_of_total_opex"] = grouped["total_amount"] / grouped["total_amount"].sum()
    return grouped.sort_values("total_amount", ascending=False).reset_index(drop=True)


def opex_trend(opex_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly total operating expense (all categories summed)."""
    return opex_df.groupby(schema.OPEX_MONTH).agg(
        total_opex=(schema.OPEX_AMOUNT, "sum")
    ).reset_index()


def company_pnl_trend(cleaned_orders_df: pd.DataFrame, opex_df: pd.DataFrame) -> pd.DataFrame:
    """The monthly P&L a CEO actually reads: Contribution Margin minus Opex.

    Formula: Operating Profit = Contribution Margin (Net Revenue - Net COGS
    - Variable Fulfillment Costs) - Operating Expenses. Operating Margin %
    = Operating Profit / Net Revenue.
    Business question: after covering the product, fulfillment, AND fixed
    overhead, is the business actually profitable month to month? (Note:
    marketing spend is still excluded here -- see Stage 3/5 for
    marketing-loaded margin -- so this is "profit before marketing," a
    deliberately conservative-but-incomplete view documented as such.)
    """
    cm = contribution_margin_trend(cleaned_orders_df)
    opex = opex_trend(opex_df).rename(columns={schema.OPEX_MONTH: "month"})
    merged = cm.merge(opex, on="month", how="outer").sort_values("month").reset_index(drop=True)
    merged["operating_profit"] = merged["contribution_margin_abs"] - merged["total_opex"]
    net_rev = revenue_trend(cleaned_orders_df)[["month", "net_revenue"]]
    merged = merged.merge(net_rev, on="month", how="left")
    merged["operating_margin_pct"] = merged["operating_profit"] / merged["net_revenue"]
    return merged


# ---------------------------------------------------------------------------
# SKU-level views
# ---------------------------------------------------------------------------

def get_sku_unit_cogs(cleaned_orders_df: pd.DataFrame) -> pd.Series:
    """SKU -> unit COGS lookup, averaged over valid rows for that SKU (unit
    COGS is constant per SKU in this dataset; `.mean()` is a defensive
    choice in case a real export has small cost-entry variance). Used to
    value inventory in `inventory_turns`.
    """
    df = _filter_valid(cleaned_orders_df)
    return df.groupby(schema.SKU_ID)[schema.UNIT_COGS].mean()


def margin_by_sku(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU revenue, cost, and margin.

    Formula: for each SKU, net_revenue, net_cogs, gross_margin_abs (=
    net_revenue - net_cogs), gross_margin_pct, contribution_margin_abs,
    units_sold (net of returns), and contribution_margin_per_unit
    (contribution_margin_abs / units_sold).
    Business question: which specific products are actually making money,
    and which are selling but quietly losing money once fulfillment cost is
    counted? This table feeds the Stage 5 margin-risk alerts directly.
    """
    df = add_line_economics(cleaned_orders_df)
    df["units_sold_net"] = np.where(df[schema.IS_RETURN], 0, df[schema.QUANTITY])
    grouped = df.groupby([schema.SKU_ID, schema.CATEGORY]).agg(
        net_revenue=("net_revenue_line", "sum"),
        net_cogs=("net_cogs_line", "sum"),
        contribution_margin_abs=("contribution_margin_line", "sum"),
        units_sold=("units_sold_net", "sum"),
    ).reset_index()
    grouped["gross_margin_abs"] = grouped["net_revenue"] - grouped["net_cogs"]
    grouped["gross_margin_pct"] = grouped["gross_margin_abs"] / grouped["net_revenue"].replace(0, np.nan)
    grouped["contribution_margin_per_unit"] = (
        grouped["contribution_margin_abs"] / grouped["units_sold"].replace(0, np.nan)
    )
    return grouped.sort_values("net_revenue", ascending=False).reset_index(drop=True)


def sku_concentration(cleaned_orders_df: pd.DataFrame,
                       top_n_list: tuple = (5, 10, 20)) -> tuple[pd.DataFrame, dict]:
    """SKU revenue concentration (a Pareto / "top-N%" view).

    Formula: rank SKUs by net_revenue descending; cumulative_share = running
    sum of net_revenue / total net_revenue. Summary reports the revenue
    share held by the top N SKUs (for each N in top_n_list) and by the top
    20% of SKUs by count.
    Business question: how dependent is the business on a small number of
    hero products? High concentration means a single SKU going out of stock
    or falling out of fashion is a real revenue risk, not just a
    line-item problem.
    """
    ranked = margin_by_sku(cleaned_orders_df).sort_values("net_revenue", ascending=False).reset_index(drop=True)
    total_revenue = ranked["net_revenue"].sum()
    ranked["revenue_share"] = ranked["net_revenue"] / total_revenue
    ranked["cumulative_share"] = ranked["revenue_share"].cumsum()

    n_skus = len(ranked)
    summary = {f"top_{n}_skus_revenue_share": ranked["net_revenue"].head(n).sum() / total_revenue
               for n in top_n_list if n <= n_skus}
    top20pct_n = max(1, round(n_skus * 0.2))
    summary["top_20pct_skus_revenue_share"] = ranked["net_revenue"].head(top20pct_n).sum() / total_revenue
    summary["n_skus"] = n_skus
    return ranked, summary


# ---------------------------------------------------------------------------
# Inventory efficiency and working-capital cycle
# ---------------------------------------------------------------------------

def inventory_turns(inventory_df: pd.DataFrame, cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly inventory turns and Days of Inventory Outstanding (DIO).

    Formula: Inventory Turns (monthly) = COGS of units sold that month /
    Average Inventory Value that month, where Average Inventory Value =
    (beginning_inventory + ending_inventory) / 2, valued at each SKU's unit
    COGS. Annualized Turns = monthly turns x 12. DIO = 365 / Annualized
    Turns.
    Business question: how many times a year is inventory investment being
    sold through (high = capital working hard), and on average how many
    days of stock are sitting in the warehouse before selling (DIO -- high
    means cash trapped in slow-moving inventory)?
    """
    sku_cogs = get_sku_unit_cogs(cleaned_orders_df)
    inv = inventory_df.copy()
    inv["unit_cogs"] = inv[schema.INV_SKU_ID].map(sku_cogs)
    inv["cogs_sold_value"] = inv[schema.INV_SOLD] * inv["unit_cogs"]
    inv["avg_inventory_value"] = (
        (inv[schema.INV_BEGIN] + inv[schema.INV_END]) / 2 * inv["unit_cogs"]
    )

    monthly = inv.groupby(schema.INV_MONTH).agg(
        cogs_sold_value=("cogs_sold_value", "sum"),
        avg_inventory_value=("avg_inventory_value", "sum"),
    ).reset_index().rename(columns={schema.INV_MONTH: "month"})

    monthly["inventory_turns_monthly"] = (
        monthly["cogs_sold_value"] / monthly["avg_inventory_value"]
    )
    monthly["inventory_turns_annualized"] = monthly["inventory_turns_monthly"] * 12
    monthly["days_inventory_outstanding"] = 365 / monthly["inventory_turns_annualized"]
    return monthly


def working_capital_cycle(inventory_df: pd.DataFrame, cleaned_orders_df: pd.DataFrame,
                           assumptions: dict = WORKING_CAPITAL_ASSUMPTIONS) -> pd.DataFrame:
    """Monthly Cash Conversion Cycle (CCC) = DSO + DIO - DPO.

    Formula: CCC = Days Sales Outstanding + Days Inventory Outstanding -
    Days Payable Outstanding. DIO is derived from actual inventory data
    (see `inventory_turns`); DSO and DPO are stated ASSUMPTIONS (see
    `WORKING_CAPITAL_ASSUMPTIONS` docstring above) because this dataset has
    no receivables or payables sub-ledger.
    Business question: how many days does cash sit tied up in the cycle of
    buying/holding inventory and collecting from customers, net of how long
    the business can delay paying its own suppliers? A shorter (or
    negative) CCC means the business needs less of its own cash to fund
    growth.
    """
    turns = inventory_turns(inventory_df, cleaned_orders_df)
    out = turns[["month", "days_inventory_outstanding"]].copy()
    out["dso_days"] = assumptions["dso_days"]
    out["dpo_days"] = assumptions["dpo_days"]
    out["cash_conversion_cycle"] = (
        out["dso_days"] + out["days_inventory_outstanding"] - out["dpo_days"]
    )
    return out
