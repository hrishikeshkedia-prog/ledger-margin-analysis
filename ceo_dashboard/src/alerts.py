"""
Margin-risk alert engine (Stage 5).

Deliberately NOT a trained classifier. The 28 loss-making SKUs and the one
loss-making channel in this synthetic dataset were seeded on purpose (see
`data_generator.py`'s `PROBLEM_SKU_*` constants and the Influencer channel
cost calibration) -- a model trained to "predict" loss-making status from
this data would just re-learn the exact rule that generated the seed. That
is circular, not predictive, and does not generalize to a real business
where nobody has hand-planted the answer key. The genuine predictive
model in this project is Stage 6's demand/inventory forecast, which
predicts something that was NOT seeded (future units).

This module is transparent rules + arithmetic decomposition instead:
1. FLAG: read loss-making status straight off `fashion.channel_margin_after_cac`
   / `fashion.sku_margin_after_cac` -- not reimplemented here.
2. EXPLAIN: decompose the shortfall into four named, dollar-quantified
   drivers (returns, CAC, thin gross margin, fixed-cost burden) and name
   whichever is largest as the primary cause.
3. EARLY-WARNING: a simple, stated trend rule (consecutive-month decline,
   or linear extrapolation crossing zero) over each SKU's own monthly
   contribution-margin history -- not a classifier, a trend fit on one
   SKU's own past, projected forward.
4. OUTPUT: one ranked table combining both, each row naming its cause and
   a recommended action.

Every threshold below is named, given a default, and justified in one
sentence -- no unexplained magic numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import core, fashion, schema
from .fashion_config import FASHION_CONFIG

ALERTS_CONFIG = {
    # A SKU's value-return-rate at or above this is treated as a candidate
    # "high returns" driver -- reuses the same bar fashion.py already uses
    # to label a category "high-return", for one consistent definition of
    # "high" across the whole project.
    "high_return_rate_threshold": FASHION_CONFIG["high_return_rate_threshold"],

    # A SKU needs at least this many months of sales history before its
    # trend is evaluated at all -- fewer points make "declining" or a
    # fitted slope statistically meaningless noise, not a real signal.
    "min_months_data": 4,

    # Flag a currently-positive SKU if its monthly contribution margin
    # strictly decreased in each of this many most-recent consecutive
    # months it sold in.
    "decline_streak_months": 3,

    # For the linear-projection check: fit a straight line to the last N
    # months of contribution margin.
    "projection_lookback_months": 4,

    # ...and flag if that line's slope is negative and crosses zero within
    # this many months beyond the last observed one.
    "projection_horizon_months": 2,

    # A SKU selling fewer than this many units over the year is "low
    # volume" -- too few orders for its per-order fixed costs (shipping,
    # payment fee) to be absorbed the way a higher-volume SKU absorbs them.
    "low_volume_units_threshold": 20,
}

_CAUSE_LABELS = {
    "returns": "High returns rate",
    "cac": "High customer acquisition cost",
    "thin_margin": "Thin gross margin vs. category peers",
    "fixed_cost": "High fixed-cost burden (low price/volume)",
    "channel_cac": "CAC well above other channels",
    "channel_payback": "Contribution margin per order below company average",
}

_RECOMMENDED_ACTIONS = {
    "returns": "Investigate fit/quality/sizing; fix the return driver or discontinue the SKU/size.",
    "cac": "Reduce marketing spend allocated to this SKU's channel mix, or raise price to cover acquisition cost.",
    "thin_margin": "Renegotiate COGS with the supplier or raise price; this SKU's product economics trail its category.",
    "fixed_cost": "Bundle with other items, raise the free-shipping threshold, or discontinue -- fixed costs dominate at this price/volume.",
    "channel_cac": "Cut or renegotiate spend on this channel; reallocate budget to lower-CAC channels.",
    "channel_payback": "Review targeting/creative on this channel -- orders it drives earn less contribution margin than average.",
}


# ---------------------------------------------------------------------------
# 1 & 2. Flag + explain: SKU-level decomposition
# ---------------------------------------------------------------------------

def decompose_sku_causes(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame) -> pd.DataFrame:
    """For every SKU with at least one order line, breaks contribution
    margin after CAC into four named, dollar-quantified drivers and names
    the largest as the primary cause.

    Drivers (all in rupees, all floored at 0 -- a driver that isn't a
    problem contributes nothing, never a negative "credit"):

    - `returns_drag`: margin given back specifically to returns = sum, over
      the SKU's returned lines, of (gross_revenue_line - unit_cogs x line
      quantity). Same formula `core.returns_margin_bridge` uses company/
      month-wide, applied at SKU grain here.
    - `cac_drag`: total CAC allocated to the SKU = this SKU's
      `contribution_margin` minus its `cm_after_cac`
      (from `fashion.sku_margin_after_cac`'s own two columns -- CAC isn't
      recomputed, just isolated by subtracting two already-computed values).
    - `thin_margin_drag`: (this SKU's category's revenue-weighted average
      gross-margin % minus this SKU's own gross-margin %) x this SKU's net
      revenue -- how many rupees short of its category peers' typical
      product economics, at this SKU's actual revenue level.
    - `fixed_cost_drag`: (this SKU's variable-cost-as-%-of-gross-revenue
      minus the company-wide average) x this SKU's gross revenue --
      shipping/payment-fee cost is roughly flat in rupees per order
      regardless of item price, so a cheap or low-volume SKU can carry a
      disproportionate fixed-cost burden relative to its revenue.

    `primary_cause` = whichever driver has the largest rupee magnitude
    (ties broken returns > cac > thin_margin > fixed_cost, since returns
    and CAC are usually the higher-leverage levers to act on first).

    Business question: for a SKU that's losing money, is it because
    customers are sending it back, because it costs too much to acquire
    the customers who buy it, because its underlying product economics
    trail similar products, or because its price point is too low to
    carry fixed per-order costs? Each cause needs a different fix.
    """
    econ = core.add_line_economics(cleaned_orders_df)
    sku_cac = fashion.sku_margin_after_cac(cleaned_orders_df, marketing_df)
    returns_by_sku = fashion.returns_rate_by_sku(cleaned_orders_df)[[schema.SKU_ID, "value_return_rate"]]
    margin_by_sku = core.margin_by_sku(cleaned_orders_df)

    # returns_drag: margin given back on returned lines, per SKU. A
    # returned line's "margin if kept" is positive for a profitable line;
    # the drag is exactly that amount given back, so it's just the sum.
    econ["line_margin_if_kept"] = econ["gross_revenue_line"] - econ[schema.UNIT_COGS] * econ[schema.QUANTITY]
    returns_drag = (econ.loc[econ[schema.IS_RETURN]]
                     .groupby(schema.SKU_ID)["line_margin_if_kept"].sum()
                     .rename("returns_drag").clip(lower=0))

    # category benchmark gross margin %, revenue-weighted
    category_benchmark = (margin_by_sku.groupby(schema.CATEGORY)
                           .apply(lambda g: g["gross_margin_abs"].sum() / g["net_revenue"].sum(), include_groups=False)
                           .rename("category_avg_gross_margin_pct"))

    # company-wide average variable-cost-as-%-of-gross-revenue
    company_variable_cost_pct = econ["variable_cost_line"].sum() / econ["gross_revenue_line"].sum()
    sku_variable_cost = (econ.groupby(schema.SKU_ID)
                          .agg(variable_cost=("variable_cost_line", "sum"),
                               gross_revenue=("gross_revenue_line", "sum")))
    sku_variable_cost["variable_cost_pct"] = sku_variable_cost["variable_cost"] / sku_variable_cost["gross_revenue"]

    out = sku_cac.merge(returns_by_sku, on=schema.SKU_ID, how="left")
    out = out.merge(margin_by_sku[[schema.SKU_ID, "net_revenue", "gross_margin_pct", "units_sold"]],
                     on=schema.SKU_ID, how="left")
    out = out.merge(returns_drag, on=schema.SKU_ID, how="left")
    out["returns_drag"] = out["returns_drag"].fillna(0)
    out["cac_drag"] = (out["contribution_margin"] - out["cm_after_cac"]).clip(lower=0)

    out["category_avg_gross_margin_pct"] = out[schema.CATEGORY].map(category_benchmark)
    out["thin_margin_drag"] = ((out["category_avg_gross_margin_pct"] - out["gross_margin_pct"])
                                .clip(lower=0) * out["net_revenue"])
    # A SKU with zero net revenue (every unit sold was returned) has an
    # undefined gross-margin % (core.py leaves it NaN rather than a
    # misleading 0) -- there's no revenue base for a "thin margin" driver
    # to mean anything, so it's 0, not NaN. returns_drag already captures
    # that SKU's real problem.
    out["thin_margin_drag"] = out["thin_margin_drag"].fillna(0)

    out = out.merge(sku_variable_cost[["variable_cost_pct"]], on=schema.SKU_ID, how="left")
    out["fixed_cost_drag"] = ((out["variable_cost_pct"] - company_variable_cost_pct)
                               .clip(lower=0) * sku_variable_cost.reindex(out[schema.SKU_ID])["gross_revenue"].values)
    out["fixed_cost_drag"] = out["fixed_cost_drag"].fillna(0)

    # short keys here match _CAUSE_LABELS / _RECOMMENDED_ACTIONS -- the
    # driver COLUMNS keep their "_drag" suffix (they hold rupee values),
    # this map is what turns "which column won" into "which named cause".
    driver_cols = {"returns_drag": "returns", "cac_drag": "cac",
                   "thin_margin_drag": "thin_margin", "fixed_cost_drag": "fixed_cost"}
    priority = {"returns": 0, "cac": 1, "thin_margin": 2, "fixed_cost": 3}

    def _pick_primary(row):
        vals = {key: row[col] for col, key in driver_cols.items()}
        max_val = max(vals.values())
        if max_val <= 0:
            return "none"
        # ties (within 1 rupee) broken by stated priority order
        tied = [key for key, v in vals.items() if v >= max_val - 1]
        tied.sort(key=lambda key: priority[key])
        return tied[0]

    out["primary_driver"] = out.apply(_pick_primary, axis=1)
    return out


# ---------------------------------------------------------------------------
# 1 & 2. Flag + explain: channel-level decomposition
# ---------------------------------------------------------------------------

def decompose_channel_causes(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame) -> pd.DataFrame:
    """For every paid channel, breaks contribution margin after CAC into
    two named, rupee-quantified drivers and names the larger as the
    primary cause.

    - `channel_cac_excess`: (this channel's True CAC minus the median True
      CAC of the OTHER paid channels) x this channel's new customers,
      floored at 0 -- how many extra rupees this channel spent acquiring
      customers versus what its peer channels typically pay.
    - `channel_payback_shortfall`: (company-wide average Contribution
      Margin per Order, from `fashion.ltv_cac_ratio` -- minus this
      channel's own contribution margin per order) x this channel's
      orders, floored at 0 -- how many rupees short this channel's orders
      run of average product/fulfillment economics.

    Business question: is a channel unprofitable because it costs too much
    to acquire a customer through it (fix: cut spend, renegotiate), or
    because the orders it drives are just less profitable once acquired
    (fix: review targeting/creative, not just the price paid for clicks)?
    """
    channel_pnl = fashion.channel_margin_after_cac(cleaned_orders_df, marketing_df)
    cac_summary = fashion.cac_summary_by_channel(cleaned_orders_df, marketing_df)
    ltv_cac = fashion.ltv_cac_ratio(cleaned_orders_df, marketing_df)
    company_avg_cm_per_order = ltv_cac["avg_contribution_margin_per_order"].iloc[0]

    paid = cac_summary[cac_summary["total_spend"] > 0]
    out = channel_pnl.merge(paid[["channel", "true_cac", "total_new_customers"]], on="channel", how="left")
    out["contribution_margin_per_order"] = out["contribution_margin"] / out["orders"]

    def _median_other_cac(channel):
        others = paid.loc[paid["channel"] != channel, "true_cac"]
        return others.median() if len(others) else np.nan

    out["benchmark_cac"] = out["channel"].apply(_median_other_cac)
    out["channel_cac_excess"] = ((out["true_cac"] - out["benchmark_cac"]).clip(lower=0)
                                  * out["total_new_customers"].fillna(0))
    out["channel_payback_shortfall"] = ((company_avg_cm_per_order - out["contribution_margin_per_order"])
                                         .clip(lower=0) * out["orders"])

    def _pick_primary(row):
        a, b = row["channel_cac_excess"], row["channel_payback_shortfall"]
        if max(a, b) <= 0 or pd.isna(a):
            return "none"
        return "channel_cac" if a >= b else "channel_payback"

    out["primary_driver"] = out.apply(_pick_primary, axis=1)
    return out


# ---------------------------------------------------------------------------
# 3. Early warning: currently positive, trending toward negative
# ---------------------------------------------------------------------------

def sku_monthly_contribution_margin(cleaned_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly contribution margin per SKU (pre-CAC: revenue - COGS -
    shipping/payment fee, i.e. product + fulfillment economics only).
    Used only for trend detection -- CAC is allocated at the channel-month
    level company-wide, not tracked as a stable per-SKU monthly rate over
    a short window, so the early-warning signal deliberately looks at
    whether a SKU's OWN product economics are deteriorating, before
    layering month-to-month CAC allocation noise on top.
    """
    econ = core.add_line_economics(cleaned_orders_df)
    return (econ.groupby([schema.SKU_ID, "month"])["contribution_margin_line"]
            .sum().reset_index().sort_values([schema.SKU_ID, "month"]))


def detect_early_warning_skus(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame,
                               config: dict = ALERTS_CONFIG) -> pd.DataFrame:
    """Flags SKUs that are NOT currently loss-making after CAC, but whose
    own monthly (pre-CAC) contribution margin trend says trouble is coming.

    Only evaluated for SKUs with at least `config['min_months_data']`
    months of sales history (fewer points make any trend statistically
    meaningless). A SKU is flagged if EITHER simple, stated rule fires:

    1. Decline streak: monthly contribution margin strictly decreased in
       each of the last `config['decline_streak_months']` consecutive
       months it sold in.
    2. Projected zero-cross: an ordinary least-squares line (`numpy.polyfit`,
       degree 1 -- a trend FIT on this one SKU's own history, not a
       classifier trained across SKUs) through the last
       `config['projection_lookback_months']` months has a negative slope,
       and extrapolating that line forward crosses zero within
       `config['projection_horizon_months']` months of the last observed
       month.

    Business question: which currently-healthy SKUs are heading toward
    becoming the next loss-maker, early enough to act before they cross
    zero rather than after?
    """
    monthly = sku_monthly_contribution_margin(cleaned_orders_df)
    sku_cac = fashion.sku_margin_after_cac(cleaned_orders_df, marketing_df)
    currently_healthy = set(sku_cac.loc[~sku_cac["is_loss_making"], schema.SKU_ID])

    rows = []
    for sku_id, g in monthly.groupby(schema.SKU_ID):
        if sku_id not in currently_healthy:
            continue
        g = g.sort_values("month")
        values = g["contribution_margin_line"].to_numpy()
        if len(values) < config["min_months_data"]:
            continue

        streak_n = config["decline_streak_months"]
        recent = values[-(streak_n + 1):]
        declining_streak = len(recent) == streak_n + 1 and all(
            recent[i] > recent[i + 1] for i in range(len(recent) - 1)
        )

        lookback = min(config["projection_lookback_months"], len(values))
        recent_vals = values[-lookback:]
        x = np.arange(lookback)
        slope, intercept = np.polyfit(x, recent_vals, 1)
        projected_zero_cross = False
        if slope < 0:
            months_to_zero = -intercept / slope - (lookback - 1)
            projected_zero_cross = 0 < months_to_zero <= config["projection_horizon_months"]

        if declining_streak or projected_zero_cross:
            rows.append({
                schema.SKU_ID: sku_id,
                "months_of_history": len(values),
                "latest_monthly_cm": values[-1],
                "trend_slope_per_month": slope,
                "declining_streak": declining_streak,
                "projected_zero_cross": projected_zero_cross,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Combined, ranked alerts table
# ---------------------------------------------------------------------------

def _format_sku_reason(row) -> str:
    label = _CAUSE_LABELS.get(row["primary_driver"], "Multiple factors")
    parts = [
        f"returns {row['value_return_rate']:.0%}",
        f"CAC ₹{row['cac_drag']:,.0f} allocated",
        f"gross margin {row['gross_margin_pct']:.0%}",
        f"{row['units_sold']:.0f} units sold",
    ]
    return f"{label} — " + ", ".join(parts)


def _format_channel_reason(row) -> str:
    label = _CAUSE_LABELS.get(row["primary_driver"], "Multiple factors")
    parts = [f"True CAC ₹{row['true_cac']:,.0f}", f"contribution margin/order ₹{row['contribution_margin_per_order']:,.0f}"]
    return f"{label} — " + ", ".join(parts)


def build_alerts_table(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame,
                        config: dict = ALERTS_CONFIG) -> pd.DataFrame:
    """The single ranked output of this module: every currently loss-making
    SKU/channel (with its decomposed primary cause) plus every SKU trending
    toward loss-making, one row each, ranked loss-making-first (most
    negative margin first) then at-risk (by how close/fast it's declining).

    Columns: entity_type, entity_id, group (category or "Channel"),
    current_margin, severity ("Loss-Making" / "At-Risk"), primary_cause,
    reason (plain-language detail), recommended_action.
    """
    sku_causes = decompose_sku_causes(cleaned_orders_df, marketing_df)
    channel_causes = decompose_channel_causes(cleaned_orders_df, marketing_df)
    early_warning = detect_early_warning_skus(cleaned_orders_df, marketing_df, config)

    loss_sku = sku_causes[sku_causes["is_loss_making"]].copy()
    loss_sku["entity_type"] = "SKU"
    loss_sku["entity_id"] = loss_sku[schema.SKU_ID]
    loss_sku["group"] = loss_sku[schema.CATEGORY]
    loss_sku["current_margin"] = loss_sku["cm_after_cac"]
    loss_sku["severity"] = "Loss-Making"
    loss_sku["primary_cause"] = loss_sku["primary_driver"].map(_CAUSE_LABELS).fillna("Multiple factors")
    loss_sku["reason"] = loss_sku.apply(_format_sku_reason, axis=1)
    loss_sku["recommended_action"] = loss_sku["primary_driver"].map(_RECOMMENDED_ACTIONS).fillna(
        "Review this SKU's full economics -- no single dominant driver.")

    loss_channel = channel_causes[channel_causes["is_loss_making"]].copy()
    loss_channel["entity_type"] = "Channel"
    loss_channel["entity_id"] = loss_channel["channel"]
    loss_channel["group"] = "Channel"
    loss_channel["current_margin"] = loss_channel["cm_after_cac"]
    loss_channel["severity"] = "Loss-Making"
    loss_channel["primary_cause"] = loss_channel["primary_driver"].map(_CAUSE_LABELS).fillna("Multiple factors")
    loss_channel["reason"] = loss_channel.apply(_format_channel_reason, axis=1)
    loss_channel["recommended_action"] = loss_channel["primary_driver"].map(_RECOMMENDED_ACTIONS).fillna(
        "Review this channel's full economics -- no single dominant driver.")

    at_risk = early_warning.copy()
    if len(at_risk):
        sku_cac_lookup = fashion.sku_margin_after_cac(cleaned_orders_df, marketing_df).set_index(schema.SKU_ID)
        at_risk["entity_type"] = "SKU"
        at_risk["entity_id"] = at_risk[schema.SKU_ID]
        at_risk["group"] = sku_cac_lookup.reindex(at_risk[schema.SKU_ID])[schema.CATEGORY].values
        at_risk["current_margin"] = sku_cac_lookup.reindex(at_risk[schema.SKU_ID])["cm_after_cac"].values
        at_risk["severity"] = "At-Risk"
        at_risk["primary_cause"] = "Declining margin trend"
        reasons = []
        for _, r in at_risk.iterrows():
            bits = []
            if r["declining_streak"]:
                bits.append(f"contribution margin fell {config['decline_streak_months']} months in a row")
            if r["projected_zero_cross"]:
                bits.append(f"trend projects crossing zero within {config['projection_horizon_months']} months")
            reasons.append("; ".join(bits) + f" (latest monthly CM ₹{r['latest_monthly_cm']:,.0f})")
        at_risk["reason"] = reasons
        at_risk["recommended_action"] = "Monitor closely and investigate the cause now, before this SKU turns loss-making."

    cols = ["entity_type", "entity_id", "group", "current_margin", "severity",
            "primary_cause", "reason", "recommended_action"]
    loss_table = pd.concat([loss_sku[cols], loss_channel[cols]], ignore_index=True).sort_values("current_margin")
    at_risk_table = at_risk[cols].sort_values("current_margin") if len(at_risk) else pd.DataFrame(columns=cols)

    return pd.concat([loss_table, at_risk_table], ignore_index=True)
