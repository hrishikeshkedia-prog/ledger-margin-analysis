"""
Synthetic D2C fashion dataset generator.

*** ALL DATA PRODUCED BY THIS MODULE IS SYNTHETIC. *** It is generated from
seeded random distributions chosen to look like a plausible small D2C
fashion brand -- it is not real sales data and must not be interpreted as
such. Every notebook output built on top of it should repeat this label.

What gets generated (three tables, matching src/schema.py)
------------------------------------------------------------
1. orders   - one row per order LINE ITEM over a 12-month window: SKU,
              category, size, quantity, price, COGS, shipping cost, payment
              fee, marketing-channel attribution, return flag/date, and the
              customer who placed it.
2. marketing_spend - one row per (month, channel): total spend on that
              channel that month. Kept separate from orders because spend is
              a channel-level input (an ad platform invoice), not a
              property of one order.
3. inventory_snapshots - one row per (month, SKU): opening stock, units
              received (restocked), units sold, closing stock. Needed for
              turns / days-of-inventory / sell-through, which can't be
              computed from orders alone.

Design choices worth knowing (so the numbers aren't a black box)
------------------------------------------------------------------
- 12 months, Jul 2025 - Jun 2026, with an order-volume ramp (the brand is
  growing) and two seasonal demand spikes (an autumn sale, an end-of-season
  sale) modeled as multipliers on the daily order rate.
- Six product categories, each with its own price band, COGS-as-%-of-price
  band, and RETURN RATE. Fit-sensitive categories (Dresses, Tops, Bottoms)
  are seeded with high return rates (28-38%), which is realistic for fashion
  and deliberately large enough that the margin impact of returns is visible
  in later stages -- this is the "roughly 20-40%" range called for in the
  brief.
- Customers arrive over time (new-customer pool grows monthly) and a
  fraction of existing customers re-order in later months, weighted by
  RECENCY of their last order (`RETAIN_DECAY_RATE`) rather than drawn
  uniformly from the whole pool -- this is what makes cohort retention
  actually decay by cohort age, rather than the flat, noisy line a
  uniform draw produces (caught by a Stage 3 diagnostic).
- Discounting is heavier in the two sale months, which is what will make
  markdown % visible in Stage 3.
- Marketing spend per channel is generated to roughly track the number of
  orders that channel is credited with (plus noise and a channel-specific
  efficiency factor), which is what makes CAC/ROAS/MER differ meaningfully
  by channel in Stage 3 rather than being flat by construction.
- Inventory is generated from realized sales (so units_sold in the
  inventory table reconciles with orders) but purchasing decisions are
  deliberately imperfect for a subset of SKUs -- a few are over-bought
  (feeding dead-stock/markdown analysis) and a few are under-bought (feeding
  stockout warnings) in later stages. When realized demand bursts past what
  the restock heuristic forecast, an emergency top-up keeps sold <=
  available, and that event is RECORDED (`near_stockout_flag` /
  `emergency_units` in `inventory_snapshots`) rather than silently folded
  into `units_received` -- a Stage 3.5 fix, so Stage 6's forecast has a
  real, detectable near-stockout signal instead of one smoothed away.
- SKU demand is Pareto-skewed, not uniform: each SKU gets a `popularity_weight`
  drawn from a Pareto distribution, used to weight which SKU an order line
  picks. This produces a realistic head of best-sellers and a long tail of
  slow/near-dead SKUs, instead of ~150 SKUs each selling about the same
  (which a Stage-2 diagnostic caught: uniform sampling had produced a flat
  top-10-SKUs-are-18%-of-revenue result, unrealistic for fashion).
- A small subset (~7%) of SKUs are deliberately seeded as structurally weak
  "problem SKUs": elevated return rate (50-68%) AND elevated COGS-as-%-of-
  price (58-72%), landing some of them at negative contribution margin even
  before marketing cost is considered, and pushing others there once CAC is
  allocated. One channel (Influencer) is deliberately calibrated inefficient
  (high cost per order relative to what its orders are worth), so it fails
  to earn back its own acquisition cost. Both exist so the Stage 5
  margin-risk alert model has genuine loss-makers to catch -- a Stage 2/3
  diagnostic found zero loss-making channels and only 3 of 150 SKUs
  loss-making under the original (uncalibrated) generator, too thin a
  signal for an alert model to be meaningful against.

Reproducibility
----------------
Every function takes a `numpy.random.Generator` seeded from a single
top-level seed (default 42) via `generate_all(seed=...)`, so the entire
dataset regenerates identically on any machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

# ---------------------------------------------------------------------------
# Fixed generation parameters
# ---------------------------------------------------------------------------

START_DATE = pd.Timestamp("2025-07-01")
END_DATE = pd.Timestamp("2026-06-30")   # inclusive; 12 full months
N_MONTHS = 12
CURRENCY = "INR"  # illustrative only; every monetary column is currency-agnostic

# Category economics: price band, COGS-as-%-of-price band, return rate,
# how many distinct SKUs to create, and the size variants those SKUs come in.
CATEGORY_SPECS = {
    "Tops":        {"sizes": ["XS", "S", "M", "L", "XL"], "price_range": (399, 1499),
                     "cogs_pct_range": (0.32, 0.42), "return_rate": 0.32, "n_skus": 35},
    "Bottoms":     {"sizes": ["XS", "S", "M", "L", "XL"], "price_range": (699, 2199),
                     "cogs_pct_range": (0.34, 0.44), "return_rate": 0.28, "n_skus": 30},
    "Dresses":     {"sizes": ["XS", "S", "M", "L", "XL"], "price_range": (899, 3499),
                     "cogs_pct_range": (0.30, 0.40), "return_rate": 0.38, "n_skus": 30},
    "Outerwear":   {"sizes": ["XS", "S", "M", "L", "XL"], "price_range": (1499, 4999),
                     "cogs_pct_range": (0.35, 0.45), "return_rate": 0.18, "n_skus": 20},
    "Footwear":    {"sizes": ["6", "7", "8", "9", "10", "11"], "price_range": (999, 3999),
                     "cogs_pct_range": (0.40, 0.50), "return_rate": 0.22, "n_skus": 20},
    "Accessories": {"sizes": ["One Size"], "price_range": (199, 1499),
                     "cogs_pct_range": (0.25, 0.35), "return_rate": 0.08, "n_skus": 15},
}

# Marketing channels: base efficiency = expected spend per order credited to
# that channel (Organic/Email is ~free -- it's word-of-mouth / retention,
# not a paid acquisition channel). Influencer is deliberately calibrated
# expensive -- a real, common D2C failure mode (glossy vanity-metric
# campaigns that don't earn back their cost) -- so it comes out
# contribution-margin-negative after CAC in the Stage 3 diagnostic, giving
# the Stage 5 alert model a genuine channel-level loss-maker to catch.
CHANNEL_SPECS = {
    "Instagram Ads": {"cost_per_order": (350, 90), "new_customer_weight": 0.34},
    "Google Ads":    {"cost_per_order": (420, 110), "new_customer_weight": 0.24},
    "Influencer":    {"cost_per_order": (1650, 350), "new_customer_weight": 0.16},
    "Affiliate":     {"cost_per_order": (280, 70), "new_customer_weight": 0.10},
    "Organic/Email": {"cost_per_order": (0, 0), "new_customer_weight": 0.16},
}
CHANNELS = list(CHANNEL_SPECS.keys())

# Sale months get a demand bump and heavier discounting.
SALE_MONTHS = {pd.Timestamp("2025-11-01").to_period("M"), pd.Timestamp("2026-06-01").to_period("M")}

# Per-month decay applied to a customer's odds of being selected for a
# repeat order, based on months since their last order (see
# `generate_orders`). Lower = faster churn = sharper cohort-retention decay.
RETAIN_DECAY_RATE = 0.55


def _month_index(ts: pd.Timestamp) -> int:
    return (ts.year - START_DATE.year) * 12 + (ts.month - START_DATE.month)


# Fraction of SKUs seeded as structurally weak "problem SKUs" -- elevated
# return rate AND elevated COGS-as-%-of-price, independent of category.
# Models a real, common failure mode (a sizing issue driving returns, or a
# supplier cost renegotiation gone bad) that a category-level average would
# never reveal on its own.
PROBLEM_SKU_FRACTION = 0.07
PROBLEM_SKU_COGS_PCT_RANGE = (0.58, 0.72)
PROBLEM_SKU_RETURN_RATE_RANGE = (0.50, 0.68)

# Shape of the Pareto distribution used for the SKU popularity base weight.
# Lower alpha = heavier tail = more concentrated demand. Chosen empirically
# (3.5) to land top-10-SKUs-by-revenue around 35-45% -- a uniform weight
# (the original Stage 1/2 approach) produced an unrealistic ~18%, while a
# heavier tail (alpha ~1.4) over-concentrated to ~75-90%, more Zipfian than
# real fashion retail. A pure Pareto draw at this alpha rarely produces a
# genuine dead-stock tail on its own, so a separate SLOW_MOVER mechanism
# below guarantees one explicitly, rather than leaving it to chance.
SKU_POPULARITY_PARETO_ALPHA = 3.5

# A designated slow-mover subset gets its popularity weight crushed by this
# factor range, guaranteeing a real long tail of near-dead-stock SKUs
# regardless of how the Pareto draw above happens to land.
SLOW_MOVER_FRACTION = 0.10
SLOW_MOVER_WEIGHT_MULTIPLIER_RANGE = (0.03, 0.12)


def _build_sku_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Internal product master: one row per SKU. Not exported as its own
    file -- a real order export is usually already denormalized with
    category/size/price on the order line, which is the shape orders.csv
    uses. This catalog exists only to generate orders/inventory consistently.

    Three calibration mechanisms live here (added after a Stage 2/3
    diagnostic found the original uniform-demand, uniformly-healthy-margin
    catalog gave the margin-risk alert model nothing real to catch):

    - `popularity_weight`: a Pareto-distributed base weight, used in
      `generate_orders` to bias which SKU an order line picks, producing a
      realistic head of best-sellers instead of every SKU selling about
      the same amount.
    - a designated `is_slow_mover` subset (~10% of SKUs) has that weight
      crushed by an extra small multiplier, guaranteeing a genuine
      near-dead-stock tail rather than leaving it to chance -- a pure
      Pareto draw at the alpha needed for a realistic head concentration
      rarely produces enough true dead stock on its own.
    - `is_problem_sku` / overridden `return_rate` and `unit_cogs`: ~7% of
      SKUs get a materially worse return rate and cost structure than their
      category average, landing some at negative contribution margin
      outright and others there once marketing cost is allocated.
    """
    rows = []
    for category, spec in CATEGORY_SPECS.items():
        for i in range(spec["n_skus"]):
            base_price = rng.uniform(*spec["price_range"])
            is_problem = rng.random() < PROBLEM_SKU_FRACTION
            if is_problem:
                cogs_pct = rng.uniform(*PROBLEM_SKU_COGS_PCT_RANGE)
                return_rate = rng.uniform(*PROBLEM_SKU_RETURN_RATE_RANGE)
            else:
                cogs_pct = rng.uniform(*spec["cogs_pct_range"])
                return_rate = spec["return_rate"]
            # Most SKUs launch at the start of the window; ~15% are
            # "new arrivals" launched partway through the year, so
            # inventory/sell-through analysis has genuinely young SKUs.
            launch_month = 0 if rng.random() > 0.15 else rng.integers(1, N_MONTHS - 2)
            rows.append({
                "sku_id": f"{category[:3].upper()}-{i + 1:03d}",
                "category": category,
                "sizes": spec["sizes"],
                "base_price": round(base_price, -1) - 1,   # e.g. 899, 1299 style pricing
                "unit_cogs": round(base_price * cogs_pct, 2),
                "return_rate": return_rate,
                "is_problem_sku": is_problem,
                "launch_month": launch_month,
            })
    catalog = pd.DataFrame(rows)

    n = len(catalog)
    popularity = rng.pareto(SKU_POPULARITY_PARETO_ALPHA, size=n) + 0.05
    is_slow_mover = rng.random(n) < SLOW_MOVER_FRACTION
    popularity[is_slow_mover] *= rng.uniform(*SLOW_MOVER_WEIGHT_MULTIPLIER_RANGE, size=is_slow_mover.sum())
    catalog["popularity_weight"] = popularity
    catalog["is_slow_mover"] = is_slow_mover
    return catalog


def _daily_order_volume(rng: np.random.Generator) -> pd.Series:
    """Baseline daily order count: grows over the year and spikes in sale months."""
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    growth = np.linspace(28, 58, len(days))  # brand growing: ~28 orders/day -> ~58/day
    month_periods = days.to_period("M")
    sale_bump = np.where(month_periods.isin(SALE_MONTHS), 1.6, 1.0)
    weekday_effect = np.where(days.dayofweek >= 5, 1.25, 1.0)  # weekends busier
    noise = rng.normal(1.0, 0.12, len(days)).clip(0.6, 1.6)
    lam = growth * sale_bump * weekday_effect * noise
    counts = rng.poisson(lam)
    return pd.Series(counts, index=days)


def generate_orders(rng: np.random.Generator, sku_catalog: pd.DataFrame,
                     n_customers_target: int = 5500) -> pd.DataFrame:
    """Generate order-line-grain synthetic orders.

    Customer model: a pool of customers grows through the year; each order
    is placed either by a brand-new customer or by drawing from the
    existing pool, with the repeat-purchase probability rising from 15%
    (early, pool is small) to a ceiling of 40% as the pool approaches
    `n_customers_target` -- this produces a realistic mix of new and
    repeat orders for the repeat-purchase-rate and cohort analysis in
    Stage 3. `n_customers_target` shapes this ramp; it is not a hard cap,
    so the final unique-customer count will typically exceed it somewhat.
    """
    daily_orders = _daily_order_volume(rng)
    total_orders = int(daily_orders.sum())

    # Cap the eventual customer pool near the target by tuning the
    # new-vs-repeat split as the pool grows. Preallocated arrays (not a
    # plain Python list) so recency-weighted repeat selection below stays
    # fast even as the pool grows into the thousands.
    max_customers = n_customers_target * 4
    customer_pool = np.empty(max_customers, dtype=object)
    last_order_month = np.full(max_customers, -999, dtype=np.int32)
    pool_size = 0
    next_customer_num = 1

    order_rows = []
    line_rows = []
    order_counter = 1

    sale_month_periods = SALE_MONTHS

    for order_date, n_orders_today in daily_orders.items():
        month_period = order_date.to_period("M")
        m_idx = _month_index(order_date)
        is_sale_month = month_period in sale_month_periods

        for _ in range(int(n_orders_today)):
            # Repeat probability rises from ~15% early on to ~40% once the
            # pool is established, capped so the pool still grows toward target.
            repeat_prob = min(0.40, 0.15 + 0.30 * (pool_size / n_customers_target))
            if pool_size > 0 and rng.random() < repeat_prob:
                # Repeat customer selection is RECENCY-weighted, not
                # uniform: a customer who ordered last month is far more
                # likely to reorder than one who ordered 6 months ago
                # (RETAIN_DECAY_RATE ** months_since_last_order). This is
                # what makes cohort retention actually decay by cohort age
                # -- a Stage 3 diagnostic found uniform selection produced
                # flat, noisy retention with no decay pattern at all.
                months_since = m_idx - last_order_month[:pool_size]
                weights = RETAIN_DECAY_RATE ** np.maximum(months_since, 0)
                weights = weights / weights.sum()
                idx = rng.choice(pool_size, p=weights)
                customer_id = customer_pool[idx]
                last_order_month[idx] = m_idx
                is_new_customer = False
            else:
                customer_id = f"CUST-{next_customer_num:05d}"
                customer_pool[pool_size] = customer_id
                last_order_month[pool_size] = m_idx
                pool_size += 1
                next_customer_num += 1
                is_new_customer = True

            # Marketing channel attribution: new customers skew paid,
            # repeat customers skew Organic/Email.
            if is_new_customer:
                weights = np.array([CHANNEL_SPECS[c]["new_customer_weight"] for c in CHANNELS])
            else:
                weights = np.array([0.06, 0.05, 0.04, 0.05, 0.80])  # mostly Organic/Email
            weights = weights / weights.sum()
            channel = rng.choice(CHANNELS, p=weights)

            order_id = f"ORD-{order_counter:06d}"
            order_counter += 1

            # 1-3 line items per order, weighted toward 1. SKU choice is
            # weighted by `popularity_weight` (Pareto-distributed), not
            # uniform, so a head of best-sellers gets picked far more often
            # than the long tail. Uses numpy's weighted choice without
            # replacement directly -- pandas' `.sample(weights=...)` raises
            # when a single weight dominates the rest, which a Pareto tail
            # does by design.
            n_lines = rng.choice([1, 2, 3], p=[0.68, 0.24, 0.08])
            eligible_skus = sku_catalog[sku_catalog["launch_month"] <= m_idx]
            eligible_p = eligible_skus["popularity_weight"].to_numpy()
            eligible_p = eligible_p / eligible_p.sum()
            chosen_idx = rng.choice(len(eligible_skus), size=min(n_lines, len(eligible_skus)),
                                     replace=False, p=eligible_p)
            chosen = eligible_skus.iloc[chosen_idx]

            for _, sku_row in chosen.iterrows():
                quantity = 1 if rng.random() < 0.85 else 2
                size = rng.choice(sku_row["sizes"])

                # Discount: light everyday variance, heavier in sale months.
                if is_sale_month:
                    discount = rng.uniform(0.15, 0.40)
                else:
                    discount = rng.choice([0.0, 0.0, 0.0, rng.uniform(0.05, 0.15)], p=[0.55, 0.15, 0.15, 0.15]) \
                        if rng.random() > 0.8 else 0.0
                unit_price = round(sku_row["base_price"] * (1 - discount), 2)

                shipping_cost = round(rng.normal(70, 12), 2)
                shipping_cost = max(shipping_cost, 30.0)
                payment_fee = round(unit_price * quantity * 0.021 + 2.0, 2)  # ~2.1% + fixed fee

                is_return = rng.random() < sku_row["return_rate"]
                return_date = pd.NaT
                if is_return:
                    return_lag = rng.integers(4, 21)
                    return_date = order_date + pd.Timedelta(days=int(return_lag))

                line_rows.append({
                    schema.ORDER_ID: order_id,
                    schema.ORDER_LINE_ID: f"{order_id}-{len(line_rows) % 10 + 1}",
                    schema.ORDER_DATE: order_date,
                    schema.CUSTOMER_ID: customer_id,
                    schema.SKU_ID: sku_row["sku_id"],
                    schema.CATEGORY: sku_row["category"],
                    schema.SIZE: size,
                    schema.QUANTITY: int(quantity),
                    schema.UNIT_PRICE: unit_price,
                    schema.LIST_PRICE: sku_row["base_price"],
                    schema.UNIT_COGS: sku_row["unit_cogs"],
                    schema.SHIPPING_COST: shipping_cost,
                    schema.PAYMENT_FEE: payment_fee,
                    schema.MARKETING_CHANNEL: channel,
                    schema.IS_RETURN: bool(is_return),
                    schema.RETURN_DATE: return_date,
                })

    orders_df = pd.DataFrame(line_rows)
    # order_line_id must be globally unique; rebuild cleanly now that we
    # know each order's line count.
    orders_df[schema.ORDER_LINE_ID] = (
        orders_df.groupby(schema.ORDER_ID).cumcount().add(1).astype(str).radd(orders_df[schema.ORDER_ID] + "-")
    )
    return orders_df.sort_values(schema.ORDER_DATE).reset_index(drop=True)


def generate_marketing_spend(rng: np.random.Generator, orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly spend per channel, built to roughly track the orders each
    channel is credited with in that month (plus noise and a channel
    efficiency factor), so CAC/ROAS differ meaningfully by channel.
    """
    orders_only = orders_df.drop_duplicates(subset=schema.ORDER_ID).copy()
    orders_only["month"] = orders_only[schema.ORDER_DATE].dt.to_period("M").astype(str)

    counts = orders_only.groupby(["month", schema.MARKETING_CHANNEL]).size().reset_index(name="orders_credited")

    rows = []
    for _, r in counts.iterrows():
        channel = r[schema.MARKETING_CHANNEL]
        cost_mean, cost_sd = CHANNEL_SPECS[channel]["cost_per_order"]
        if cost_mean == 0:
            spend = 0.0
        else:
            per_order_cost = max(rng.normal(cost_mean, cost_sd), cost_mean * 0.4)
            spend = round(per_order_cost * r["orders_credited"], 2)
        rows.append({schema.SPEND_MONTH: r["month"], schema.SPEND_CHANNEL: channel, schema.SPEND_AMOUNT: spend})

    return pd.DataFrame(rows).sort_values([schema.SPEND_MONTH, schema.SPEND_CHANNEL]).reset_index(drop=True)


def generate_inventory_snapshots(rng: np.random.Generator, sku_catalog: pd.DataFrame,
                                  orders_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly opening/closing stock per SKU.

    units_sold is realized demand (gross units sold before returns, taken
    straight from orders_df, so the two tables reconcile). Purchasing
    (units_received) is generated to roughly cover realized-plus-buffer
    demand for most SKUs, but ~15% of SKUs are deliberately over-bought
    (too much stock relative to what sells -> dead stock) and ~10% are
    under-bought (too little -> stockout risk), so later stages have real
    overstock/stockout cases to detect rather than a uniformly well-run
    warehouse.

    `near_stockout_flag` / `emergency_units`: whenever the restock
    heuristic under-forecasts a month's realized demand (Pareto-skewed
    popularity and sale-month spikes both cause bursts above a SKU's
    average), the shortfall is topped up so units_sold never exceeds what
    was available -- and that event is recorded here rather than folded
    invisibly into `units_received`. This is the detectable near-stockout
    signal Stage 6's demand/inventory forecast is meant to catch.
    """
    orders_df = orders_df.copy()
    orders_df["month"] = orders_df[schema.ORDER_DATE].dt.to_period("M").astype(str)
    monthly_sold = (
        orders_df.groupby([schema.SKU_ID, "month"])[schema.QUANTITY].sum().rename("units_sold")
    )

    months = pd.period_range(START_DATE, END_DATE, freq="M").astype(str)

    # Assign each SKU a purchasing "style": normal, over-bought, under-bought.
    style_draw = rng.random(len(sku_catalog))
    styles = np.where(style_draw < 0.15, "over",
              np.where(style_draw < 0.25, "under", "normal"))
    sku_style = dict(zip(sku_catalog["sku_id"], styles))

    rows = []
    for sku_id, launch_month in zip(sku_catalog["sku_id"], sku_catalog["launch_month"]):
        style = sku_style[sku_id]
        # Estimate average monthly demand for this SKU from realized sales
        # (fallback to a small constant for SKUs with sparse sales).
        sku_sales = monthly_sold.get(sku_id, pd.Series(dtype=float))
        avg_demand = sku_sales.mean() if len(sku_sales) else 3.0
        avg_demand = max(avg_demand, 2.0)

        opening = {
            "over": avg_demand * rng.uniform(3.0, 4.5),
            "under": avg_demand * rng.uniform(0.4, 0.7),
            "normal": avg_demand * rng.uniform(1.3, 2.0),
        }[style]
        beginning = round(opening)

        for m_idx, month in enumerate(months):
            if m_idx < launch_month:
                continue  # SKU doesn't exist yet
            sold = int(monthly_sold.get((sku_id, month), 0))

            target_cover = {"over": 3.5, "under": 0.6, "normal": 1.5}[style]
            restock_trigger = beginning < avg_demand * target_cover
            if restock_trigger:
                received = int(max(0, round(avg_demand * target_cover * rng.uniform(0.85, 1.15))))
            else:
                received = 0

            # The restock heuristic above is based on this SKU's AVERAGE
            # monthly demand, but realized demand (`sold`, from actual
            # orders.csv) can burst well above average -- Pareto-skewed
            # popularity and sale-month spikes both do this. Selling more
            # than was ever in stock is a logical impossibility, so an
            # under-forecast restock is topped up to cover realized demand
            # exactly (an emergency reorder/backorder fulfillment, in
            # business terms) -- this keeps orders.csv and
            # inventory_snapshots.csv reconciled while still leaving
            # `beginning` inventory low for "under" style SKUs, which is
            # what actually drives their stockout-risk signal (low
            # beginning stock, thin weeks-of-cover), not an impossible sale.
            #
            # Stage 3.5 fix: this event is now RECORDED, not silently
            # folded into `received` -- a diagnostic found the original
            # version smoothed the emergency top-up away without a trace,
            # leaving Stage 6 nothing to detect or forecast against.
            shortfall = sold - (beginning + received)
            near_stockout = shortfall > 0
            emergency_units = max(shortfall, 0)
            if near_stockout:
                received += shortfall

            ending = max(beginning + received - sold, 0)
            rows.append({
                schema.INV_SKU_ID: sku_id,
                schema.INV_MONTH: month,
                schema.INV_BEGIN: beginning,
                schema.INV_RECEIVED: received,
                schema.INV_SOLD: sold,
                schema.INV_END: ending,
                schema.INV_NEAR_STOCKOUT: bool(near_stockout),
                schema.INV_EMERGENCY_UNITS: int(emergency_units),
            })
            beginning = ending

    return pd.DataFrame(rows).sort_values([schema.INV_SKU_ID, schema.INV_MONTH]).reset_index(drop=True)


# Monthly overhead: base amount + slow growth (headcount/rent creep) + small
# noise. This is the "cost pool" that doesn't attach to any single order --
# it funds the business existing, not any one sale.
OPEX_CATEGORY_SPECS = {
    "Salaries & Team":     {"base": 480000, "monthly_growth": 6000, "noise_pct": 0.03},
    "Rent & Utilities":    {"base": 150000, "monthly_growth": 1500, "noise_pct": 0.02},
    "Software & Tools":    {"base": 60000, "monthly_growth": 800, "noise_pct": 0.05},
    "Warehousing & Ops":   {"base": 110000, "monthly_growth": 2000, "noise_pct": 0.06},
    "Professional & Misc": {"base": 45000, "monthly_growth": 500, "noise_pct": 0.15},
}


def generate_opex(rng: np.random.Generator) -> pd.DataFrame:
    """Monthly operating expenses by category -- overhead that funds the
    business existing (salaries, rent, tools) rather than any single order.
    Grows slowly month over month (a growing team/footprint) with small
    category-specific noise.
    """
    months = pd.period_range(START_DATE, END_DATE, freq="M").astype(str)
    rows = []
    for category, spec in OPEX_CATEGORY_SPECS.items():
        for m_idx, month in enumerate(months):
            amount = spec["base"] + spec["monthly_growth"] * m_idx
            amount *= 1 + rng.normal(0, spec["noise_pct"])
            rows.append({
                schema.OPEX_MONTH: month,
                schema.OPEX_CATEGORY: category,
                schema.OPEX_AMOUNT: round(max(amount, 0), 2),
            })
    return pd.DataFrame(rows).sort_values([schema.OPEX_MONTH, schema.OPEX_CATEGORY]).reset_index(drop=True)


def generate_all(seed: int = 42, n_customers_target: int = 5500) -> dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset. Returns a dict of DataFrames:
    {'orders', 'marketing_spend', 'inventory_snapshots', 'opex'}.

    *** SYNTHETIC DATA *** -- see module docstring for generation assumptions.
    """
    rng = np.random.default_rng(seed)
    sku_catalog = _build_sku_catalog(rng)
    orders_df = generate_orders(rng, sku_catalog, n_customers_target=n_customers_target)
    marketing_df = generate_marketing_spend(rng, orders_df)
    inventory_df = generate_inventory_snapshots(rng, sku_catalog, orders_df)
    opex_df = generate_opex(rng)

    orders_df = orders_df.drop(columns=[c for c in orders_df.columns if c not in schema.ORDERS_SCHEMA])
    schema.validate_columns(orders_df, schema.ORDERS_SCHEMA, "orders")
    schema.validate_columns(marketing_df, schema.MARKETING_SCHEMA, "marketing_spend")
    schema.validate_columns(inventory_df, schema.INVENTORY_SCHEMA, "inventory_snapshots")
    schema.validate_columns(opex_df, schema.OPEX_SCHEMA, "opex")

    return {
        "orders": orders_df,
        "marketing_spend": marketing_df,
        "inventory_snapshots": inventory_df,
        "opex": opex_df,
    }


def save_all(dfs: dict[str, pd.DataFrame], out_dir: str) -> None:
    """Write the generated tables to CSV under `out_dir`."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for name, df in dfs.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic D2C fashion dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/synthetic")
    args = parser.parse_args()

    data = generate_all(seed=args.seed)
    save_all(data, args.out)
    for name, df in data.items():
        print(f"{name}: {df.shape}")
