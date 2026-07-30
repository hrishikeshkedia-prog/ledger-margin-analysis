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
  fraction of existing customers re-order in later months, which is what
  makes repeat-purchase-rate and cohort-retention analysis possible in
  Stage 3.
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
  stockout warnings) in later stages.

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
# not a paid acquisition channel).
CHANNEL_SPECS = {
    "Instagram Ads": {"cost_per_order": (350, 90), "new_customer_weight": 0.34},
    "Google Ads":    {"cost_per_order": (420, 110), "new_customer_weight": 0.24},
    "Influencer":    {"cost_per_order": (500, 150), "new_customer_weight": 0.16},
    "Affiliate":     {"cost_per_order": (280, 70), "new_customer_weight": 0.10},
    "Organic/Email": {"cost_per_order": (0, 0), "new_customer_weight": 0.16},
}
CHANNELS = list(CHANNEL_SPECS.keys())

# Sale months get a demand bump and heavier discounting.
SALE_MONTHS = {pd.Timestamp("2025-11-01").to_period("M"), pd.Timestamp("2026-06-01").to_period("M")}


def _month_index(ts: pd.Timestamp) -> int:
    return (ts.year - START_DATE.year) * 12 + (ts.month - START_DATE.month)


def _build_sku_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Internal product master: one row per SKU. Not exported as its own
    file -- a real order export is usually already denormalized with
    category/size/price on the order line, which is the shape orders.csv
    uses. This catalog exists only to generate orders/inventory consistently.
    """
    rows = []
    for category, spec in CATEGORY_SPECS.items():
        for i in range(spec["n_skus"]):
            base_price = rng.uniform(*spec["price_range"])
            cogs_pct = rng.uniform(*spec["cogs_pct_range"])
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
                "return_rate": spec["return_rate"],
                "launch_month": launch_month,
            })
    return pd.DataFrame(rows)


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
    # new-vs-repeat split as the pool grows.
    customer_pool: list[str] = []
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
            pool_size = len(customer_pool)
            # Repeat probability rises from ~15% early on to ~40% once the
            # pool is established, capped so the pool still grows toward target.
            repeat_prob = min(0.40, 0.15 + 0.30 * (pool_size / n_customers_target))
            if pool_size > 0 and rng.random() < repeat_prob:
                customer_id = customer_pool[rng.integers(0, pool_size)]
                is_new_customer = False
            else:
                customer_id = f"CUST-{next_customer_num:05d}"
                customer_pool.append(customer_id)
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

            # 1-3 line items per order, weighted toward 1.
            n_lines = rng.choice([1, 2, 3], p=[0.68, 0.24, 0.08])
            eligible_skus = sku_catalog[sku_catalog["launch_month"] <= m_idx]
            chosen = eligible_skus.sample(n=min(n_lines, len(eligible_skus)), random_state=rng.integers(0, 2**31 - 1))

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

            ending = max(beginning + received - sold, 0)
            rows.append({
                schema.INV_SKU_ID: sku_id,
                schema.INV_MONTH: month,
                schema.INV_BEGIN: beginning,
                schema.INV_RECEIVED: received,
                schema.INV_SOLD: sold,
                schema.INV_END: ending,
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
