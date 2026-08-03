"""
Canonical column schema for the CEO Cost & Margin Dashboard.

Why this file exists
---------------------
Every calculation module in this project (Layer 1 core KPIs, Layer 2 fashion
module, alerts, forecasts) refers to columns by the constants defined here,
never by a hard-coded string. That means:

  1. If the raw data source changes (e.g. a real POS/Shopify export uses
     "order_no" instead of "order_id"), only `data_loader.py`'s column-mapping
     dict needs to change. No calculation code is touched.
  2. Every table's shape is documented in exactly one place.

Three tables make up the raw data layer:

  ORDERS_SCHEMA     - one row per order LINE ITEM (an order with 2 SKUs is
                       2 rows sharing the same order_id). This is the main
                       fact table: revenue, cost, returns, channel attribution
                       and customer identity all live here.
  MARKETING_SCHEMA  - one row per (month, channel): total spend. Spend is a
                       channel-level input in real businesses (an ad platform
                       invoice), not a property of a single order, so it is
                       kept in its own table and joined in when needed (e.g.
                       for CAC).
  INVENTORY_SCHEMA  - one row per (month, SKU): opening/closing stock. Needed
                       for turns, days-of-inventory, sell-through and
                       weeks-of-cover, none of which can be computed from
                       order data alone (you also need what was NOT sold).
"""

# ---------------------------------------------------------------------------
# Orders table (order-line grain)
# ---------------------------------------------------------------------------
ORDER_ID = "order_id"                 # groups line items placed in the same checkout
ORDER_LINE_ID = "order_line_id"       # unique row key
ORDER_DATE = "order_date"             # date the order was placed
CUSTOMER_ID = "customer_id"           # links repeat orders to the same buyer
SKU_ID = "sku_id"                     # product+variant identifier
CATEGORY = "category"                 # product category (Tops, Dresses, ...)
SIZE = "size"                         # size variant of the SKU
QUANTITY = "quantity"                 # units on this line
UNIT_PRICE = "unit_price"             # realized selling price per unit (post-discount)
LIST_PRICE = "list_price"             # the SKU's un-discounted price -- added in Stage 3
                                       # so a markdown % (unit_price < list_price) is
                                       # computable; purely additive, existing columns unchanged
UNIT_COGS = "unit_cogs"               # cost of goods per unit for this SKU
SHIPPING_COST = "shipping_cost"       # fulfillment/shipping cost allocated to this line
PAYMENT_FEE = "payment_gateway_fee"   # payment processor fee allocated to this line
MARKETING_CHANNEL = "marketing_channel"  # channel credited with driving the parent order
IS_RETURN = "is_return"               # True if this line was returned
RETURN_DATE = "return_date"           # date of return, NaT if not returned

ORDERS_SCHEMA = [
    ORDER_ID, ORDER_LINE_ID, ORDER_DATE, CUSTOMER_ID, SKU_ID, CATEGORY, SIZE,
    QUANTITY, UNIT_PRICE, LIST_PRICE, UNIT_COGS, SHIPPING_COST, PAYMENT_FEE,
    MARKETING_CHANNEL, IS_RETURN, RETURN_DATE,
]

# ---------------------------------------------------------------------------
# Marketing spend table (channel x month grain)
# ---------------------------------------------------------------------------
SPEND_MONTH = "month"
SPEND_CHANNEL = "channel"
SPEND_AMOUNT = "spend"

MARKETING_SCHEMA = [SPEND_MONTH, SPEND_CHANNEL, SPEND_AMOUNT]

# ---------------------------------------------------------------------------
# Inventory snapshots table (SKU x month grain)
# ---------------------------------------------------------------------------
INV_SKU_ID = "sku_id"
INV_MONTH = "month"
INV_BEGIN = "beginning_inventory"
INV_RECEIVED = "units_received"
INV_SOLD = "units_sold"
INV_END = "ending_inventory"
INV_NEAR_STOCKOUT = "near_stockout_flag"  # Stage 3.5: True if this month's planned
                                           # restock underforecast realized demand and
                                           # had to be emergency-topped-up to avoid
                                           # selling more than was ever in stock
INV_EMERGENCY_UNITS = "emergency_units"   # Stage 3.5: size of that emergency top-up
                                           # (0 if near_stockout_flag is False)

INVENTORY_SCHEMA = [
    INV_SKU_ID, INV_MONTH, INV_BEGIN, INV_RECEIVED, INV_SOLD, INV_END,
    INV_NEAR_STOCKOUT, INV_EMERGENCY_UNITS,
]

# ---------------------------------------------------------------------------
# Operating expenses table (expense category x month grain)
# ---------------------------------------------------------------------------
# Added in Stage 2 to support the Universal Core "operating expense
# breakdown" KPI, which needs overhead cost data that order-level rows can
# never contain (rent, salaries, tools are not a property of any one order).
OPEX_MONTH = "month"
OPEX_CATEGORY = "expense_category"
OPEX_AMOUNT = "amount"

OPEX_SCHEMA = [OPEX_MONTH, OPEX_CATEGORY, OPEX_AMOUNT]

# ---------------------------------------------------------------------------
# Default column mappings (identity mapping: raw header -> canonical name)
# ---------------------------------------------------------------------------
# data_loader.py renames incoming columns using dicts like these. To plug in
# a real CSV export whose headers differ, copy one of these dicts and edit
# the KEYS (raw headers) to match the real file -- the VALUES (canonical
# names) must stay as they are, since every downstream module depends on them.
DEFAULT_ORDERS_MAPPING = {col: col for col in ORDERS_SCHEMA}
DEFAULT_MARKETING_MAPPING = {col: col for col in MARKETING_SCHEMA}
DEFAULT_INVENTORY_MAPPING = {col: col for col in INVENTORY_SCHEMA}
DEFAULT_OPEX_MAPPING = {col: col for col in OPEX_SCHEMA}


def validate_columns(df, expected_columns, table_name):
    """Raise a clear error if a loaded/generated table is missing canonical columns.

    This is a boundary check: it runs once, right after loading or generating
    a table, so every calculation module downstream can assume the schema is
    correct and never has to defend against a missing column itself.
    """
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing expected columns after mapping: {missing}. "
            f"Check the column-mapping dict passed to the loader."
        )
