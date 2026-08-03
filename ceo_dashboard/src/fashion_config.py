"""
Layer 2 config: D2C Fashion industry parameters.

This is the ENTIRE point of the two-layer architecture. Every number in
`fashion.py` that represents a business judgment call -- what counts as a
"high" return rate, how long a customer is assumed to keep buying, what
counts as dead stock -- lives here as plain data, never hardcoded inside a
function. Porting this dashboard to a different goods business (FMCG, auto
parts, electronics) should mean writing a NEW file shaped like this one
and pointing `fashion.py`'s functions at it via the `config=` argument --
never editing `fashion.py` itself, and certainly never `core.py`.

Every key below states, in its comment, the business judgment it encodes
and (where relevant) why it can't simply be computed from data.
"""

FASHION_CONFIG = {
    "industry_name": "D2C Fashion",

    # LTV cannot be OBSERVED from only 12 months of order history -- no
    # customer in this dataset has a multi-year purchase history to fit a
    # retention curve against. This is a stated assumption standing in for
    # that curve: the average number of orders a customer is assumed to
    # place over their entire relationship with the brand. A real business
    # would replace this with an empirically fitted number once it has
    # 2-3+ years of cohort data to actually observe decay against.
    "ltv_assumed_customer_lifetime_orders": 3.5,

    # A category (or SKU)'s return rate at or above this is labeled
    # "high-return" / fit-sensitive in the returns-rate view.
    "high_return_rate_threshold": 0.30,

    # Cohort retention is reported out to this many months after
    # acquisition (bounded by the 12-month data window regardless).
    "cohort_window_months": 6,

    # Weeks-of-cover classification bands: below the low end is stockout
    # risk, above the high end is overstock risk, in between is healthy.
    "weeks_of_cover_healthy_range": (4.0, 12.0),

    # Dead-stock definition: a SKU must have been available at least this
    # many months (so a brand-new arrival isn't unfairly flagged) AND have
    # sold through less than this fraction of everything it ever had in
    # stock, to be labeled dead.
    "dead_stock_min_months_available": 3,
    "dead_stock_max_cumulative_sell_through": 0.15,

    # A repeat customer is one who placed at least this many orders within
    # the analysis window.
    "repeat_purchase_min_orders": 2,
}


# ---------------------------------------------------------------------------
# A second, deliberately different config -- proves the boundary is real.
# ---------------------------------------------------------------------------
# Every `fashion.py` function that reads a threshold takes `config=` as an
# argument (defaulting to FASHION_CONFIG above). Passing this config instead
# changes classification output -- more SKUs/customers reclassified, a
# different LTV -- with ZERO changes to fashion.py's code. This is the
# concrete proof, run in the notebook, that "swap the config" is a real
# mechanism and not just an architectural claim. It represents a more
# conservative analyst's judgment call on the same D2C fashion data (a
# stricter definition of "high return", a shorter assumed customer
# lifetime, a tighter dead-stock bar) -- not a different industry, since a
# genuinely different industry (FMCG, auto parts) would also need different
# CATEGORY/CHANNEL assumptions in `data_generator.py`, out of scope here.
CONSERVATIVE_FASHION_CONFIG = {
    **FASHION_CONFIG,
    "ltv_assumed_customer_lifetime_orders": 2.0,
    "high_return_rate_threshold": 0.25,
    "dead_stock_min_months_available": 2,
    "dead_stock_max_cumulative_sell_through": 0.25,
}
