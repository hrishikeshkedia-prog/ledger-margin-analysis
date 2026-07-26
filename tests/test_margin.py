import math

import pandas as pd

from src import margin, schema


def _ledger() -> pd.DataFrame:
    # Two periods, two rows each. Hand-verifiable allocation math below.
    return pd.DataFrame(
        {
            schema.TXN_ID: ["T1", "T2", "T3", "T4"],
            schema.TXN_DATE: pd.to_datetime(
                ["2024-01-10", "2024-01-20", "2024-02-05", "2024-02-15"]
            ),
            schema.REVENUE: [100.0, 200.0, 50.0, 150.0],
            schema.DIRECT_COST: [40.0, None, 20.0, 60.0],
            schema.QUANTITY: [10.0, 5.0, 2.0, 8.0],
        }
    )


def _overhead() -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.PERIOD_MONTH: pd.to_datetime(["2024-01-01", "2024-02-01"]),
            schema.COST_CATEGORY: ["rent", "rent"],
            schema.AMOUNT: [30.0, 40.0],
        }
    )


def test_direct_cost_only_ignores_overhead_and_leaves_missing_cost_null():
    result = margin.direct_cost_only(_ledger(), _overhead())
    assert result[margin.ALLOCATED_COST].tolist()[0] == 40.0
    assert math.isnan(result[margin.ALLOCATED_COST].tolist()[1])  # T2's direct_cost was None
    assert result[margin.MARGIN].tolist()[0] == 60.0
    assert math.isnan(result[margin.MARGIN].tolist()[1])
    assert result[margin.MARGIN_PCT].tolist()[0] == 0.6


def test_proportional_by_revenue_allocates_overhead_by_revenue_share():
    result = margin.proportional_by_revenue(_ledger(), _overhead())
    allocated = result[margin.ALLOCATED_COST].tolist()

    # Jan: revenue 100/300 and 200/300 shares of the 30 overhead pool -> 10 and 20
    assert math.isclose(allocated[0], 40.0 + 10.0)
    assert math.isclose(allocated[1], 0.0 + 20.0)  # missing direct_cost treated as 0
    # Feb: revenue 50/200 and 150/200 shares of the 40 overhead pool -> 10 and 30
    assert math.isclose(allocated[2], 20.0 + 10.0)
    assert math.isclose(allocated[3], 60.0 + 30.0)

    margins = result[margin.MARGIN].tolist()
    assert math.isclose(margins[0], 100.0 - 50.0)
    assert math.isclose(margins[1], 200.0 - 20.0)


def test_per_unit_by_quantity_allocates_overhead_by_quantity_share():
    result = margin.per_unit_by_quantity(_ledger(), _overhead())
    allocated = result[margin.ALLOCATED_COST].tolist()

    # Jan: quantity 10/15 and 5/15 shares of the 30 overhead pool -> 20 and 10
    assert math.isclose(allocated[0], 40.0 + 20.0)
    assert math.isclose(allocated[1], 0.0 + 10.0)
    # Feb: quantity 2/10 and 8/10 shares of the 40 overhead pool -> 8 and 32
    assert math.isclose(allocated[2], 20.0 + 8.0)
    assert math.isclose(allocated[3], 60.0 + 32.0)


def test_zero_revenue_period_yields_null_allocation_not_a_crash():
    ledger = pd.DataFrame(
        {
            schema.TXN_ID: ["T1"],
            schema.TXN_DATE: pd.to_datetime(["2024-01-10"]),
            schema.REVENUE: [0.0],
            schema.DIRECT_COST: [10.0],
            schema.QUANTITY: [1.0],
        }
    )
    overhead = pd.DataFrame(
        {
            schema.PERIOD_MONTH: pd.to_datetime(["2024-01-01"]),
            schema.COST_CATEGORY: ["rent"],
            schema.AMOUNT: [30.0],
        }
    )
    result = margin.proportional_by_revenue(ledger, overhead)
    assert math.isnan(result[margin.ALLOCATED_COST].iloc[0])


def test_compute_margin_dispatches_by_name():
    via_name = margin.compute_margin(_ledger(), _overhead(), "direct_cost_only")
    via_fn = margin.direct_cost_only(_ledger(), _overhead())
    pd.testing.assert_frame_equal(via_name, via_fn)


def test_all_strategies_registered_and_runnable():
    for name in margin.STRATEGIES:
        result = margin.compute_margin(_ledger(), _overhead(), name)
        assert margin.MARGIN in result.columns
        assert margin.MARGIN_PCT in result.columns
