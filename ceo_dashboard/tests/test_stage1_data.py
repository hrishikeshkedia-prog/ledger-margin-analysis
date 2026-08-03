"""Stage 1 sanity tests: synthetic generator output shape + loader round-trip."""

import pandas as pd
import pytest

from ceo_dashboard.src import data_generator as dg
from ceo_dashboard.src import data_loader as dl
from ceo_dashboard.src import schema


@pytest.fixture(scope="module")
def generated():
    return dg.generate_all(seed=42)


def test_tables_present(generated):
    assert set(generated.keys()) == {"orders", "marketing_spend", "inventory_snapshots", "opex"}


def test_schema_columns(generated):
    assert list(generated["orders"].columns) == schema.ORDERS_SCHEMA
    assert list(generated["marketing_spend"].columns) == schema.MARKETING_SCHEMA
    assert list(generated["inventory_snapshots"].columns) == schema.INVENTORY_SCHEMA
    assert list(generated["opex"].columns) == schema.OPEX_SCHEMA


def test_reproducible_with_same_seed():
    a = dg.generate_all(seed=7)
    b = dg.generate_all(seed=7)
    assert a["orders"].equals(b["orders"])
    assert a["marketing_spend"].equals(b["marketing_spend"])
    assert a["inventory_snapshots"].equals(b["inventory_snapshots"])


def test_return_rates_in_realistic_fashion_range(generated):
    orders = generated["orders"]
    by_cat = orders.groupby(schema.CATEGORY)[schema.IS_RETURN].mean()
    # at least one category should sit in the 20-40% band called for fashion
    assert (by_cat.between(0.20, 0.45)).any()


def test_inventory_never_negative(generated):
    assert (generated["inventory_snapshots"][schema.INV_END] >= 0).all()


def test_inventory_chain_consistency(generated):
    inv = generated["inventory_snapshots"].sort_values([schema.INV_SKU_ID, schema.INV_MONTH])
    for _, g in inv.groupby(schema.INV_SKU_ID):
        assert (g[schema.INV_END].iloc[:-1].values == g[schema.INV_BEGIN].iloc[1:].values).all()


def test_near_stockout_events_are_recorded(generated):
    """Stage 3.5: emergency restock top-ups must be a detectable signal
    (near_stockout_flag / emergency_units), not silently folded into
    units_received the way they originally were."""
    inv = generated["inventory_snapshots"]
    assert inv[schema.INV_NEAR_STOCKOUT].dtype == bool
    # emergency_units > 0 exactly when the flag is True, never otherwise
    assert (inv.loc[inv[schema.INV_NEAR_STOCKOUT], schema.INV_EMERGENCY_UNITS] > 0).all()
    assert (inv.loc[~inv[schema.INV_NEAR_STOCKOUT], schema.INV_EMERGENCY_UNITS] == 0).all()
    # the recalibrated (Pareto-skewed) generator should actually produce some
    assert inv[schema.INV_NEAR_STOCKOUT].sum() > 0


def test_loader_round_trip(tmp_path, generated):
    dg.save_all(generated, str(tmp_path))
    loaded = dl.load_orders(str(tmp_path / "orders.csv"))
    assert list(loaded.columns) == schema.ORDERS_SCHEMA
    assert len(loaded) == len(generated["orders"])
    assert pd.api.types.is_datetime64_any_dtype(loaded[schema.ORDER_DATE])


def test_loader_swappable_mapping(tmp_path, generated):
    dg.save_all(generated, str(tmp_path))
    raw = pd.read_csv(tmp_path / "orders.csv")
    raw = raw.rename(columns={"order_id": "Order No", "unit_price": "Selling Price"})
    raw.to_csv(tmp_path / "custom_export.csv", index=False)

    custom_mapping = dict(schema.DEFAULT_ORDERS_MAPPING)
    del custom_mapping["order_id"]
    del custom_mapping["unit_price"]
    custom_mapping["Order No"] = schema.ORDER_ID
    custom_mapping["Selling Price"] = schema.UNIT_PRICE

    loaded = dl.load_orders(str(tmp_path / "custom_export.csv"), mapping=custom_mapping)
    assert list(loaded.columns) == schema.ORDERS_SCHEMA


def test_loader_raises_on_missing_column(tmp_path, generated):
    dg.save_all(generated, str(tmp_path))
    raw = pd.read_csv(tmp_path / "orders.csv")
    raw = raw.drop(columns=["unit_price"])
    raw.to_csv(tmp_path / "broken_export.csv", index=False)
    with pytest.raises(ValueError):
        dl.load_orders(str(tmp_path / "broken_export.csv"))
