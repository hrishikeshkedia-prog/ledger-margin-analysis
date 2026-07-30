"""Stage 2 tests: cleaning pipeline decision log + Universal Core KPI formulas."""

import numpy as np
import pandas as pd
import pytest

from ceo_dashboard.src import clean, core, data_generator as dg, raw_noise, schema


@pytest.fixture(scope="module")
def generated():
    return dg.generate_all(seed=42)


@pytest.fixture(scope="module")
def cleaned(generated):
    df, _ = clean.clean_orders(generated["orders"])
    return df


# ---------------------------------------------------------------------------
# clean.py
# ---------------------------------------------------------------------------

def test_clean_orders_is_noop_on_already_clean_data(generated):
    _, log = clean.clean_orders(generated["orders"])
    assert (log["rows_affected"] == 0).all()


def test_clean_orders_catches_injected_messiness(generated):
    rng = np.random.default_rng(1)
    messy = raw_noise.make_messy(generated["orders"], rng)
    cleaned_df, log = clean.clean_orders(messy)
    assert (log["rows_affected"] > 0).all()
    # normalise_categories should leave no leading/trailing whitespace, and
    # collapse the injected UPPERCASE variants back to the canonical casing
    assert not cleaned_df[schema.CATEGORY].str.contains(r"^\s|\s$", regex=True).any()
    assert set(cleaned_df[schema.CATEGORY].unique()) == set(dg.CATEGORY_SPECS.keys())


def test_clean_never_imputes_missing_cogs(generated):
    rng = np.random.default_rng(2)
    messy = raw_noise.make_messy(generated["orders"], rng)
    cleaned_df, _ = clean.clean_orders(messy)
    still_missing = cleaned_df[schema.UNIT_COGS].isna()
    assert (cleaned_df.loc[still_missing, "valid_economics"] == False).all()  # noqa: E712


def test_deduplication_removes_exact_duplicates(generated):
    rng = np.random.default_rng(3)
    messy = raw_noise.make_messy(generated["orders"], rng)
    cleaned_df, _ = clean.clean_orders(messy)
    assert cleaned_df[schema.ORDER_LINE_ID].is_unique


# ---------------------------------------------------------------------------
# core.py -- revenue / margin
# ---------------------------------------------------------------------------

def test_net_revenue_excludes_returns(cleaned):
    econ = core.add_line_economics(cleaned)
    returned = econ[econ[schema.IS_RETURN]]
    assert (returned["net_revenue_line"] == 0).all()
    assert (returned["net_cogs_line"] == 0).all()


def test_variable_costs_not_reversed_on_return(cleaned):
    econ = core.add_line_economics(cleaned)
    returned = econ[econ[schema.IS_RETURN]]
    assert (returned["variable_cost_line"] > 0).all()


def test_revenue_trend_hand_check(cleaned):
    econ = core.add_line_economics(cleaned)
    trend = core.revenue_trend(cleaned)
    first_month = econ["month"].min()
    expected_gross = econ.loc[econ["month"] == first_month, "gross_revenue_line"].sum()
    actual_gross = trend.loc[trend["month"] == first_month, "gross_revenue"].iloc[0]
    assert actual_gross == pytest.approx(expected_gross)


def test_gross_margin_pct_between_0_and_1(cleaned):
    gm = core.gross_margin_trend(cleaned)
    assert gm["gross_margin_pct"].between(0, 1).all()


def test_contribution_margin_le_gross_margin(cleaned):
    econ = core.add_line_economics(cleaned)
    assert (econ["contribution_margin_line"] <= econ["gross_margin_line"] + 1e-6).all()


def test_returns_margin_bridge_reconciles_to_contribution_margin(cleaned):
    bridge = core.returns_margin_bridge(cleaned)
    cm = core.contribution_margin_trend(cleaned)
    merged = bridge.merge(cm, on="month")
    np.testing.assert_allclose(
        merged["net_contribution_margin"].to_numpy(), merged["contribution_margin_abs"].to_numpy()
    )


def test_returns_margin_impact_is_never_positive(cleaned):
    bridge = core.returns_margin_bridge(cleaned)
    assert (bridge["returns_margin_impact"] <= 0).all()


def test_cogs_breakdown_shares_sum_to_one(cleaned):
    b = core.cogs_breakdown(cleaned)
    assert b["pct_of_total_cogs"].sum() == pytest.approx(1.0)


def test_opex_breakdown_shares_sum_to_one(generated):
    b = core.opex_breakdown(generated["opex"])
    assert b["pct_of_total_opex"].sum() == pytest.approx(1.0)


def test_company_pnl_operating_profit_consistent(cleaned, generated):
    pnl = core.company_pnl_trend(cleaned, generated["opex"])
    recomputed = pnl["contribution_margin_abs"] - pnl["total_opex"]
    np.testing.assert_allclose(pnl["operating_profit"].to_numpy(), recomputed.to_numpy())


# ---------------------------------------------------------------------------
# core.py -- SKU views
# ---------------------------------------------------------------------------

def test_margin_by_sku_covers_all_skus(cleaned):
    mbs = core.margin_by_sku(cleaned)
    assert mbs[schema.SKU_ID].nunique() == cleaned[schema.SKU_ID].nunique()


def test_sku_concentration_shares_are_monotonic(cleaned):
    ranked, summary = core.sku_concentration(cleaned)
    assert ranked["cumulative_share"].is_monotonic_increasing
    assert ranked["cumulative_share"].iloc[-1] == pytest.approx(1.0)
    assert 0 < summary["top_10_skus_revenue_share"] < summary["top_20pct_skus_revenue_share"] < 1


# ---------------------------------------------------------------------------
# core.py -- inventory + working capital
# ---------------------------------------------------------------------------

def test_inventory_turns_positive(generated, cleaned):
    turns = core.inventory_turns(generated["inventory_snapshots"], cleaned)
    assert (turns["inventory_turns_monthly"] > 0).all()
    assert (turns["days_inventory_outstanding"] > 0).all()


def test_working_capital_cycle_formula(generated, cleaned):
    wc = core.working_capital_cycle(generated["inventory_snapshots"], cleaned)
    expected = wc["dso_days"] + wc["days_inventory_outstanding"] - wc["dpo_days"]
    np.testing.assert_allclose(wc["cash_conversion_cycle"].to_numpy(), expected.to_numpy())
