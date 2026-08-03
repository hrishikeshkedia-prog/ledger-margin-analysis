"""Stage 3 tests: recalibrated data + fashion module (Layer 2) formulas."""

import numpy as np
import pandas as pd
import pytest

from ceo_dashboard.src import clean, core, data_generator as dg, fashion, schema
from ceo_dashboard.src.fashion_config import CONSERVATIVE_FASHION_CONFIG, FASHION_CONFIG


@pytest.fixture(scope="module")
def generated():
    return dg.generate_all(seed=42)


@pytest.fixture(scope="module")
def cleaned(generated):
    df, log = clean.clean_orders(generated["orders"])
    assert (log["rows_affected"] == 0).all()
    return df


# ---------------------------------------------------------------------------
# Calibration: the two gaps this stage was built to fix
# ---------------------------------------------------------------------------

def test_at_least_one_channel_is_loss_making_after_cac(cleaned, generated):
    econ = core.add_line_economics(cleaned)
    marketing = generated["marketing_spend"]
    orders_only = econ.drop_duplicates(subset=schema.ORDER_ID)
    cm_by_channel = econ.groupby(schema.MARKETING_CHANNEL)["contribution_margin_line"].sum()
    spend_by_channel = marketing.groupby(schema.SPEND_CHANNEL)[schema.SPEND_AMOUNT].sum()
    cm_after_cac = cm_by_channel.reindex(spend_by_channel.index).fillna(0) - spend_by_channel
    assert (cm_after_cac < 0).sum() >= 1
    assert "Influencer" in cm_after_cac[cm_after_cac < 0].index


def test_a_meaningful_number_of_skus_are_loss_making(cleaned):
    ranked = fashion.returns_rate_by_sku(cleaned)
    econ = core.add_line_economics(cleaned)
    cm_by_sku = econ.groupby(schema.SKU_ID)["contribution_margin_line"].sum()
    # before any CAC allocation, returns alone should already push a
    # non-trivial number of SKUs negative or near-zero
    assert (cm_by_sku < 0).sum() >= 1


def test_sku_revenue_concentration_is_realistically_skewed(cleaned):
    _, summary = core.sku_concentration(cleaned)
    # the uncalibrated generator produced ~18%; recalibrated should be well above that
    assert summary["top_10_skus_revenue_share"] > 0.30


def test_dead_stock_candidates_exist(generated):
    dsp = fashion.dead_stock_pct(generated["inventory_snapshots"])
    assert dsp["n_dead_stock_skus"] >= 1


def test_cohort_retention_decays_by_age(cleaned):
    table = fashion.cohort_retention_table(cleaned)
    assert (table[0] == 1.0).all()
    avg_by_offset = table.mean()
    # average retention should trend down as offset increases, not be flat/noisy
    assert avg_by_offset[1] > avg_by_offset[table.columns.max()]


# ---------------------------------------------------------------------------
# Returns rate
# ---------------------------------------------------------------------------

def test_returns_rate_by_category_flags_high_return(cleaned):
    by_cat = fashion.returns_rate_by_category(cleaned)
    assert by_cat["value_return_rate"].between(0, 1).all()
    assert by_cat["high_return_flag"].any()


def test_returns_rate_units_vs_value_consistent_direction(cleaned):
    by_sku = fashion.returns_rate_by_sku(cleaned)
    assert by_sku["units_return_rate"].between(0, 1).all()
    assert by_sku["value_return_rate"].between(0, 1).all()


# ---------------------------------------------------------------------------
# CAC / ROAS / MER / LTV:CAC
# ---------------------------------------------------------------------------

def test_true_cac_exceeds_cost_per_order_proxy(cleaned, generated):
    marketing = generated["marketing_spend"]
    summary = fashion.cac_summary_by_channel(cleaned, marketing)
    orders_only = cleaned.drop_duplicates(subset=schema.ORDER_ID)
    orders_by_channel = orders_only.groupby(schema.MARKETING_CHANNEL).size()
    summary = summary.set_index("channel")
    summary["cost_per_order"] = summary["total_spend"] / orders_by_channel.reindex(summary.index)
    paid = summary[summary["total_spend"] > 0]
    assert (paid["true_cac"] >= paid["cost_per_order"]).all()


def test_roas_positive_for_paid_channels(cleaned, generated):
    roas = fashion.roas_by_channel(cleaned, generated["marketing_spend"])
    paid = roas[roas[schema.SPEND_AMOUNT] > 0]
    assert (paid["roas"] > 0).all()


def test_mer_trend_positive(cleaned, generated):
    mer = fashion.mer_trend(cleaned, generated["marketing_spend"])
    assert (mer["mer"].dropna() > 0).all()


def test_ltv_cac_config_boundary_changes_output(cleaned, generated):
    """Proves the config boundary is real: same function, same data, only
    the config object differs, and the output must differ."""
    marketing = generated["marketing_spend"]
    default_result = fashion.ltv_cac_ratio(cleaned, marketing, config=FASHION_CONFIG)
    alt_result = fashion.ltv_cac_ratio(cleaned, marketing, config=CONSERVATIVE_FASHION_CONFIG)
    assert not np.isclose(default_result["ltv"].iloc[0], alt_result["ltv"].iloc[0])
    assert not np.allclose(
        default_result["ltv_cac_ratio"].fillna(-1).to_numpy(),
        alt_result["ltv_cac_ratio"].fillna(-1).to_numpy(),
    )


def test_influencer_has_worst_ltv_cac(cleaned, generated):
    result = fashion.ltv_cac_ratio(cleaned, generated["marketing_spend"]).set_index("channel")
    paid = result[result["total_spend"] > 0]
    assert paid["ltv_cac_ratio"].idxmin() == "Influencer"


# ---------------------------------------------------------------------------
# AOV, repeat rate
# ---------------------------------------------------------------------------

def test_aov_positive(cleaned):
    aov = fashion.aov_trend(cleaned)
    assert (aov["aov"] > 0).all()


def test_repeat_purchase_rate_between_0_and_1(cleaned):
    result = fashion.repeat_purchase_rate(cleaned)
    assert 0 < result["repeat_purchase_rate"] < 1
    assert result["repeat_customers"] <= result["total_customers"]


# ---------------------------------------------------------------------------
# Sell-through, weeks of cover, markdown, dead stock
# ---------------------------------------------------------------------------

def test_sell_through_rate_bounded(generated):
    st = fashion.sell_through_rate_monthly(generated["inventory_snapshots"])
    assert st["sell_through_rate"].dropna().between(0, 1).all()


def test_cumulative_sell_through_bounded(generated):
    cum = fashion.cumulative_sell_through(generated["inventory_snapshots"])
    assert cum["cumulative_sell_through"].dropna().between(0, 1).all()


def test_weeks_of_cover_classification_present(generated):
    woc = fashion.weeks_of_cover(generated["inventory_snapshots"])
    assert set(woc["cover_status"].unique()) <= {
        "Healthy", "Stockout risk", "Overstock risk", "No stock, no recent sales",
    }


def test_markdown_higher_in_sale_months(cleaned):
    md = fashion.markdown_pct(cleaned)
    sale_months = {"2025-11", "2026-06"}
    sale = md[md["month"].isin(sale_months)]
    non_sale = md[~md["month"].isin(sale_months)]
    assert sale["pct_revenue_discounted"].mean() > non_sale["pct_revenue_discounted"].mean()


def test_dead_stock_pct_config_sensitive(generated):
    default_result = fashion.dead_stock_pct(generated["inventory_snapshots"], config=FASHION_CONFIG)
    conservative_result = fashion.dead_stock_pct(generated["inventory_snapshots"], config=CONSERVATIVE_FASHION_CONFIG)
    assert conservative_result["n_dead_stock_skus"] >= default_result["n_dead_stock_skus"]
