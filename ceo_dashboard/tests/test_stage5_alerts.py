"""Stage 5 tests: rules-based margin-risk alert engine."""

import numpy as np
import pandas as pd
import pytest

from ceo_dashboard.src import alerts, clean, data_generator as dg, fashion, schema


@pytest.fixture(scope="module")
def generated():
    return dg.generate_all(seed=42)


@pytest.fixture(scope="module")
def cleaned(generated):
    df, log = clean.clean_orders(generated["orders"])
    assert (log["rows_affected"] == 0).all()
    return df


# ---------------------------------------------------------------------------
# 1. Flagging matches the known, verified counts
# ---------------------------------------------------------------------------

def test_flags_known_28_skus_and_influencer(cleaned, generated):
    sku_causes = alerts.decompose_sku_causes(cleaned, generated["marketing_spend"])
    assert sku_causes["is_loss_making"].sum() == 28

    channel_causes = alerts.decompose_channel_causes(cleaned, generated["marketing_spend"])
    loss_channels = channel_causes.loc[channel_causes["is_loss_making"], "channel"].tolist()
    assert loss_channels == ["Influencer"]


def test_flags_never_recompute_fashion_functions(cleaned, generated):
    """decompose_* must read is_loss_making straight off the fashion.py
    functions, not derive it independently -- a regression here would mean
    the two disagree, which should never happen since one wraps the other."""
    mkt = generated["marketing_spend"]
    direct_sku = fashion.sku_margin_after_cac(cleaned, mkt).set_index(schema.SKU_ID)["is_loss_making"]
    decomposed_sku = alerts.decompose_sku_causes(cleaned, mkt).set_index(schema.SKU_ID)["is_loss_making"]
    pd.testing.assert_series_equal(direct_sku.sort_index(), decomposed_sku.sort_index(), check_names=False)

    direct_ch = fashion.channel_margin_after_cac(cleaned, mkt).set_index("channel")["is_loss_making"]
    decomposed_ch = alerts.decompose_channel_causes(cleaned, mkt).set_index("channel")["is_loss_making"]
    pd.testing.assert_series_equal(direct_ch.sort_index(), decomposed_ch.sort_index(), check_names=False)


# ---------------------------------------------------------------------------
# 2. Explain: decomposition drivers
# ---------------------------------------------------------------------------

def test_sku_drivers_are_nonnegative(cleaned, generated):
    sc = alerts.decompose_sku_causes(cleaned, generated["marketing_spend"])
    for col in ["returns_drag", "cac_drag", "thin_margin_drag", "fixed_cost_drag"]:
        assert (sc[col] >= 0).all()


def test_sku_primary_driver_is_named_for_every_loss_maker(cleaned, generated):
    sc = alerts.decompose_sku_causes(cleaned, generated["marketing_spend"])
    flagged = sc[sc["is_loss_making"]]
    assert (flagged["primary_driver"] != "none").all()
    assert set(flagged["primary_driver"]) <= {"returns", "cac", "thin_margin", "fixed_cost"}


def test_sku_primary_driver_matches_labels_dict(cleaned, generated):
    """Regression test for the label/column-name mismatch bug: every
    primary_driver value must have a real, non-fallback label."""
    sc = alerts.decompose_sku_causes(cleaned, generated["marketing_spend"])
    flagged = sc[sc["is_loss_making"]]
    labels = flagged["primary_driver"].map(alerts._CAUSE_LABELS)
    assert labels.notna().all()


def test_channel_drivers_are_nonnegative(cleaned, generated):
    cc = alerts.decompose_channel_causes(cleaned, generated["marketing_spend"])
    assert (cc["channel_cac_excess"].dropna() >= 0).all()
    assert (cc["channel_payback_shortfall"] >= 0).all()


def test_influencer_primary_driver_is_cac(cleaned, generated):
    cc = alerts.decompose_channel_causes(cleaned, generated["marketing_spend"])
    row = cc.set_index("channel").loc["Influencer"]
    assert row["primary_driver"] == "channel_cac"


# ---------------------------------------------------------------------------
# 3. Early warning
# ---------------------------------------------------------------------------

def test_early_warning_excludes_already_loss_making(cleaned, generated):
    mkt = generated["marketing_spend"]
    ew = alerts.detect_early_warning_skus(cleaned, mkt)
    loss_making_skus = set(fashion.sku_margin_after_cac(cleaned, mkt)
                            .loc[lambda d: d["is_loss_making"], schema.SKU_ID])
    assert not (set(ew[schema.SKU_ID]) & loss_making_skus)


def test_early_warning_requires_minimum_history(cleaned, generated):
    ew = alerts.detect_early_warning_skus(cleaned, generated["marketing_spend"])
    assert (ew["months_of_history"] >= alerts.ALERTS_CONFIG["min_months_data"]).all()


def test_early_warning_flag_reason_is_declining_or_projected(cleaned, generated):
    ew = alerts.detect_early_warning_skus(cleaned, generated["marketing_spend"])
    assert len(ew) > 0
    assert (ew["declining_streak"] | ew["projected_zero_cross"]).all()


def test_decline_streak_logic_hand_verified():
    """Hand-built series: needs streak_n+1 points, ALL strictly decreasing,
    to represent streak_n consecutive declining steps at the tail."""
    values = np.array([10, 5, 100, 90, 80, 60])  # last 4: 100 -> 90 -> 80 -> 60
    streak_n = 3
    recent = values[-(streak_n + 1):]
    declining = len(recent) == streak_n + 1 and all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
    assert declining

    # a non-monotonic tail (noise breaks the streak) must NOT trigger
    noisy = np.array([100, 90, 95, 60])
    recent_noisy = noisy[-(streak_n + 1):]
    declining_noisy = len(recent_noisy) == streak_n + 1 and all(
        recent_noisy[i] > recent_noisy[i + 1] for i in range(len(recent_noisy) - 1))
    assert not declining_noisy


# ---------------------------------------------------------------------------
# 4. Combined output table
# ---------------------------------------------------------------------------

def test_alerts_table_row_count_matches_flags(cleaned, generated):
    mkt = generated["marketing_spend"]
    table = alerts.build_alerts_table(cleaned, mkt)
    n_loss_sku = fashion.sku_margin_after_cac(cleaned, mkt)["is_loss_making"].sum()
    n_loss_channel = fashion.channel_margin_after_cac(cleaned, mkt)["is_loss_making"].sum()
    n_at_risk = len(alerts.detect_early_warning_skus(cleaned, mkt))
    assert len(table) == n_loss_sku + n_loss_channel + n_at_risk


def test_alerts_table_loss_making_ranked_most_negative_first(cleaned, generated):
    table = alerts.build_alerts_table(cleaned, generated["marketing_spend"])
    loss = table[table["severity"] == "Loss-Making"]["current_margin"]
    assert loss.is_monotonic_increasing


def test_alerts_table_every_row_has_action_and_reason(cleaned, generated):
    table = alerts.build_alerts_table(cleaned, generated["marketing_spend"])
    assert table["recommended_action"].notna().all()
    assert (table["reason"].str.len() > 0).all()


def test_alerts_table_severities_are_only_two_values(cleaned, generated):
    table = alerts.build_alerts_table(cleaned, generated["marketing_spend"])
    assert set(table["severity"]) <= {"Loss-Making", "At-Risk"}
