"""Stage 4 tests: dashboard module renders valid figures and consumes
(never redefines) existing core.py/fashion.py KPI output."""

import plotly.graph_objects as go
import pytest

from ceo_dashboard.src import clean, dashboard as dv, data_generator as dg, fashion


@pytest.fixture(scope="module")
def generated():
    return dg.generate_all(seed=42)


@pytest.fixture(scope="module")
def cleaned(generated):
    df, log = clean.clean_orders(generated["orders"])
    assert (log["rows_affected"] == 0).all()
    return df


def test_scorecard_metrics_trace_to_existing_functions(cleaned, generated):
    """Every scorecard number must reconcile with the KPI function it rolls up."""
    metrics = dv.compute_scorecard_metrics(cleaned, generated["opex"], generated["marketing_spend"], generated["inventory_snapshots"])
    assert metrics["dead_stock_pct"] == fashion.dead_stock_pct(generated["inventory_snapshots"])["dead_stock_pct_of_eligible"]
    assert 0 < metrics["gross_margin_pct"] < 1
    assert metrics["net_revenue"] > 0
    assert 0 < metrics["returns_pct_of_contribution_margin"] < 1


@pytest.mark.parametrize("fn_name,args", [
    ("render_revenue_margin_trend", ("cleaned",)),
    ("render_returns_bridge", ("cleaned",)),
    ("render_margin_by_sku", ("cleaned", "marketing")),
    ("render_channel_view", ("cleaned", "marketing")),
    ("render_inventory_view", ("inventory",)),
    ("render_cohort_heatmap", ("cleaned",)),
])
def test_render_functions_return_valid_figure(fn_name, args, cleaned, generated):
    ctx = {"cleaned": cleaned, "marketing": generated["marketing_spend"], "inventory": generated["inventory_snapshots"]}
    fn = getattr(dv, fn_name)
    fig = fn(*[ctx[a] for a in args])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.layout.title.text  # every chart has a title
    # every chart has a CEO-takeaway caption annotation
    assert any(a.y < 0 for a in fig.layout.annotations)


def test_scorecard_figure_valid(cleaned, generated):
    metrics = dv.compute_scorecard_metrics(cleaned, generated["opex"], generated["marketing_spend"], generated["inventory_snapshots"])
    fig = dv.render_kpi_scorecard(metrics)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) == 6  # six stat tiles


def test_margin_by_sku_denominator_uses_full_catalog(cleaned, generated):
    total = generated["inventory_snapshots"]["sku_id"].nunique()
    fig = dv.render_margin_by_sku(cleaned, generated["marketing_spend"], total_sku_count=total)
    caption = next(a.text for a in fig.layout.annotations if "loss-making" in a.text)
    assert f"of {total} SKUs" in caption


def test_channel_colors_are_fixed_and_distinct():
    assert len(set(dv.CHANNEL_COLORS.values())) == len(dv.CHANNEL_COLORS)


def test_status_critical_reserved_for_risk():
    """Status red must never appear in the ordinary categorical palette --
    it's reserved so a risk color can never be mistaken for a routine series."""
    assert dv.STATUS["critical"] not in dv.CATEGORICAL


def test_inventory_view_marks_known_dead_stock_red(generated):
    dead_stock = fashion.dead_stock_pct(generated["inventory_snapshots"])
    fig = dv.render_inventory_view(generated["inventory_snapshots"], n_tail=15)
    bar_trace = fig.data[1]
    dead_ids = set(dead_stock["dead_stock_sku_ids"])
    for sku_id, color in zip(bar_trace.y, bar_trace.marker.color):
        expected = dv.STATUS["critical"] if sku_id in dead_ids else dv.CATEGORICAL[0]
        assert color == expected


def test_channel_view_marks_influencer_red(cleaned, generated):
    fig = dv.render_channel_view(cleaned, generated["marketing_spend"])
    pnl_trace = fig.data[1]
    idx = list(pnl_trace.x).index("Influencer")
    assert pnl_trace.marker.color[idx] == dv.STATUS["critical"]
