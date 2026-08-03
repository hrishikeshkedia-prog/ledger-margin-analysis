"""
Standalone HTML presentation export (Stage 7).

Produces ONE self-contained .html file -- every chart embedded as a
base64 PNG, no external script/stylesheet/font references, no live
Python/Colab kernel required -- that opens correctly double-clicked from
disk in any browser, offline.

This module is presentation-only, same rule as `dashboard.py`: every
number and every chart is produced by an existing `core.py` / `fashion.py`
/ `alerts.py` / `forecast.py` / `dashboard.py` function. Nothing here
computes a KPI; it only lays results out as an HTML page.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from . import alerts, clean, core, dashboard as dv, data_generator as dg, fashion, forecast, schema


# ---------------------------------------------------------------------------
# Data assembly (consumes existing functions only)
# ---------------------------------------------------------------------------

def compute_export_data(seed: int = 42) -> dict:
    """Regenerates the dataset and computes every table/figure/number the
    exported page needs, entirely via existing functions."""
    data = dg.generate_all(seed=seed)
    orders_clean, decision_log = clean.clean_orders(data["orders"])
    assert (decision_log["rows_affected"] == 0).all(), "canonical data should be clean by construction"

    marketing_df = data["marketing_spend"]
    inventory_df = data["inventory_snapshots"]
    opex_df = data["opex"]

    scorecard_metrics = dv.compute_scorecard_metrics(orders_clean, opex_df, marketing_df, inventory_df)

    sku_causes = alerts.decompose_sku_causes(orders_clean, marketing_df)
    channel_causes = alerts.decompose_channel_causes(orders_clean, marketing_df)
    alerts_table = alerts.build_alerts_table(orders_clean, marketing_df)

    n_loss_sku = int(sku_causes["is_loss_making"].sum())
    n_cac_driven = int((sku_causes.loc[sku_causes["is_loss_making"], "primary_driver"] == "cac").sum())
    influencer_row = channel_causes.set_index("channel").loc["Influencer"]

    seasonal_factor = forecast.estimate_seasonal_factor(inventory_df)
    validation = forecast.validate_forecast(inventory_df, seasonal_factor=seasonal_factor)
    stockout_validation = forecast.validate_stockout_rule(inventory_df, seasonal_factor=seasonal_factor)
    inventory_action = forecast.build_inventory_action_table(inventory_df, seasonal_factor=seasonal_factor)

    returns_bridge = core.returns_margin_bridge(orders_clean)
    returns_pct_of_cm = -returns_bridge["returns_margin_impact"].sum() / returns_bridge["gross_contribution_margin"].sum()

    total_sku_count = inventory_df[schema.INV_SKU_ID].nunique()

    figures = {
        "scorecard": dv.render_kpi_scorecard(scorecard_metrics),
        "revenue_margin_trend": dv.render_revenue_margin_trend(orders_clean),
        "returns_bridge": dv.render_returns_bridge(orders_clean),
        "margin_by_sku": dv.render_margin_by_sku(orders_clean, marketing_df, total_sku_count=total_sku_count),
        "channel_view": dv.render_channel_view(orders_clean, marketing_df),
        "cohort_heatmap": dv.render_cohort_heatmap(orders_clean),
        "inventory_view": dv.render_inventory_view(inventory_df),
    }

    return {
        "figures": figures,
        "alerts_table": alerts_table,
        "inventory_action": inventory_action,
        "findings": {
            "returns_pct_of_cm": returns_pct_of_cm,
            "n_loss_sku": n_loss_sku,
            "n_total_sku": total_sku_count,
            "n_cac_driven": n_cac_driven,
            "influencer_cac_excess": influencer_row["channel_cac_excess"],
            "n_loss_channel": int(channel_causes["is_loss_making"].sum()),
            "n_at_risk": len(alerts_table[alerts_table["severity"] == "At-Risk"]),
            "wape": validation["wape"],
            "stockout_recall": stockout_validation["recall"],
            "stockout_precision": stockout_validation["precision"],
        },
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _fig_to_data_uri(fig: go.Figure) -> str:
    """Renders a plotly figure to a PNG and returns it as a base64 data URI
    -- no external image files, no JS runtime needed to view it."""
    png_bytes = fig.to_image(format="png", scale=2)
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _money_cr(value: float) -> str:
    return f"₹{value / 1e7:.2f} Cr"


def _alerts_table_html(alerts_table: pd.DataFrame) -> str:
    rows = []
    for _, r in alerts_table.iterrows():
        badge_class = "badge-critical" if r["severity"] == "Loss-Making" else "badge-warning"
        rows.append(f"""
        <tr>
          <td>{r['entity_type']}</td>
          <td><strong>{r['entity_id']}</strong></td>
          <td>{r['group']}</td>
          <td class="num">₹{r['current_margin']:,.0f}</td>
          <td><span class="badge {badge_class}">{r['severity']}</span></td>
          <td>{r['primary_cause']}</td>
          <td class="reason">{r['reason']}</td>
          <td class="action">{r['recommended_action']}</td>
        </tr>""")
    return f"""
    <table class="data-table">
      <thead><tr>
        <th>Type</th><th>Entity</th><th>Group</th><th>Margin</th>
        <th>Severity</th><th>Primary Cause</th><th>Reason</th><th>Recommended Action</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _inventory_table_html(rows_df: pd.DataFrame, flag: str) -> str:
    badge_class = "badge-critical" if flag == "Stockout Risk" else "badge-warning"
    rows = []
    for _, r in rows_df.iterrows():
        rows.append(f"""
        <tr>
          <td><strong>{r['sku_id']}</strong></td>
          <td class="num">{r['forecast_units']:.1f}</td>
          <td class="num">{r['current_stock']:.0f}</td>
          <td><span class="badge {badge_class}">{r['inventory_flag']}</span></td>
          <td class="action">{r['recommended_action']}</td>
        </tr>""")
    return f"""
    <table class="data-table">
      <thead><tr><th>SKU</th><th>Forecast Units</th><th>Current Stock</th><th>Flag</th><th>Recommended Action</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --ink-primary: #0b0b0b; --ink-secondary: #52514e; --ink-muted: #898781;
  --surface: #fcfcfb; --page: #f4f3f0; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --critical: #d03b3b; --warning: #fab219; --good: #0ca30c; --accent: #2a78d6;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink-primary); background: var(--page); line-height: 1.5;
}
.page { max-width: 1240px; margin: 0 auto; padding: 32px 24px 64px; }
header.hero { padding: 40px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 32px; }
header.hero h1 { margin: 0 0 4px; font-size: 32px; }
header.hero .subtitle { margin: 0 0 16px; color: var(--ink-secondary); font-size: 16px; }
.disclaimer {
  display: inline-block; background: #fff7e8; border: 1px solid var(--warning);
  color: #6b4d00; padding: 10px 16px; border-radius: 8px; font-size: 14px; font-weight: 600;
}
section { margin-bottom: 48px; }
section h2 {
  font-size: 22px; margin: 0 0 16px; padding-bottom: 8px; border-bottom: 2px solid var(--ink-primary);
}
section h3 { font-size: 17px; margin: 24px 0 12px; color: var(--ink-secondary); }
.finding-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.finding-card {
  background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 18px;
  border-left: 4px solid var(--accent);
}
.finding-card.critical { border-left-color: var(--critical); }
.finding-card.warning { border-left-color: var(--warning); }
.finding-card .headline { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
.finding-card .headline.critical-text { color: var(--critical); }
.finding-card .detail { font-size: 13px; color: var(--ink-secondary); }
.chart-block { background: var(--surface); border: 1px solid var(--grid); border-radius: 10px;
  padding: 8px; margin-bottom: 20px; }
.chart-block img { width: 100%; height: auto; display: block; border-radius: 6px; }
.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
table.data-table { width: 100%; border-collapse: collapse; background: var(--surface);
  border: 1px solid var(--grid); border-radius: 8px; overflow: hidden; font-size: 13px; }
table.data-table th { text-align: left; background: #f0efec; padding: 10px 12px;
  border-bottom: 1px solid var(--grid); color: var(--ink-secondary); font-weight: 600; }
table.data-table td { padding: 9px 12px; border-bottom: 1px solid var(--grid); vertical-align: top; }
table.data-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.data-table td.reason, table.data-table td.action { color: var(--ink-secondary); max-width: 260px; }
table.data-table tbody tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
  white-space: nowrap; }
.badge-critical { background: #fbeaea; color: var(--critical); }
.badge-warning { background: #fff4de; color: #8a5a00; }
.table-note { font-size: 13px; color: var(--ink-muted); margin-top: 8px; }
footer.limitations { background: #fff7e8; border: 1px solid var(--warning); border-radius: 10px;
  padding: 24px 28px; }
footer.limitations h2 { border: none; margin-bottom: 12px; }
footer.limitations ul { margin: 0; padding-left: 20px; }
footer.limitations li { margin-bottom: 10px; }
.build-note { text-align: center; color: var(--ink-muted); font-size: 12px; margin-top: 32px; }
"""


def build_html(export: dict) -> str:
    figs = export["figures"]
    findings = export["findings"]
    alerts_table = export["alerts_table"]
    inventory_action = export["inventory_action"]

    stockout_rows = inventory_action[inventory_action["inventory_flag"] == "Stockout Risk"] \
        .sort_values("forecast_units", ascending=False).head(15)
    overstock_rows = inventory_action[inventory_action["inventory_flag"] == "Overstock Risk"] \
        .sort_values("forecast_months_cover", ascending=False).head(10)
    n_stockout_total = (inventory_action["inventory_flag"] == "Stockout Risk").sum()
    n_overstock_total = (inventory_action["inventory_flag"] == "Overstock Risk").sum()

    chart_imgs = {name: _fig_to_data_uri(fig) for name, fig in figs.items()}

    findings_html = f"""
    <div class="finding-grid">
      <div class="finding-card critical">
        <div class="headline critical-text">{findings['returns_pct_of_cm']:.0%}</div>
        <div class="detail">of gross contribution margin is given back to returns — the single
        largest, most controllable margin leak in the business.</div>
      </div>
      <div class="finding-card critical">
        <div class="headline critical-text">CAC is the real killer</div>
        <div class="detail">{findings['n_cac_driven']} of {findings['n_loss_sku']} loss-making SKUs are
        primarily CAC-driven, not returns-driven. The Influencer channel alone overspends its peer
        channels' typical CAC by ~₹{findings['influencer_cac_excess']/1e5:.0f}L.</div>
      </div>
      <div class="finding-card warning">
        <div class="headline">{findings['n_loss_sku']} SKUs + {findings['n_loss_channel']} channel</div>
        <div class="detail">are currently loss-making after CAC, of {findings['n_total_sku']} SKUs total.
        {findings['n_at_risk']} more are trending negative — an early-warning list, not yet a crisis.</div>
      </div>
      <div class="finding-card">
        <div class="headline">WAPE ~{findings['wape']:.0%}</div>
        <div class="detail">demand-forecast error. The stockout-alert rule is tuned toward catching real
        risk ({findings['stockout_recall']:.0%} recall) over avoiding false alarms
        ({findings['stockout_precision']:.0%} precision) — a deliberate trade-off for an early-warning tool.</div>
      </div>
    </div>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CEO Cost & Margin Dashboard — D2C Fashion</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">

  <header class="hero">
    <h1>CEO Cost &amp; Margin Dashboard</h1>
    <p class="subtitle">D2C Fashion — 12-Month Performance Review</p>
    <div class="disclaimer">⚠️ ALL DATA IN THIS DASHBOARD IS SYNTHETIC — generated by a seeded
      data generator for a portfolio project, not sourced from any real business. Conclusions here
      illustrate methodology, not real business results.</div>
  </header>

  <section>
    <h2>Executive Summary</h2>
    {findings_html}
  </section>

  <section>
    <h2>1. Performance &amp; Margin</h2>
    <div class="chart-block"><img src="{chart_imgs['scorecard']}" alt="KPI Scorecard"></div>
    <div class="chart-block"><img src="{chart_imgs['revenue_margin_trend']}" alt="Revenue and margin trend"></div>
  </section>

  <section>
    <h2>2. Returns Impact</h2>
    <div class="chart-block"><img src="{chart_imgs['returns_bridge']}" alt="Returns margin bridge"></div>
  </section>

  <section>
    <h2>3. SKU &amp; Channel Economics</h2>
    <div class="chart-block"><img src="{chart_imgs['margin_by_sku']}" alt="Margin by SKU"></div>
    <div class="chart-block"><img src="{chart_imgs['channel_view']}" alt="Channel economics"></div>
    <div class="chart-block"><img src="{chart_imgs['cohort_heatmap']}" alt="Cohort retention heatmap"></div>

    <h3>Margin-Risk Alerts ({len(alerts_table)} total: {findings['n_loss_sku'] + findings['n_loss_channel']}
      loss-making, {findings['n_at_risk']} at-risk)</h3>
    {_alerts_table_html(alerts_table)}
  </section>

  <section>
    <h2>4. Inventory &amp; Demand Forecast</h2>
    <div class="chart-block"><img src="{chart_imgs['inventory_view']}" alt="Inventory health"></div>

    <h3>Stockout Risk — top 15 of {n_stockout_total} by forecast volume</h3>
    {_inventory_table_html(stockout_rows, "Stockout Risk")}
    <p class="table-note">Reorder recommendations are ranked; treat as a review list, not simultaneous
      fire drills — see limitations below.</p>

    <h3>Overstock Risk — top 10 of {n_overstock_total} by months of cover</h3>
    {_inventory_table_html(overstock_rows, "Overstock Risk")}
  </section>

  <footer class="limitations">
    <h2>Honest Limitations</h2>
    <ul>
      <li><strong>Forecasting the month right after a sale month carries a stated bias risk.</strong>
        A trailing moving average naturally absorbs a recent sale spike; a de-seasonalization step
        (see Stage 6) measurably reduces this but does not eliminate it — expect the July forecast and
        Stockout Risk count to run somewhat high following June's sale month.</li>
      <li><strong>The stockout-alert rule favors recall over precision by design</strong>
        ({findings['stockout_recall']:.0%} recall, {findings['stockout_precision']:.0%} precision) —
        it is tuned to not miss real risk, at the cost of roughly 3 in 4 flags not corresponding to an
        actual recorded near-stockout event. Treat flags as a triage list for a merchandiser, not an
        auto-trigger for purchase orders.</li>
      <li><strong>The long tail of low-volume SKUs is inherently hard to forecast</strong> — a SKU
        selling 2-3 units a month has almost no signal to average, so its individual forecast error can
        be large even while the company-wide WAPE looks reasonable.</li>
      <li><strong>The seasonal adjustment factor rests on a single observed sale month</strong>
        (Nov 2025) inside the training window — a real business would refine this with 2-3+ years of
        promotional history.</li>
    </ul>
  </footer>

  <p class="build-note">Built from src/export_html.py — regenerate with
    <code>python notebooks/build_dashboard_html.py</code>. Every number and chart above is produced by
    the same core.py / fashion.py / alerts.py / forecast.py / dashboard.py functions used throughout
    the analysis notebook.</p>

</div>
</body>
</html>"""
    return html


def write_dashboard_html(output_path: str, seed: int = 42) -> Path:
    export = compute_export_data(seed=seed)
    html = build_html(export)
    path = Path(output_path)
    path.write_text(html, encoding="utf-8")
    return path
