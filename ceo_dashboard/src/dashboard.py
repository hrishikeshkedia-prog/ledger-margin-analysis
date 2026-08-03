"""
Presentation layer (Stage 4) -- CONSUMES core.py / fashion.py output, never
computes or redefines a KPI.

Two kinds of function live here, and the line between them matters:

- `compute_*` functions do light ARITHMETIC ROLL-UPS of already-computed
  KPI tables (sum a column, take a ratio of two totals, pick out a
  headline number for a scorecard). They never touch raw order/inventory
  data and never introduce a new formula -- every number they produce
  traces back to a `core.py` or `fashion.py` function's output. This is
  what a scorecard inherently is: a summary of numbers computed elsewhere.
- `render_*` functions are pure plotly rendering: given already-computed
  data, produce a styled `go.Figure`. No aggregation, no business logic.

Every chart is a small, reusable, parameterized function (never a
one-off notebook cell) so Stage 7 can call the same functions to export a
standalone HTML dashboard.

Color system
------------
Colors are fixed hex values from a validated categorical/status palette
(CVD-safe, contrast-checked) -- not chosen ad hoc per chart. Status red is
RESERVED for risk (loss-making SKU/channel, dead stock) and never reused
as an ordinary series color; a fixed categorical order is used for
multi-series charts (channels, categories) so a given entity keeps the
same color across every chart it appears in.

Layout
------
`_finalize()` is the single place that sizes the bottom margin and places
the one-line (possibly wrapped) CEO-takeaway caption every chart ends
with. It computes the caption's pixel footprint from its wrapped line
count and reserves exactly that much margin, rather than a fixed guess --
which is what earlier drafts got wrong (captions collided with axis
titles/rotated tick labels at some figure heights and not others).
"""

from __future__ import annotations

import textwrap

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import core, fashion

# ---------------------------------------------------------------------------
# Color system (validated categorical + status palette; light-surface values)
# ---------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

# Fixed channel -> color assignment, so a channel keeps the same color on
# every chart it appears on (color follows the entity, never its rank).
CHANNEL_COLORS = {
    "Instagram Ads": CATEGORICAL[0],
    "Google Ads": CATEGORICAL[1],
    "Influencer": CATEGORICAL[2],
    "Affiliate": CATEGORICAL[3],
    "Organic/Email": CATEGORICAL[4],
}

_FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK["primary"], size=13)
_CAPTION_WRAP_WIDTH = 128     # characters per line, tuned for ~1150px figure width at 12px font
_CAPTION_LINE_PX = 17
_CAPTION_TOP_GAP_PX = 16      # gap between the axis baseline and the caption block
_MARGIN_TOP = 70
_MARGIN_LR = dict(l=70, r=40)


def _style_axes(fig):
    fig.update_xaxes(showgrid=False, showline=True, linecolor=BASELINE, ticks="outside",
                      tickcolor=BASELINE, tickfont=dict(color=INK["muted"]))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                      tickfont=dict(color=INK["muted"]))
    return fig


def _finalize(fig: go.Figure, title: str, caption: str, height: int, extra_bottom_px: int = 0) -> go.Figure:
    """Sets title, margins, and a wrapped caption whose vertical footprint
    is computed (not guessed) from its own wrapped line count, so the
    caption never collides with an x-axis title or rotated tick labels
    below it. `extra_bottom_px` reserves additional room for those when
    a chart has them (see call sites)."""
    _style_axes(fig)
    wrapped = textwrap.wrap(caption, width=_CAPTION_WRAP_WIDTH)
    n_lines = max(1, len(wrapped))
    tick_zone_px = 48 + extra_bottom_px  # axis line + tick marks + tick label text (+ rotated/2-line allowance)
    caption_block_px = _CAPTION_TOP_GAP_PX + n_lines * _CAPTION_LINE_PX
    margin_b = tick_zone_px + caption_block_px
    plot_h = max(height - _MARGIN_TOP - margin_b, 40)

    fig.update_layout(
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=_FONT, height=height,
        margin=dict(t=_MARGIN_TOP, b=margin_b, **_MARGIN_LR),
        title=dict(text=title, font=dict(size=16)),
    )
    # The caption's y must clear BOTH the tick-label zone AND its own gap --
    # not just the gap -- since paper y=0 sits at the axis baseline, and the
    # tick-label text lives in the pixels immediately below that, inside the
    # same margin region the caption also occupies further down.
    fig.add_annotation(
        text="<br>".join(wrapped), xref="paper", yref="paper", x=0,
        y=-((tick_zone_px + _CAPTION_TOP_GAP_PX) / plot_h),
        showarrow=False, align="left", xanchor="left", yanchor="top",
        font=dict(size=12, color=INK["secondary"]),
    )
    return fig


# ---------------------------------------------------------------------------
# 1. KPI scorecard
# ---------------------------------------------------------------------------

def compute_scorecard_metrics(cleaned_orders_df: pd.DataFrame, opex_df: pd.DataFrame,
                               marketing_df: pd.DataFrame, inventory_df: pd.DataFrame) -> dict:
    """Roll up headline numbers from existing KPI functions. Every value
    here is a sum/ratio of columns already returned by core.py/fashion.py
    -- no new formula, no raw-data logic (see module docstring).
    """
    revenue = core.revenue_trend(cleaned_orders_df)
    gross_margin = core.gross_margin_trend(cleaned_orders_df)
    contribution = core.contribution_margin_trend(cleaned_orders_df)
    bridge = core.returns_margin_bridge(cleaned_orders_df)
    ltv_cac = fashion.ltv_cac_ratio(cleaned_orders_df, marketing_df)
    dead_stock = fashion.dead_stock_pct(inventory_df)

    net_revenue = revenue["net_revenue"].sum()
    gross_margin_pct = gross_margin["gross_margin_abs"].sum() / gross_margin["net_revenue"].sum()
    contribution_margin_total = contribution["contribution_margin_abs"].sum()
    returns_pct_of_cm = -bridge["returns_margin_impact"].sum() / bridge["gross_contribution_margin"].sum()

    paid = ltv_cac[ltv_cac["total_spend"] > 0]
    blended_cac = paid["total_spend"].sum() / paid["total_new_customers"].sum()
    blended_ltv_cac = ltv_cac["ltv"].iloc[0] / blended_cac

    return {
        "net_revenue": net_revenue,
        "gross_margin_pct": gross_margin_pct,
        "contribution_margin_total": contribution_margin_total,
        "returns_pct_of_contribution_margin": returns_pct_of_cm,
        "blended_ltv_cac": blended_ltv_cac,
        "dead_stock_pct": dead_stock["dead_stock_pct_of_eligible"],
    }


def render_kpi_scorecard(metrics: dict) -> go.Figure:
    """Six headline numbers as a row of stat tiles on a blank canvas (plain
    annotations + shapes, not go.Indicator subplots). Red value text flags
    a metric that's a warning sign (high returns drag, weak LTV:CAC, dead
    stock); everything else uses primary ink.
    """
    tiles = [
        ("Net Revenue (12mo)", f"₹{metrics['net_revenue']/1e7:.2f} Cr", False),
        ("Gross Margin %", f"{metrics['gross_margin_pct']:.1%}", False),
        ("Contribution Margin (12mo)", f"₹{metrics['contribution_margin_total']/1e7:.2f} Cr", False),
        ("Returns Cost (% of Gross CM)", f"{metrics['returns_pct_of_contribution_margin']:.1%}",
         metrics["returns_pct_of_contribution_margin"] > 0.20),
        ("Blended LTV:CAC", f"{metrics['blended_ltv_cac']:.1f}x", metrics["blended_ltv_cac"] < 3),
        ("Dead-Stock %", f"{metrics['dead_stock_pct']:.1%}", metrics["dead_stock_pct"] > 0.05),
    ]
    n = len(tiles)
    fig = go.Figure()
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])

    tile_w = 1.0 / n
    for i, (label, value, is_warning) in enumerate(tiles):
        cx = tile_w * (i + 0.5)
        fig.add_shape(type="rect", x0=tile_w * i + tile_w * 0.06, x1=tile_w * (i + 1) - tile_w * 0.06,
                      y0=0.10, y1=0.95, line=dict(color=GRID, width=1), fillcolor=SURFACE, layer="below")
        fig.add_annotation(x=cx, y=0.72, text=label, showarrow=False,
                            font=dict(size=12, color=INK["secondary"]), xanchor="center")
        fig.add_annotation(x=cx, y=0.40, text=f"<b>{value}</b>", showarrow=False,
                            font=dict(size=25, color=STATUS["critical"] if is_warning else INK["primary"]),
                            xanchor="center")

    return _finalize(
        fig, "CEO Scorecard — D2C Fashion (synthetic data)",
        "Red values mark a headline warning sign (heavy returns drag, weak acquisition payback, or meaningful dead stock).",
        height=230,
    )


# ---------------------------------------------------------------------------
# 2. Revenue & margin trend (three single-axis small multiples -- never dual-axis)
# ---------------------------------------------------------------------------

def render_revenue_margin_trend(cleaned_orders_df: pd.DataFrame) -> go.Figure:
    """Three stacked panels, each with its own single y-axis (revenue,
    gross margin %, contribution margin per order) -- deliberately NOT a
    dual-axis combo chart, since revenue (currency) and margin % (ratio)
    are different units on different scales.
    """
    revenue = core.revenue_trend(cleaned_orders_df)
    gross_margin = core.gross_margin_trend(cleaned_orders_df)
    contribution = core.contribution_margin_trend(cleaned_orders_df)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                         subplot_titles=("Net Revenue", "Gross Margin %", "Contribution Margin per Order"))

    fig.add_trace(go.Bar(x=revenue["month"], y=revenue["net_revenue"], marker_color=CATEGORICAL[0],
                          name="Net Revenue", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=gross_margin["month"], y=gross_margin["gross_margin_pct"], mode="lines+markers",
                              line=dict(color=CATEGORICAL[2], width=2), marker=dict(size=8),
                              name="Gross Margin %", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=contribution["month"], y=contribution["contribution_margin_per_order"],
                              mode="lines+markers", line=dict(color=CATEGORICAL[6], width=2), marker=dict(size=8),
                              name="Contribution Margin / Order", showlegend=False), row=3, col=1)

    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=",.0f", row=3, col=1)

    return _finalize(
        fig, "Revenue & Margin Trend — 12 Months",
        "Revenue is growing steadily, but the two sale months (Nov, Jun) visibly compress gross margin % and "
        "contribution margin per order — growth is coming partly from margin-diluting discounting.",
        height=700,
    )


# ---------------------------------------------------------------------------
# 3. Returns-margin bridge (waterfall)
# ---------------------------------------------------------------------------

def render_returns_bridge(cleaned_orders_df: pd.DataFrame) -> go.Figure:
    """Waterfall from Gross Contribution Margin (as if nothing were ever
    returned) down to Net Contribution Margin (actual), full 12-month
    totals. Makes the returns-margin bridge (~29%) visually immediate.
    """
    bridge = core.returns_margin_bridge(cleaned_orders_df)
    gross = bridge["gross_contribution_margin"].sum()
    impact = bridge["returns_margin_impact"].sum()
    net = bridge["net_contribution_margin"].sum()

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Gross Contribution<br>Margin (no returns)", "Returns Margin<br>Impact", "Net Contribution<br>Margin (actual)"],
        y=[gross, impact, net],
        text=[f"₹{gross/1e7:.2f} Cr", f"−₹{-impact/1e7:.2f} Cr", f"₹{net/1e7:.2f} Cr"],
        textposition="outside",
        connector=dict(line=dict(color=BASELINE, width=1)),
        increasing=dict(marker_color=CATEGORICAL[0]),
        decreasing=dict(marker_color=STATUS["critical"]),
        totals=dict(marker_color=CATEGORICAL[0]),
    ))
    fig.update_yaxes(tickformat=",.0f")
    pct = -impact / gross

    return _finalize(
        fig, "Returns Margin Bridge — 12-Month Total",
        f"Returns alone erase {pct:.0%} of what contribution margin would have been with zero returns — the "
        "single clearest argument for treating returns as a first-class margin driver.",
        height=520, extra_bottom_px=25,  # two-line x tick labels ("Gross Contribution<br>Margin...")
    )


# ---------------------------------------------------------------------------
# 4. Margin by SKU (ranked, loss-makers in red)
# ---------------------------------------------------------------------------

def render_margin_by_sku(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame,
                          n: int = 12, total_sku_count: int | None = None) -> go.Figure:
    """Two side-by-side panels -- best N and worst N SKUs by contribution
    margin after CAC -- each with its own x-range, so a handful of
    high-volume winners (₹ hundreds of thousands) don't visually flatten
    a much smaller-magnitude loss-maker tail into illegible slivers on a
    single shared scale. Every bar in the "worst" panel is status red if
    loss-making, matching `fashion.sku_margin_after_cac`'s own flag.

    `total_sku_count`, if given (e.g. the full inventory catalog size),
    is used only for the caption's denominator -- `sku_margin_after_cac`
    itself only returns SKUs with at least one order line (see its
    docstring), so the raw table's length under-counts the full catalog.
    """
    sku_cac = fashion.sku_margin_after_cac(cleaned_orders_df, marketing_df)
    total = total_sku_count if total_sku_count is not None else len(sku_cac)
    top = sku_cac.sort_values("cm_after_cac", ascending=False).head(n).sort_values("cm_after_cac")
    bottom = sku_cac.sort_values("cm_after_cac", ascending=True).head(n).sort_values("cm_after_cac", ascending=False)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                         subplot_titles=(f"Best {n} SKUs", f"Worst {n} SKUs"))

    fig.add_trace(go.Bar(
        x=top["cm_after_cac"], y=top["sku_id"], orientation="h", marker_color=CATEGORICAL[0],
        text=[f"₹{v:,.0f}" for v in top["cm_after_cac"]], textposition="outside", cliponaxis=False,
        showlegend=False,
    ), row=1, col=1)

    bottom_colors = [STATUS["critical"] if v < 0 else CATEGORICAL[0] for v in bottom["cm_after_cac"]]
    fig.add_trace(go.Bar(
        x=bottom["cm_after_cac"], y=bottom["sku_id"], orientation="h", marker_color=bottom_colors,
        text=[f"₹{v:,.0f}" for v in bottom["cm_after_cac"]], textposition="outside", cliponaxis=False,
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(range=[0, top["cm_after_cac"].max() * 1.28], row=1, col=1)
    fig.update_xaxes(range=[bottom["cm_after_cac"].min() * 1.35, max(bottom["cm_after_cac"].max() * 1.35, 1)], row=1, col=2)
    fig.update_xaxes(tickformat=",.0f")

    n_loss = int(sku_cac["is_loss_making"].sum())
    return _finalize(
        fig, "Margin by SKU, after CAC",
        f"{n_loss} of {total} SKUs are loss-making after CAC (red, right panel) — a cluster of Tops SKUs with "
        "very high return rates dominate the worst offenders.",
        height=max(460, 30 * n),
    )


# ---------------------------------------------------------------------------
# 5. Channel view: CAC / payback, negative channel flagged
# ---------------------------------------------------------------------------

def render_channel_view(cleaned_orders_df: pd.DataFrame, marketing_df: pd.DataFrame) -> go.Figure:
    """Two panels sharing the fixed CHANNEL_COLORS assignment: True CAC by
    channel (left), and full-window contribution margin after CAC by
    channel (right) -- the definitive "does this channel pay for itself"
    number. A loss-making channel renders in status red on the right
    panel regardless of its CAC on the left. Units are named in each
    subplot title rather than a y-axis title, so nothing can bleed across
    the gap between panels.
    """
    cac_summary = fashion.cac_summary_by_channel(cleaned_orders_df, marketing_df)
    channel_pnl = fashion.channel_margin_after_cac(cleaned_orders_df, marketing_df)
    cac_summary = cac_summary[cac_summary["total_spend"] > 0]  # Organic/Email has no CAC to show

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                         subplot_titles=("True CAC (₹ per new customer)", "Contribution Margin after CAC (₹)"))

    fig.add_trace(go.Bar(
        x=cac_summary["channel"], y=cac_summary["true_cac"],
        marker_color=[CHANNEL_COLORS[c] for c in cac_summary["channel"]],
        text=[f"₹{v:,.0f}" for v in cac_summary["true_cac"]], textposition="outside",
        cliponaxis=False, showlegend=False,
    ), row=1, col=1)

    pnl_colors = [STATUS["critical"] if flag else CHANNEL_COLORS[c]
                  for c, flag in zip(channel_pnl["channel"], channel_pnl["is_loss_making"])]
    fig.add_trace(go.Bar(
        x=channel_pnl["channel"], y=channel_pnl["cm_after_cac"], marker_color=pnl_colors,
        text=[f"₹{v/1e5:.1f}L" for v in channel_pnl["cm_after_cac"]], textposition="outside",
        cliponaxis=False, showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(tickformat=",.0f", row=1, col=2, range=[channel_pnl["cm_after_cac"].min() * 1.25,
                                                              channel_pnl["cm_after_cac"].max() * 1.15])

    loss_making = channel_pnl.loc[channel_pnl["is_loss_making"], "channel"].tolist()
    caption = (f"{', '.join(loss_making)} has the highest CAC and is the only channel that doesn't earn back its "
               "own spend (red, right panel) — a clear case to cut or renegotiate." if loss_making
               else "Every channel currently earns back its own spend.")
    return _finalize(fig, "Channel Economics: CAC and Payback", caption, height=520)


# ---------------------------------------------------------------------------
# 6. Inventory: sell-through trend + dead-stock tail
# ---------------------------------------------------------------------------

def render_inventory_view(inventory_df: pd.DataFrame, n_tail: int = 15) -> go.Figure:
    """Left: company-wide monthly sell-through rate trend (single line,
    single axis). Right: the worst N SKUs by lifetime cumulative
    sell-through, with dead-stock-flagged SKUs (per `fashion.dead_stock_pct`)
    in status red. A true-zero SKU is given a small visible floor bar
    width (display only -- the printed value stays the real 0%) so its
    red flag is actually visible instead of a zero-length bar.
    """
    sell_through = fashion.sell_through_rate_monthly(inventory_df)
    company_monthly = sell_through.groupby("month").apply(
        lambda g: g["units_sold"].sum() / g["units_available"].sum(), include_groups=False
    ).rename("rate").reset_index()

    cum = fashion.cumulative_sell_through(inventory_df)
    dead_stock = fashion.dead_stock_pct(inventory_df)
    dead_ids = set(dead_stock["dead_stock_sku_ids"])
    worst = cum.sort_values("cumulative_sell_through").head(n_tail).sort_values("cumulative_sell_through", ascending=False)
    colors = [STATUS["critical"] if sku in dead_ids else CATEGORICAL[0] for sku in worst["sku_id"]]
    display_values = worst["cumulative_sell_through"].clip(lower=0.01)  # visible floor; label shows the real value

    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.12,
                         subplot_titles=("Company Sell-Through Rate (monthly)", f"Bottom {n_tail} SKUs — Cumulative Sell-Through"))

    fig.add_trace(go.Scatter(x=company_monthly["month"], y=company_monthly["rate"], mode="lines+markers",
                              line=dict(color=CATEGORICAL[0], width=2), marker=dict(size=8), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=display_values, y=worst["sku_id"], orientation="h",
                          marker_color=colors, text=[f"{v:.0%}" for v in worst["cumulative_sell_through"]],
                          textposition="outside", cliponaxis=False, showlegend=False), row=1, col=2)

    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_xaxes(tickformat=".0%", range=[0, 0.8], row=1, col=2)

    return _finalize(
        fig, "Inventory Health: Sell-Through and Dead Stock",
        f"{dead_stock['n_dead_stock_skus']} SKUs (red) are flagged dead stock — old enough to have proven "
        "themselves, but under 15% sold through everything they ever had in stock.",
        height=560,
    )


# ---------------------------------------------------------------------------
# 7. Cohort retention heatmap
# ---------------------------------------------------------------------------

def render_cohort_heatmap(cleaned_orders_df: pd.DataFrame) -> go.Figure:
    """Sequential-blue heatmap of the cohort retention table (light = low
    retention, dark = high) -- offset 0 is always 100% by construction and
    included for context, but the story is in how fast color fades left to
    right within each row. Every cohort-month row gets an explicit tick
    (plotly's default categorical tick spacing otherwise skips rows).
    """
    table = fashion.cohort_retention_table(cleaned_orders_df)
    fig = go.Figure(go.Heatmap(
        z=table.values, x=[f"+{c}mo" for c in table.columns], y=table.index,
        colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
        zmin=0, zmax=1, colorbar=dict(title="Retention", tickformat=".0%"),
        text=[[f"{v:.0%}" if pd.notna(v) else "" for v in row] for row in table.values],
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="Cohort %{y}, %{x}: %{z:.1%}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="Acquisition Cohort",
                      tickmode="array", tickvals=list(table.index), ticktext=list(table.index))
    fig.update_xaxes(title="Months Since Acquisition")

    avg_m1 = table[1].mean()
    return _finalize(
        fig, "Cohort Retention Heatmap",
        f"Retention drops sharply after month 0 (avg ~{avg_m1:.0%} by month 1) and keeps decaying — most of the "
        "customer base doesn't come back without a reason to.",
        height=520, extra_bottom_px=25,  # x-axis title below rotated-free but still-present tick labels
    )
