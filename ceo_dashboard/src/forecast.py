"""
Demand / inventory forecast (Stage 6) -- the one genuinely predictive
model in this project.

Why this stage (and only this stage) gets a real model: Stage 5's
loss-makers were deliberately seeded in `data_generator.py` (a planted
answer key), so "predicting" them with a trained model would be circular.
Demand, by contrast, is an EMERGENT outcome of many seeded ingredients
interacting (Pareto popularity weights, seasonal multipliers, random
noise, category mix) -- nothing in the generator writes "SKU X sells N
units in month M" directly. Forecasting it is a legitimate prediction
problem with a real, checkable right answer, which is exactly why this
module holds itself to the standard the others didn't need to: an
honest, out-of-sample validation with reported error, not just a claim.

Method: a 3-month trailing moving average, with a single multiplicative
seasonal adjustment for known promotional months. Two sentences, as
required:
1. Forecast(SKU, month) = mean(that SKU's actual units sold in the 3
   months immediately before `month`) x a seasonal multiplier that's 1.0
   for an ordinary month and an empirically-estimated factor (>1) for a
   month on the business's known promotional calendar.
2. The seasonal factor itself is estimated from TRAINING data only (the
   ratio of average company-wide units sold in past promotional months
   vs. ordinary months), never from the held-out validation months.

No black box: every forecast is "the recent average, bumped up if this
month is a known sale month" -- nothing else is happening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

FORECAST_CONFIG = {
    # Months of trailing actual sales averaged for the moving-average
    # forecast. 3 balances responsiveness (a shorter window reacts faster
    # to a real shift) against noise (a 1-2 month window is dominated by
    # random month-to-month variance, especially for low-volume SKUs).
    "ma_window_months": 3,

    # A SKU needs at least this many months of prior sales before it gets
    # a forecast at all -- fewer points make "the average" statistically
    # meaningless, not a real signal. Equal to ma_window_months by
    # construction (can't average 3 months with fewer than 3 data points).
    "min_history_months": 3,

    # Trailing months held out for honest validation -- NEVER used to
    # estimate the seasonal factor or anything else the forecast depends
    # on. 2 months gives one ordinary month and one sale month in this
    # 12-month dataset, which is exactly the harder, more informative case.
    "holdout_months": 2,

    # The business's own promotional calendar -- a real merchandising
    # input a real forecasting team would have in advance (they planned
    # the sale), not something inferred from data. Matches this dataset's
    # actual sale months, generated in Stage 1.
    "known_sale_months": ["2025-11", "2026-06"],

    # A SKU whose current stock covers more than this many months of
    # forecast demand, at the forecast pace, is flagged overstock risk.
    "overstock_months_cover_threshold": 3.0,

    # Forecast demand x this safety margin is compared against current
    # stock for the stockout flag. 1.0 = flag as soon as forecast demand
    # alone would exceed what's on hand, no extra buffer.
    "stockout_safety_margin": 1.0,
}


# ---------------------------------------------------------------------------
# 1. Forecast
# ---------------------------------------------------------------------------

def _dataset_months(inventory_df: pd.DataFrame) -> list[str]:
    return sorted(inventory_df[schema.INV_MONTH].unique())


def train_holdout_split(inventory_df: pd.DataFrame, config: dict = FORECAST_CONFIG) -> tuple[list[str], list[str]]:
    """Splits the dataset's months into (train_months, holdout_months) --
    the trailing `config['holdout_months']` months are held out."""
    months = _dataset_months(inventory_df)
    n_holdout = config["holdout_months"]
    return months[:-n_holdout], months[-n_holdout:]


def estimate_seasonal_factor(inventory_df: pd.DataFrame, config: dict = FORECAST_CONFIG) -> float:
    """Estimates a single company-wide seasonal multiplier from TRAINING
    months only (the holdout is never touched here).

    Formula: (average company-wide units sold, across training months on
    the known sale-month calendar) / (average company-wide units sold,
    across training months NOT on it). Returns 1.0 (no adjustment) if no
    training month happens to be a sale month.

    Limitation, stated plainly: this dataset has only one sale month
    inside the training window (Nov 2025 -- June 2026 falls in the
    holdout), so the factor is a single-observation estimate, not an
    average over several sale events. A real business would refine this
    once it has 2-3+ years of promotional-calendar history.
    """
    train_months, _ = train_holdout_split(inventory_df, config)
    company_monthly = inventory_df[inventory_df[schema.INV_MONTH].isin(train_months)].groupby(schema.INV_MONTH)[schema.INV_SOLD].sum()
    sale_months = [m for m in train_months if m in config["known_sale_months"]]
    non_sale_months = [m for m in train_months if m not in config["known_sale_months"]]
    if not sale_months or not non_sale_months:
        return 1.0
    return company_monthly.loc[sale_months].mean() / company_monthly.loc[non_sale_months].mean()


def _deseasonalized_units(df: pd.DataFrame, config: dict, seasonal_factor: float) -> pd.Series:
    """Puts every month's actual units_sold on a common "ordinary month"
    footing by dividing out the seasonal bump from any month that was
    itself a known sale month. Used only as the INPUT to the trailing
    average below -- without this, forecasting the month right after a
    sale month averages in the sale spike with no corresponding drop
    correction, inflating the forecast exactly when stock is simultaneously
    depleted from the sale. (Caught during development: this exact "day
    after the sale" case produced 105 of 150 SKUs flagged stockout-risk in
    a first pass, an unusable/implausible fraction that a sanity check --
    inspecting a flagged SKU's forecast and stock side by side -- traced
    straight to this bias.)
    """
    return np.where(df[schema.INV_MONTH].isin(config["known_sale_months"]),
                     df[schema.INV_SOLD] / seasonal_factor, df[schema.INV_SOLD])


def walk_forward_forecast(inventory_df: pd.DataFrame, config: dict = FORECAST_CONFIG,
                           seasonal_factor: float | None = None) -> pd.DataFrame:
    """One-step-ahead forecast for every (SKU, month) with enough prior
    history, using ONLY actual data from months strictly before it -- a
    proper walk-forward backtest, never peeking at the month being
    forecast. This is what both the validation step and the stockout-rule
    backtest are computed against.

    Formula: forecast(SKU, month) = mean(that SKU's DE-SEASONALIZED actual
    units_sold -- see `_deseasonalized_units` -- in the `ma_window_months`
    months immediately preceding `month`) x (seasonal_factor if `month`
    itself is a known sale month, else 1.0).
    """
    if seasonal_factor is None:
        seasonal_factor = estimate_seasonal_factor(inventory_df, config)

    df = inventory_df.sort_values([schema.INV_SKU_ID, schema.INV_MONTH]).copy()
    df["units_sold_deseasonalized"] = _deseasonalized_units(df, config, seasonal_factor)
    window = config["ma_window_months"]

    def _rolling_prior_mean(g: pd.DataFrame) -> pd.Series:
        return g["units_sold_deseasonalized"].shift(1).rolling(window, min_periods=window).mean()

    df["forecast_raw"] = df.groupby(schema.INV_SKU_ID, group_keys=False).apply(
        _rolling_prior_mean, include_groups=False
    )
    df["seasonal_multiplier"] = np.where(df[schema.INV_MONTH].isin(config["known_sale_months"]), seasonal_factor, 1.0)
    df["forecast_units"] = df["forecast_raw"] * df["seasonal_multiplier"]

    out = df.dropna(subset=["forecast_units"])
    return out[[schema.INV_SKU_ID, schema.INV_MONTH, "forecast_units", schema.INV_SOLD,
                schema.INV_BEGIN, schema.INV_END, schema.INV_NEAR_STOCKOUT]].reset_index(drop=True)


def forecast_next_period(inventory_df: pd.DataFrame, config: dict = FORECAST_CONFIG,
                          seasonal_factor: float | None = None) -> pd.DataFrame:
    """Forecasts the month immediately AFTER the dataset's last month, for
    every SKU with at least `ma_window_months` of trailing sales history.
    This is the forward-looking forecast the inventory-action table acts
    on (distinct from `walk_forward_forecast`, which re-forecasts past
    months purely for validation).

    Formula: mean of that SKU's last `ma_window_months` months of
    DE-SEASONALIZED actual units_sold (see `_deseasonalized_units`), x the
    seasonal multiplier if the upcoming month is on the known sale-month
    calendar (by default it isn't -- stated explicitly, not silently
    assumed).
    """
    if seasonal_factor is None:
        seasonal_factor = estimate_seasonal_factor(inventory_df, config)

    months = _dataset_months(inventory_df)
    next_month = str(pd.Period(months[-1], freq="M") + 1)
    window = config["ma_window_months"]
    inventory_df = inventory_df.copy()
    inventory_df["units_sold_deseasonalized"] = _deseasonalized_units(inventory_df, config, seasonal_factor)

    def _last_n_mean(g: pd.DataFrame) -> float:
        tail = g.sort_values(schema.INV_MONTH)["units_sold_deseasonalized"].tail(window)
        return tail.mean() if len(tail) == window else np.nan

    forecasts = (inventory_df.groupby(schema.INV_SKU_ID)
                 .apply(_last_n_mean, include_groups=False)
                 .rename("forecast_units_raw").reset_index())
    multiplier = seasonal_factor if next_month in config["known_sale_months"] else 1.0
    forecasts["forecast_units"] = forecasts["forecast_units_raw"] * multiplier
    forecasts["forecast_month"] = next_month
    forecasts = forecasts.dropna(subset=["forecast_units"])

    latest_stock = (inventory_df.sort_values(schema.INV_MONTH)
                     .groupby(schema.INV_SKU_ID).tail(1)
                     .set_index(schema.INV_SKU_ID)[schema.INV_END])
    forecasts["current_stock"] = forecasts[schema.INV_SKU_ID].map(latest_stock)
    return forecasts[[schema.INV_SKU_ID, "forecast_month", "forecast_units", "current_stock"]]


# ---------------------------------------------------------------------------
# 2. Validate honestly
# ---------------------------------------------------------------------------

def validate_forecast(inventory_df: pd.DataFrame, config: dict = FORECAST_CONFIG,
                       seasonal_factor: float | None = None) -> dict:
    """Out-of-sample validation against the trailing `config['holdout_months']`
    months -- never used to fit the seasonal factor or anything else.

    Headline metric: WAPE (Weighted Absolute Percentage Error) =
    sum(|forecast - actual|) / sum(actual), aggregated across every SKU-
    month in the holdout. WAPE, not per-SKU MAPE, is the headline because
    plain MAPE is undefined (divide by zero) or explodes for any SKU-month
    with zero actual units -- and this dataset's long tail of low-volume
    SKUs has plenty of those. WAPE sidesteps that by aggregating the
    numerator and denominator separately before dividing.

    Also returns: aggregate MAE (mean absolute error, in units); a
    per-SKU MAE table for the holdout; and a secondary MAPE computed only
    over rows with actual > 0 (explicitly caveated -- excludes exactly the
    SKU-months where MAPE would be meaningless).
    """
    if seasonal_factor is None:
        seasonal_factor = estimate_seasonal_factor(inventory_df, config)
    _, holdout_months = train_holdout_split(inventory_df, config)

    wf = walk_forward_forecast(inventory_df, config, seasonal_factor)
    holdout = wf[wf[schema.INV_MONTH].isin(holdout_months)].copy()
    holdout["error"] = holdout["forecast_units"] - holdout[schema.INV_SOLD]
    holdout["abs_error"] = holdout["error"].abs()

    wape = holdout["abs_error"].sum() / holdout[schema.INV_SOLD].sum()
    mae = holdout["abs_error"].mean()

    nonzero = holdout[holdout[schema.INV_SOLD] > 0]
    mape_nonzero = (nonzero["abs_error"] / nonzero[schema.INV_SOLD]).mean()

    per_sku = holdout.groupby(schema.INV_SKU_ID).agg(
        mae=("abs_error", "mean"),
        total_actual=(schema.INV_SOLD, "sum"),
        total_forecast=("forecast_units", "sum"),
        months_evaluated=(schema.INV_MONTH, "count"),
        had_zero_actual_month=(schema.INV_SOLD, lambda s: (s == 0).any()),
    ).reset_index().sort_values("mae", ascending=False)

    return {
        "holdout_months": holdout_months,
        "seasonal_factor_used": seasonal_factor,
        "n_sku_months_evaluated": len(holdout),
        "wape": wape,
        "mae": mae,
        "mape_nonzero_only": mape_nonzero,
        "n_rows_excluded_from_mape": len(holdout) - len(nonzero),
        "per_sku": per_sku,
    }


# ---------------------------------------------------------------------------
# 3. Inventory action + stockout-rule validation
# ---------------------------------------------------------------------------

def build_inventory_action_table(inventory_df: pd.DataFrame, config: dict = FORECAST_CONFIG,
                                  seasonal_factor: float | None = None) -> pd.DataFrame:
    """Converts the forward-looking forecast into a per-SKU action:

    - **Stockout Risk**: forecast_units x `stockout_safety_margin` > current
      stock on hand -- not enough inventory to cover forecast demand.
    - **Overstock Risk**: current stock covers more than
      `overstock_months_cover_threshold` months at the forecast pace
      (current_stock / forecast_units).
    - **Healthy**: neither.

    Business question: for each SKU, should we reorder now, mark it down
    now, or do nothing?
    """
    forecasts = forecast_next_period(inventory_df, config, seasonal_factor)
    forecasts["forecast_months_cover"] = np.where(
        forecasts["forecast_units"] > 0, forecasts["current_stock"] / forecasts["forecast_units"],
        np.where(forecasts["current_stock"] > 0, np.inf, 0.0),
    )

    def _flag(row):
        if row["forecast_units"] * config["stockout_safety_margin"] > row["current_stock"]:
            return "Stockout Risk"
        if row["forecast_months_cover"] > config["overstock_months_cover_threshold"]:
            return "Overstock Risk"
        return "Healthy"

    forecasts["inventory_flag"] = forecasts.apply(_flag, axis=1)

    def _action(row):
        if row["inventory_flag"] == "Stockout Risk":
            reorder_qty = max(0, round(row["forecast_units"] * config["stockout_safety_margin"] - row["current_stock"]))
            return f"Reorder ~{reorder_qty} units before {row['forecast_month']}"
        if row["inventory_flag"] == "Overstock Risk":
            return f"Markdown/clearance -- {row['forecast_months_cover']:.1f} months of cover at forecast pace"
        return "No action needed"

    forecasts["recommended_action"] = forecasts.apply(_action, axis=1)
    return forecasts.sort_values(["inventory_flag", "forecast_units"], ascending=[True, False]).reset_index(drop=True)


def validate_stockout_rule(inventory_df: pd.DataFrame, config: dict = FORECAST_CONFIG,
                            seasonal_factor: float | None = None) -> dict:
    """Backtests the stockout rule (forecast demand x safety margin >
    stock on hand at the start of the month) against every recorded
    `near_stockout_flag` event, using the full walk-forward forecast (not
    just the 2-month holdout, since only 152 positive events exist total
    and a larger evaluation window gives a more statistically meaningful
    precision/recall).

    Reports a standard confusion-matrix precision/recall/counts -- no
    model, just counting how often the rule's flag agrees with the
    recorded ground truth. Also reports how many of the 152 actual events
    fall OUTSIDE the evaluable set (a SKU's first `min_history_months`
    months never get a forecast at all, so an early near-stockout event
    for a brand-new SKU is structurally unreachable by this rule -- a
    real, stated limitation, not hidden in the aggregate score).
    """
    wf = walk_forward_forecast(inventory_df, config, seasonal_factor)
    wf["predicted_stockout"] = wf["forecast_units"] * config["stockout_safety_margin"] > wf[schema.INV_BEGIN]
    actual = wf[schema.INV_NEAR_STOCKOUT]
    predicted = wf["predicted_stockout"]

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    total_actual_events = int(inventory_df[schema.INV_NEAR_STOCKOUT].sum())
    evaluable_events = tp + fn
    excluded_events = total_actual_events - evaluable_events

    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "total_actual_events": total_actual_events,
        "evaluable_events": evaluable_events,
        "excluded_events_insufficient_history": excluded_events,
    }
