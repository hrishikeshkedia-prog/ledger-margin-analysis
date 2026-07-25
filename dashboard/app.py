"""Streamlit dashboard - renders processed data, no analysis logic here.

Reads data/processed/ledger.csv (the cleaned ledger, output of
src/clean.py) and data/processed/overhead.csv, both written by
`make clean-data`. Every view below is a thin render over results
already produced by src/margin.py and src/slice.py - if a number looks
wrong, the bug is in those modules, not here.

Run with: streamlit run dashboard/app.py  (or `make dashboard`)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import margin, schema
from src import slice as slice_

PROCESSED_DIR = Path("data/processed")
LEDGER_PATH = PROCESSED_DIR / "ledger.csv"
OVERHEAD_PATH = PROCESSED_DIR / "overhead.csv"

STRATEGY_LABELS = {
    "direct_cost_only": "Direct cost only",
    "proportional_by_revenue": "Proportional by revenue",
    "per_unit_by_quantity": "Per-unit by quantity",
}


@st.cache_data
def load_ledger() -> pd.DataFrame:
    if not LEDGER_PATH.exists():
        raise FileNotFoundError(f"{LEDGER_PATH} not found - run `make clean-data` first.")
    return pd.read_csv(LEDGER_PATH, parse_dates=[schema.TXN_DATE, schema.PAYMENT_DATE])


@st.cache_data
def load_overhead() -> pd.DataFrame:
    if not OVERHEAD_PATH.exists():
        raise FileNotFoundError(f"{OVERHEAD_PATH} not found - run `make clean-data` first.")
    return pd.read_csv(OVERHEAD_PATH, parse_dates=[schema.PERIOD_MONTH])


def view_margin_by_customer(ledger_with_margin: pd.DataFrame) -> None:
    st.header("Margin by customer")
    table = slice_.by_customer(ledger_with_margin).sort_values(margin.MARGIN, ascending=False)
    st.dataframe(table, width="stretch")
    st.bar_chart(table.set_index(schema.CUSTOMER_ID)[margin.MARGIN_PCT])


def view_margin_by_route(ledger_with_margin: pd.DataFrame) -> None:
    st.header("Margin by route (origin -> destination)")
    table = slice_.by_route(ledger_with_margin).sort_values(margin.MARGIN, ascending=False)
    st.dataframe(table, width="stretch")


def view_margin_by_month(ledger_with_margin: pd.DataFrame) -> None:
    st.header("Margin by month")
    table = slice_.by_month(ledger_with_margin).sort_values(schema.PERIOD_MONTH)
    st.dataframe(table, width="stretch")
    st.line_chart(table.set_index(schema.PERIOD_MONTH)[[margin.MARGIN, schema.REVENUE]])


def view_concentration(ledger_with_margin: pd.DataFrame) -> None:
    st.header("Revenue concentration")
    dimension = st.selectbox(
        "Group by", [schema.CUSTOMER_ID, schema.ORIGIN_ID, schema.DESTINATION_ID], key="concentration_dim"
    )
    top_n = st.slider("Top N", min_value=3, max_value=25, value=10, key="concentration_top_n")
    table = slice_.concentration(ledger_with_margin, dimension, value_col=schema.REVENUE, top_n=top_n)
    st.dataframe(table, width="stretch")


def view_allocation_strategy_comparison(cleaned_ledger: pd.DataFrame, overhead_df: pd.DataFrame) -> None:
    st.header("Allocation strategy comparison")
    rows = []
    for name, label in STRATEGY_LABELS.items():
        result = margin.compute_margin(cleaned_ledger, overhead_df, name)
        rows.append(
            {
                "strategy": label,
                "total_revenue": result[schema.REVENUE].sum(),
                "total_allocated_cost": result[margin.ALLOCATED_COST].sum(),
                "total_margin": result[margin.MARGIN].sum(),
                "null_allocated_cost_rows": int(result[margin.ALLOCATED_COST].isna().sum()),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Ledger Margin Analysis", layout="wide")
    st.title("Ledger Margin Analysis")

    try:
        cleaned_ledger = load_ledger()
        overhead_df = load_overhead()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
        return

    strategy_name = st.sidebar.selectbox(
        "Allocation strategy", list(STRATEGY_LABELS), format_func=lambda k: STRATEGY_LABELS[k]
    )
    ledger_with_margin = margin.compute_margin(cleaned_ledger, overhead_df, strategy_name)

    view_margin_by_customer(ledger_with_margin)
    view_margin_by_route(ledger_with_margin)
    view_margin_by_month(ledger_with_margin)
    view_concentration(ledger_with_margin)
    view_allocation_strategy_comparison(cleaned_ledger, overhead_df)


main()
