"""Streamlit dashboard - renders processed data, no analysis logic here.

Two independent data sources, either or both may be present:
- cost/margin: data/processed/{ledger,overhead}.csv, written by
  `make clean-data`. Views: margin by customer/route/month, concentration.
- routing: data/processed/trips.csv (or an uploaded CSV) + a distance
  matrix (data/reference/distances.csv by default, or uploaded). View:
  empty km by vehicle.

Every view is a thin render over results already produced by
src/margin.py, src/slice.py, or src/routing.py - if a number looks
wrong, the bug is in those modules, not here. Missing one data source
skips that source's views with a plain message rather than crashing
the whole app - the two pipelines are independent by design.

Run with: streamlit run dashboard/app.py  (or `make dashboard`)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import clean, margin, routing, schema
from src import slice as slice_

PROCESSED_DIR = Path("data/processed")
LEDGER_PATH = PROCESSED_DIR / "ledger.csv"
OVERHEAD_PATH = PROCESSED_DIR / "overhead.csv"
TRIPS_PATH = PROCESSED_DIR / "trips.csv"
DEFAULT_DISTANCES_PATH = Path("data/reference/distances.csv")

STRATEGY_LABELS = {
    "direct_cost_only": "Direct cost only",
    "proportional_by_revenue": "Proportional by revenue",
    "per_unit_by_quantity": "Per-unit by quantity",
}


@st.cache_data
def load_ledger() -> pd.DataFrame:
    return pd.read_csv(LEDGER_PATH, parse_dates=[schema.TXN_DATE, schema.PAYMENT_DATE])


@st.cache_data
def load_overhead() -> pd.DataFrame:
    return pd.read_csv(OVERHEAD_PATH, parse_dates=[schema.PERIOD_MONTH])


def load_cost_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not (LEDGER_PATH.exists() and OVERHEAD_PATH.exists()):
        return None, None
    return load_ledger(), load_overhead()


def load_trips(uploaded_file) -> pd.DataFrame | None:
    """An uploaded file is cleaned + validated fresh, since it hasn't
    been through `python -m src.pipeline --trips`. Falls back to the
    already-cleaned data/processed/trips.csv if nothing was uploaded.
    """
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        cleaned, _ = clean.clean_trips(raw)
        report = schema.validate(cleaned, schema=schema.TRIPS_SCHEMA)
        if not report.is_valid:
            st.error(f"Uploaded trips file failed validation:\n\n```\n{report.summary()}\n```")
            return None
        return cleaned
    if TRIPS_PATH.exists():
        return pd.read_csv(TRIPS_PATH, parse_dates=[schema.TRIP_DATE])
    return None


def load_distances(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DEFAULT_DISTANCES_PATH.exists():
        return pd.read_csv(DEFAULT_DISTANCES_PATH)
    return None


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


def view_empty_km_by_vehicle(trips_df: pd.DataFrame, distances_df: pd.DataFrame | None) -> None:
    st.header("Empty km by vehicle")
    if distances_df is None:
        st.warning(
            "No distance matrix available - upload one in the sidebar, or add "
            "data/reference/distances.csv (see that folder's README before using it on real data)."
        )
        return

    summary = routing.summarize_vehicle_km(trips_df, distances_df).sort_values("empty_pct", ascending=False)
    st.dataframe(summary, width="stretch")
    st.bar_chart(summary.set_index(schema.VEHICLE_ID)["empty_pct"])

    missing = int(summary["missing_km_legs"].sum())
    if missing > 0:
        st.caption(
            f"{missing} leg(s) have no matching pair in the distance matrix - their km is "
            "excluded from the totals above, not zeroed. Add those pairs to the distance matrix."
        )


def main() -> None:
    st.set_page_config(page_title="Ledger Margin Analysis", layout="wide")
    st.title("Ledger Margin Analysis")

    st.sidebar.header("Trips data (routing)")
    uploaded_trips = st.sidebar.file_uploader("Upload trips CSV", type="csv", key="trips_upload")
    uploaded_distances = st.sidebar.file_uploader(
        "Upload distance matrix CSV (optional)", type="csv", key="distances_upload"
    )

    cleaned_ledger, overhead_df = load_cost_data()
    trips_df = load_trips(uploaded_trips)
    distances_df = load_distances(uploaded_distances)

    if cleaned_ledger is None and trips_df is None:
        st.error(
            "No data available. Run `make clean-data` for the cost/margin pipeline, "
            "or upload a trips CSV in the sidebar for the routing pipeline."
        )
        st.stop()
        return

    if cleaned_ledger is not None:
        strategy_name = st.sidebar.selectbox(
            "Allocation strategy", list(STRATEGY_LABELS), format_func=lambda k: STRATEGY_LABELS[k]
        )
        ledger_with_margin = margin.compute_margin(cleaned_ledger, overhead_df, strategy_name)
        view_margin_by_customer(ledger_with_margin)
        view_margin_by_route(ledger_with_margin)
        view_margin_by_month(ledger_with_margin)
        view_concentration(ledger_with_margin)
    else:
        st.info("Cost/margin data not found - skipping those views. Run `make clean-data` to generate it.")

    if trips_df is not None:
        view_empty_km_by_vehicle(trips_df, distances_df)
    else:
        st.info("No trips data found - upload a trips CSV in the sidebar to see the empty-running view.")


main()
