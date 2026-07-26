"""Empty-running (dead-head) detection and vehicle utilisation.

Core idea: a vehicle's next trip should start where its last trip
ended. When it doesn't, the vehicle repositioned empty between the
two - a dead-head leg. This module finds those gaps, sizes them in km
via an explicit distance matrix, and quantifies utilisation (idle days)
per vehicle.

Distances are looked up from the matrix, never estimated. A leg whose
origin/destination pair is missing from the matrix is flagged
(km null, *_km_missing=True) - never silently dropped or treated as
zero km, which would understate empty running without anyone knowing.

No cost appears here except through price_empty_km, and only when the
caller supplies a diesel rate and mileage explicitly - nothing is
hardcoded, so the assumption is visible and defensible in place.
"""

from __future__ import annotations

import pandas as pd

from src import schema

EMPTY_FROM = "empty_from"
EMPTY_TO = "empty_to"
DATE_GAP_DAYS = "date_gap_days"
EMPTY_KM = "empty_km"
KM_MISSING = "km_missing"
IDLE_DAYS = "idle_days"
LOADED_KM = "loaded_km"
LOADED_KM_MISSING = "loaded_km_missing"


def _require_datetime(df: pd.DataFrame, col: str) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        raise TypeError(f"{col} must already be parsed to datetime64, got {df[col].dtype}")


def _distance_map(distances_df: pd.DataFrame) -> dict[tuple, float]:
    return dict(
        zip(
            zip(distances_df[schema.ORIGIN_ID], distances_df[schema.DESTINATION_ID]),
            distances_df[schema.DISTANCE_KM],
        )
    )


def detect_empty_legs(trips_df: pd.DataFrame, distances_df: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive same-vehicle trip pair where the previous
    trip's destination doesn't match the next trip's origin.

    Columns: vehicle_id, empty_from, empty_to, date_gap_days, empty_km,
    km_missing. A pair absent from distances_df gets empty_km=NaN and
    km_missing=True rather than a guessed or zeroed distance.
    """
    _require_datetime(trips_df, schema.TRIP_DATE)
    dist = _distance_map(distances_df)

    rows = []
    for vehicle_id, group in trips_df.sort_values(schema.TRIP_DATE).groupby(schema.VEHICLE_ID):
        trips = group.reset_index(drop=True)
        for i in range(len(trips) - 1):
            current = trips.iloc[i]
            nxt = trips.iloc[i + 1]
            if current[schema.DESTINATION_ID] == nxt[schema.ORIGIN_ID]:
                continue
            pair = (current[schema.DESTINATION_ID], nxt[schema.ORIGIN_ID])
            km = dist.get(pair)
            rows.append(
                {
                    schema.VEHICLE_ID: vehicle_id,
                    EMPTY_FROM: current[schema.DESTINATION_ID],
                    EMPTY_TO: nxt[schema.ORIGIN_ID],
                    DATE_GAP_DAYS: (nxt[schema.TRIP_DATE] - current[schema.TRIP_DATE]).days,
                    EMPTY_KM: km,
                    KM_MISSING: km is None,
                }
            )

    return pd.DataFrame(
        rows, columns=[schema.VEHICLE_ID, EMPTY_FROM, EMPTY_TO, DATE_GAP_DAYS, EMPTY_KM, KM_MISSING]
    )


def summarize_vehicle_km(trips_df: pd.DataFrame, distances_df: pd.DataFrame) -> pd.DataFrame:
    """Per vehicle: total loaded km (each trip's own origin->destination),
    total empty km (from detect_empty_legs), and empty % of total km
    moved. missing_km_legs counts loaded and empty legs whose distance
    pair wasn't in the matrix - a nonzero count means loaded_km/empty_km
    (and therefore empty_pct) understate the true total, silently by
    pandas' sum() unless this column is read alongside them.
    """
    _require_datetime(trips_df, schema.TRIP_DATE)
    dist = _distance_map(distances_df)

    loaded = trips_df.copy()
    pairs = list(zip(loaded[schema.ORIGIN_ID], loaded[schema.DESTINATION_ID]))
    loaded[LOADED_KM] = [dist.get(p) for p in pairs]
    loaded[LOADED_KM_MISSING] = loaded[LOADED_KM].isna()

    loaded_summary = loaded.groupby(schema.VEHICLE_ID).agg(
        loaded_km=(LOADED_KM, "sum"),
        loaded_legs_missing=(LOADED_KM_MISSING, "sum"),
    )

    empty_df = detect_empty_legs(trips_df, distances_df)
    if len(empty_df):
        empty_summary = empty_df.groupby(schema.VEHICLE_ID).agg(
            empty_km=(EMPTY_KM, "sum"),
            empty_legs=(EMPTY_KM, "size"),
            empty_legs_missing=(KM_MISSING, "sum"),
        )
    else:
        empty_summary = pd.DataFrame(columns=["empty_km", "empty_legs", "empty_legs_missing"])
        empty_summary.index.name = schema.VEHICLE_ID

    summary = loaded_summary.join(empty_summary, how="left")
    summary["empty_km"] = summary["empty_km"].fillna(0.0)
    summary["empty_legs"] = summary["empty_legs"].fillna(0).astype(int)
    summary["empty_legs_missing"] = summary["empty_legs_missing"].fillna(0).astype(int)
    summary["missing_km_legs"] = summary["loaded_legs_missing"] + summary["empty_legs_missing"]

    total_km = summary["loaded_km"] + summary["empty_km"]
    summary["empty_pct"] = summary["empty_km"] / total_km.where(total_km > 0)

    return summary.reset_index()[
        [
            schema.VEHICLE_ID,
            "loaded_km",
            "empty_km",
            "empty_pct",
            "empty_legs",
            "missing_km_legs",
        ]
    ]


def utilisation(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Per vehicle, per trip: idle days between this trip's date and the
    next trip's date for the same vehicle. A vehicle's last trip has no
    "next" yet, so it contributes no row - there's nothing to measure
    until another trip happens.
    """
    _require_datetime(trips_df, schema.TRIP_DATE)

    rows = []
    for vehicle_id, group in trips_df.sort_values(schema.TRIP_DATE).groupby(schema.VEHICLE_ID):
        trips = group.reset_index(drop=True)
        for i in range(len(trips) - 1):
            gap = (trips.loc[i + 1, schema.TRIP_DATE] - trips.loc[i, schema.TRIP_DATE]).days
            rows.append(
                {
                    schema.VEHICLE_ID: vehicle_id,
                    schema.TRIP_DATE: trips.loc[i, schema.TRIP_DATE],
                    IDLE_DAYS: gap,
                }
            )

    return pd.DataFrame(rows, columns=[schema.VEHICLE_ID, schema.TRIP_DATE, IDLE_DAYS])


def utilisation_summary(utilisation_df: pd.DataFrame) -> pd.DataFrame:
    """Per vehicle: total and average idle days, and how many gaps that's
    computed from (a vehicle with a single trip has zero gaps counted -
    not zero idle days, which would misleadingly read as "always busy").
    """
    return (
        utilisation_df.groupby(schema.VEHICLE_ID)
        .agg(
            total_idle_days=(IDLE_DAYS, "sum"),
            avg_idle_days=(IDLE_DAYS, "mean"),
            gaps_counted=(IDLE_DAYS, "size"),
        )
        .reset_index()
    )


def price_empty_km(empty_df: pd.DataFrame, diesel_price_per_litre: float, km_per_litre: float) -> pd.DataFrame:
    """Add an empty_cost column (rupees) to a copy of empty_df, converting
    empty_km via an explicit diesel rate and mileage - both required
    arguments, never a hardcoded default, so the assumption behind the
    number is always visible at the call site. Rows with a missing
    empty_km (km_missing=True) get a null empty_cost, not a zero.
    """
    if diesel_price_per_litre <= 0 or km_per_litre <= 0:
        raise ValueError("diesel_price_per_litre and km_per_litre must both be positive")

    df = empty_df.copy()
    litres = df[EMPTY_KM] / km_per_litre
    df["empty_cost"] = litres * diesel_price_per_litre
    return df
