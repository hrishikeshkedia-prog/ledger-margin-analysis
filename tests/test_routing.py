import math

import pandas as pd
import pytest

from src import routing, schema


def _trips() -> pd.DataFrame:
    # V1: A->B, then C->D (origin C != prev dest B => one dead-head leg
    # B->C), then D->E (origin D == prev dest D => no dead-head).
    # V2: a single, unrelated trip - nothing to compare it against yet.
    return pd.DataFrame(
        {
            schema.TXN_ID: ["T1", "T2", "T3", "T4"],
            schema.TRIP_DATE: pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-02"]),
            schema.VEHICLE_ID: ["V1", "V1", "V1", "V2"],
            schema.ORIGIN_ID: ["A", "C", "D", "X"],
            schema.DESTINATION_ID: ["B", "D", "E", "Y"],
            schema.REVENUE: [100.0, 150.0, 120.0, 90.0],
        }
    )


def _distances(include_bc: bool = True) -> pd.DataFrame:
    rows = [
        {schema.ORIGIN_ID: "A", schema.DESTINATION_ID: "B", schema.DISTANCE_KM: 100.0},
        {schema.ORIGIN_ID: "C", schema.DESTINATION_ID: "D", schema.DISTANCE_KM: 80.0},
        {schema.ORIGIN_ID: "D", schema.DESTINATION_ID: "E", schema.DISTANCE_KM: 60.0},
        {schema.ORIGIN_ID: "X", schema.DESTINATION_ID: "Y", schema.DISTANCE_KM: 40.0},
    ]
    if include_bc:
        rows.append({schema.ORIGIN_ID: "B", schema.DESTINATION_ID: "C", schema.DISTANCE_KM: 55.0})
    return pd.DataFrame(rows)


def test_detect_empty_legs_finds_the_one_known_dead_head():
    empty = routing.detect_empty_legs(_trips(), _distances())

    assert len(empty) == 1
    row = empty.iloc[0]
    assert row[schema.VEHICLE_ID] == "V1"
    assert row[routing.EMPTY_FROM] == "B"
    assert row[routing.EMPTY_TO] == "C"
    assert row[routing.DATE_GAP_DAYS] == 2
    assert row[routing.EMPTY_KM] == 55.0
    assert row[routing.KM_MISSING] == False  # noqa: E712


def test_detect_empty_legs_skips_a_matching_destination_origin_pair():
    empty = routing.detect_empty_legs(_trips(), _distances())
    # trip3's origin (D) matches trip2's destination (D) - not a dead-head.
    pairs = list(zip(empty[routing.EMPTY_FROM], empty[routing.EMPTY_TO]))
    assert ("D", "D") not in pairs
    assert ("D", "E") not in pairs


def test_missing_distance_pair_is_flagged_not_zeroed():
    empty = routing.detect_empty_legs(_trips(), _distances(include_bc=False))

    assert len(empty) == 1
    row = empty.iloc[0]
    assert row[routing.KM_MISSING] == True  # noqa: E712
    assert pd.isna(row[routing.EMPTY_KM])  # never a silent 0.0


def test_summarize_vehicle_km_reports_missing_legs_instead_of_hiding_them():
    trips = pd.DataFrame(
        {
            schema.TXN_ID: ["T1", "T2"],
            schema.TRIP_DATE: pd.to_datetime(["2024-01-01", "2024-01-03"]),
            schema.VEHICLE_ID: ["V1", "V1"],
            schema.ORIGIN_ID: ["A", "C"],
            schema.DESTINATION_ID: ["B", "D"],
            schema.REVENUE: [100.0, 150.0],
        }
    )
    empty_distances = pd.DataFrame({schema.ORIGIN_ID: [], schema.DESTINATION_ID: [], schema.DISTANCE_KM: []})

    summary = routing.summarize_vehicle_km(trips, empty_distances)
    row = summary.iloc[0]

    # 2 loaded legs (A->B, C->D) + 1 empty leg (B->C), all missing from
    # an empty matrix - the shortfall must be visible, not silently 0.
    assert row["missing_km_legs"] == 3
    assert math.isnan(row["empty_pct"])  # not a misleading 0% from 0/0


def test_utilisation_counts_idle_days_on_a_known_fixture():
    trips = pd.DataFrame(
        {
            schema.TXN_ID: ["T1", "T2", "T3"],
            schema.TRIP_DATE: pd.to_datetime(["2024-01-01", "2024-01-06", "2024-01-10"]),
            schema.VEHICLE_ID: ["V1", "V1", "V1"],
            schema.ORIGIN_ID: ["A", "B", "C"],
            schema.DESTINATION_ID: ["B", "C", "D"],
            schema.REVENUE: [100.0, 100.0, 100.0],
        }
    )

    util = routing.utilisation(trips)
    assert list(util[routing.IDLE_DAYS]) == [5, 4]  # Jan1->Jan6 = 5, Jan6->Jan10 = 4

    summary = routing.utilisation_summary(util)
    row = summary.set_index(schema.VEHICLE_ID).loc["V1"]
    assert row["total_idle_days"] == 9
    assert row["gaps_counted"] == 2
    assert row["avg_idle_days"] == 4.5


def test_utilisation_produces_no_row_for_a_vehicles_single_trip():
    trips = pd.DataFrame(
        {
            schema.TXN_ID: ["T1"],
            schema.TRIP_DATE: pd.to_datetime(["2024-01-01"]),
            schema.VEHICLE_ID: ["V1"],
            schema.ORIGIN_ID: ["A"],
            schema.DESTINATION_ID: ["B"],
            schema.REVENUE: [100.0],
        }
    )
    util = routing.utilisation(trips)
    assert len(util) == 0


def test_price_empty_km_uses_the_passed_in_rate_not_a_hardcoded_one():
    empty = routing.detect_empty_legs(_trips(), _distances())
    priced = routing.price_empty_km(empty, diesel_price_per_litre=95.0, km_per_litre=4.0)

    expected = (55.0 / 4.0) * 95.0
    assert math.isclose(priced.iloc[0]["empty_cost"], expected)


def test_price_empty_km_leaves_missing_km_as_null_cost():
    empty = routing.detect_empty_legs(_trips(), _distances(include_bc=False))
    priced = routing.price_empty_km(empty, diesel_price_per_litre=95.0, km_per_litre=4.0)
    assert pd.isna(priced.iloc[0]["empty_cost"])


def test_price_empty_km_rejects_nonpositive_assumptions():
    empty = routing.detect_empty_legs(_trips(), _distances())
    with pytest.raises(ValueError):
        routing.price_empty_km(empty, diesel_price_per_litre=0, km_per_litre=4.0)
    with pytest.raises(ValueError):
        routing.price_empty_km(empty, diesel_price_per_litre=95.0, km_per_litre=-1)


def test_routing_end_to_end_on_synthetic_trips(raw_trips: pd.DataFrame, distances_fixture: pd.DataFrame) -> None:
    from src import clean

    cleaned, _ = clean.clean_trips(raw_trips)
    empty = routing.detect_empty_legs(cleaned, distances_fixture)
    summary = routing.summarize_vehicle_km(cleaned, distances_fixture)
    util_summary = routing.utilisation_summary(routing.utilisation(cleaned))

    assert schema.validate(cleaned, schema=schema.TRIPS_SCHEMA).is_valid
    assert len(summary) == cleaned[schema.VEHICLE_ID].nunique()
    assert (summary["empty_pct"].dropna() >= 0).all()
    assert (summary["empty_pct"].dropna() <= 1).all()
    assert set(util_summary[schema.VEHICLE_ID]) <= set(cleaned[schema.VEHICLE_ID])
