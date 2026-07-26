import pandas as pd

from src import detect


def test_detects_routing_only_from_canonical_columns():
    df = pd.DataFrame(
        {
            "vehicle_id": ["V1"],
            "origin_id": ["A"],
            "destination_id": ["B"],
            "date": ["2024-01-01"],
            "revenue": [100],
        }
    )
    result = detect.detect_capabilities(df)
    assert result["routing_available"] is True
    assert result["margin_available"] is False
    assert result["missing_for_margin"] == ["direct_cost"]


def test_detects_margin_only_from_direct_cost_column():
    df = pd.DataFrame({"txn_id": ["T1"], "txn_date": ["2024-01-01"], "direct_cost": [50], "revenue": [100]})
    result = detect.detect_capabilities(df)
    assert result["margin_available"] is True
    assert result["routing_available"] is False
    assert set(result["missing_for_routing"]) == {"vehicle_id", "origin_id", "destination_id", "date"}


def test_margin_available_via_overhead_flag_without_direct_cost_column():
    df = pd.DataFrame({"txn_id": ["T1"], "txn_date": ["2024-01-01"], "revenue": [100]})
    assert detect.detect_capabilities(df)["margin_available"] is False
    assert detect.detect_capabilities(df, overhead_available=True)["margin_available"] is True


def test_detects_both_when_columns_for_both_are_present():
    df = pd.DataFrame(
        {
            "vehicle_id": ["V1"],
            "origin_id": ["A"],
            "destination_id": ["B"],
            "date": ["2024-01-01"],
            "direct_cost": [50],
            "revenue": [100],
        }
    )
    result = detect.detect_capabilities(df)
    assert result["margin_available"] is True
    assert result["routing_available"] is True


def test_neither_available_reports_missing_columns_for_both():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    result = detect.detect_capabilities(df)
    assert result["margin_available"] is False
    assert result["routing_available"] is False
    assert "cannot analyse" in result["message"].lower()


def test_column_aliases_are_mapped_to_canonical_names():
    df = pd.DataFrame(
        {"veh": ["V1"], "from": ["A"], "to": ["B"], "date": ["2024-01-01"], "revenue": [100], "notes": ["x"]}
    )
    result = detect.detect_capabilities(df)
    assert result["routing_available"] is True
    assert result["canonical_columns"]["veh"] == "vehicle_id"
    assert result["canonical_columns"]["from"] == "origin_id"
    assert result["canonical_columns"]["to"] == "destination_id"
    # an unrecognized header is reported, never silently dropped
    assert result["unmapped_columns"] == ["notes"]


def test_detector_never_fabricates_a_missing_column():
    # No direct_cost, no overhead file - margin must stay unavailable
    # no matter how many routing columns are present.
    df = pd.DataFrame(
        {"vehicle_id": ["V1"], "origin_id": ["A"], "destination_id": ["B"], "date": ["2024-01-01"]}
    )
    result = detect.detect_capabilities(df, overhead_available=False)
    assert result["margin_available"] is False
    assert result["missing_for_margin"] == ["direct_cost"]
