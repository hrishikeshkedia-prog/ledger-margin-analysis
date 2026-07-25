import pandas as pd

from src import schema


def _valid_ledger_row() -> dict:
    return {
        schema.TXN_ID: "TXN0000001",
        schema.TXN_DATE: pd.Timestamp("2024-01-01"),
        schema.TXN_TYPE: schema.TXN_TYPE_SALE,
        schema.CUSTOMER_ID: "CUST0001",
        schema.CUSTOMER_SEGMENT: "enterprise",
        schema.ORIGIN_ID: "LOC001",
        schema.DESTINATION_ID: "LOC002",
        schema.VENDOR_ID: "VEND001",
        schema.LINE_ITEM: "freight_charge",
        schema.QUANTITY: 10.0,
        schema.UOM: "trips",
        schema.REVENUE: 100.0,
        schema.DIRECT_COST: 40.0,
        schema.COST_CATEGORY: "fuel",
        schema.INVOICE_ID: "INV0000001",
        schema.PAYMENT_DATE: pd.Timestamp("2024-01-30"),
    }


def test_validate_passes_on_well_formed_ledger():
    df = pd.DataFrame([_valid_ledger_row(), _valid_ledger_row()])
    df[schema.TXN_ID] = ["TXN0000001", "TXN0000002"]  # keep unique
    report = schema.validate(df)
    assert report.is_valid
    assert report.missing_columns == []
    assert report.type_mismatches == {}
    assert report.duplicate_key_counts == {}


def test_validate_flags_missing_columns():
    df = pd.DataFrame({schema.TXN_ID: ["TXN0000001"]})
    report = schema.validate(df)
    assert not report.is_valid
    assert schema.TXN_DATE in report.missing_columns
    assert schema.REVENUE in report.missing_columns


def test_validate_flags_type_mismatch():
    row = _valid_ledger_row()
    row[schema.TXN_DATE] = "2024-01-01"  # string, not datetime
    df = pd.DataFrame([row])
    report = schema.validate(df)
    assert not report.is_valid
    assert schema.TXN_DATE in report.type_mismatches


def test_validate_flags_null_in_required_column():
    row_a = _valid_ledger_row()
    row_b = _valid_ledger_row()
    row_b[schema.TXN_ID] = "TXN0000002"
    row_b[schema.CUSTOMER_ID] = None  # customer_id is required, not nullable
    df = pd.DataFrame([row_a, row_b])
    report = schema.validate(df)
    assert not report.is_valid
    assert schema.CUSTOMER_ID in report.null_violations


def test_validate_allows_null_in_nullable_column():
    row = _valid_ledger_row()
    row[schema.ORIGIN_ID] = None
    df = pd.DataFrame([row])
    report = schema.validate(df)
    assert schema.ORIGIN_ID not in report.null_violations
    assert schema.ORIGIN_ID not in report.missing_columns


def test_validate_flags_duplicate_unique_key():
    row_a = _valid_ledger_row()
    row_b = _valid_ledger_row()  # same txn_id as row_a
    df = pd.DataFrame([row_a, row_b])
    report = schema.validate(df)
    assert not report.is_valid
    assert report.duplicate_key_counts.get(schema.TXN_ID) == 2


def test_validate_overhead_schema():
    df = pd.DataFrame(
        [
            {
                schema.PERIOD_MONTH: pd.Timestamp("2024-01-01"),
                schema.COST_CATEGORY: "rent",
                schema.AMOUNT: 5000.0,
                schema.DESCRIPTION: None,
            }
        ]
    )
    report = schema.validate(df, schema=schema.OVERHEAD_SCHEMA)
    assert report.is_valid
