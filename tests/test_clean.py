import pandas as pd

from src import clean, schema


def test_normalise_dates_parses_mixed_formats():
    df = pd.DataFrame(
        {
            schema.TXN_DATE: ["2024-01-05", "15-Feb-2024", "20/03/2024", "04/25/2024"],
            schema.PAYMENT_DATE: ["2024-02-01", None, "bad-date", "2024-05-01"],
        }
    )
    log: clean.DecisionLog = []
    cleaned = clean.normalise_dates(df, log)

    assert list(cleaned[schema.TXN_DATE]) == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-03-20"),
        pd.Timestamp("2024-04-25"),  # 04/25 only parses as %m/%d - unambiguous
    ]
    assert pd.isna(cleaned[schema.PAYMENT_DATE].iloc[1])
    assert pd.isna(cleaned[schema.PAYMENT_DATE].iloc[2])  # "bad-date" unparseable
    assert len(log) == 2
    assert all(entry.step == "normalise_dates" for entry in log)


def test_normalise_ids_strips_and_uppercases():
    df = pd.DataFrame({schema.CUSTOMER_ID: [" cust0001", "CUST0002 ", "cust0003"]})
    log: clean.DecisionLog = []
    cleaned = clean.normalise_ids(df, log)

    assert list(cleaned[schema.CUSTOMER_ID]) == ["CUST0001", "CUST0002", "CUST0003"]
    entry = next(e for e in log if "customer_id" in e.rule)
    assert entry.rows_affected == 3  # all three rows changed


def test_normalise_ids_leaves_nulls_alone():
    df = pd.DataFrame({schema.VENDOR_ID: [" VEND001", None]})
    log: clean.DecisionLog = []
    cleaned = clean.normalise_ids(df, log)
    assert cleaned[schema.VENDOR_ID].iloc[0] == "VEND001"
    assert pd.isna(cleaned[schema.VENDOR_ID].iloc[1])


def test_deduplicate_txn_id_keeps_first_occurrence():
    df = pd.DataFrame({schema.TXN_ID: ["A", "B", "A", "C", "B"]})
    log: clean.DecisionLog = []
    cleaned = clean.deduplicate_txn_id(df, log)

    assert list(cleaned[schema.TXN_ID]) == ["A", "B", "C"]
    assert log[0].rows_affected == 2


def test_flag_customer_name_variants_does_not_mutate_df():
    df = pd.DataFrame({schema.CUSTOMER_ID: ["CUST0001", "CUST0001X", "CUST0099"]})
    log: clean.DecisionLog = []
    before = df.copy()
    result = clean.flag_customer_name_variants(df, log, similarity_threshold=0.85)

    pd.testing.assert_frame_equal(result, before)
    entry = log[0]
    assert entry.step == "flag_customer_name_variants"
    assert entry.rows_affected >= 1  # CUST0001 / CUST0001X should be flagged


def test_handle_nulls_logs_every_nullable_column_present():
    df = pd.DataFrame({col: [None] for col in schema.NULLABLE_COLUMNS})
    log: clean.DecisionLog = []
    clean.handle_nulls(df, log)

    logged_columns = {entry.rule.split(":")[0] for entry in log}
    assert logged_columns == set(schema.NULLABLE_COLUMNS)
    assert all(entry.rows_affected == 1 for entry in log)


def test_clean_end_to_end_passes_schema_validation(raw_ledger: pd.DataFrame) -> None:
    cleaned, log = clean.clean(raw_ledger)
    report = schema.validate(cleaned)
    assert report.is_valid, report.summary()
    assert len(log) > 0
    # txn_id must be genuinely unique after cleaning, not just structurally checked
    assert cleaned[schema.TXN_ID].duplicated().sum() == 0
