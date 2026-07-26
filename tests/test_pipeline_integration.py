import pandas as pd

from src import margin, schema, slice as slice_


def test_synth_clean_margin_slice_end_to_end(cleaned_ledger: pd.DataFrame, raw_overhead: pd.DataFrame) -> None:
    result = margin.compute_margin(cleaned_ledger, raw_overhead, "proportional_by_revenue")
    assert margin.MARGIN in result.columns

    by_customer = slice_.by_customer(result)
    by_route = slice_.by_route(result)
    by_month = slice_.by_month(result)

    assert by_customer["revenue"].sum() > 0
    assert by_route["revenue"].sum() > 0
    assert by_month["revenue"].sum() > 0

    concentration = slice_.concentration(result, schema.CUSTOMER_ID, top_n=3)
    assert len(concentration) == 3
    assert concentration["cumulative_share_of_total"].is_monotonic_increasing
