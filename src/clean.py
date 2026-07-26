"""Ledger cleaning pipeline.

Each cleaning decision is a separate, named step that takes a DataFrame
and returns a DataFrame, appending an entry to a shared decision log
(step name, rows affected, rule applied). No step silently coerces or
drops data without logging exactly what it did and why, so the pipeline
is auditable end to end: compare the raw export against
``decision_log`` to see everything that changed and the stated rule
behind it.

``clean(df)`` runs the full pipeline and returns ``(df, decision_log)``.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import pandas as pd

from src import schema

# Ambiguous D/M vs M/D dates (e.g. "03/04/2024") are resolved day-first
# because %d/%m/%Y is tried before %m/%d/%Y below. This is a stated
# policy, not a guess made silently.
DATE_FORMATS = ["%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y"]

ID_COLUMNS = [
    schema.TXN_ID,
    schema.CUSTOMER_ID,
    schema.INVOICE_ID,
    schema.ORIGIN_ID,
    schema.DESTINATION_ID,
    schema.VENDOR_ID,
    schema.VEHICLE_ID,  # trips-only column; a no-op on ledger data, which doesn't have it
]

# Explicit, stated policy per nullable column. Every entry here means
# "leave the null as-is" - none of them are silently filled - but the
# reason differs per column and downstream code (margin.py,
# confounder.py) relies on nulls meaning exactly this.
NULL_POLICY = {
    schema.CUSTOMER_SEGMENT: "left null - segment genuinely unknown, not imputed",
    schema.ORIGIN_ID: "left null - product-line sales without a route leg have no origin",
    schema.DESTINATION_ID: "left null - product-line sales without a route leg have no destination",
    schema.VENDOR_ID: "left null - no subcontractor on this line (fulfilled in-house)",
    schema.UOM: "left null - some line items (fees) have no physical unit",
    schema.DIRECT_COST: "left null - absence handled explicitly by margin.py allocation strategies, never filled with 0",
    schema.COST_CATEGORY: "left null - no cost category without a direct_cost figure to categorise",
    schema.PAYMENT_DATE: "left null - invoice not yet settled; used by confounder.py to test payment-timing effects",
}


@dataclass
class DecisionLogEntry:
    step: str
    rows_affected: int
    rule: str


DecisionLog = list[DecisionLogEntry]


def decision_log_to_frame(log: DecisionLog) -> pd.DataFrame:
    return pd.DataFrame([{"step": e.step, "rows_affected": e.rows_affected, "rule": e.rule} for e in log])


def _parse_one_date(value):
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    return pd.NaT


def normalise_dates(
    df: pd.DataFrame,
    log: DecisionLog,
    date_columns: tuple[str, ...] = (schema.TXN_DATE, schema.PAYMENT_DATE),
) -> pd.DataFrame:
    """Parse date columns from mixed string formats into real dates.
    Values that match none of DATE_FORMATS become null rather than
    being guessed at. Defaults to the ledger's txn_date/payment_date;
    pass date_columns=(schema.TRIP_DATE,) for the trips schema.
    """
    df = df.copy()
    for col in date_columns:
        if col not in df.columns:
            continue
        had_value = df[col].notna()
        parsed = df[col].map(_parse_one_date)
        newly_unparseable = int((had_value & parsed.isna()).sum())
        df[col] = parsed
        log.append(
            DecisionLogEntry(
                step="normalise_dates",
                rows_affected=int(had_value.sum()),
                rule=(
                    f"{col}: tried formats {DATE_FORMATS} in that priority order "
                    "(ambiguous D/M vs M/D resolved day-first); "
                    f"{newly_unparseable} value(s) matched none and became null"
                ),
            )
        )
    return df


def normalise_ids(df: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    """Strip surrounding whitespace and upper-case ID-like columns so
    ' cust0003' and 'CUST0003' collapse to the same key.
    """
    df = df.copy()
    for col in ID_COLUMNS:
        if col not in df.columns:
            continue
        original = df[col]
        cleaned = original.where(original.isna(), original.astype("string").str.strip().str.upper())
        changed = int(((original != cleaned) & original.notna()).sum())
        df[col] = cleaned
        log.append(
            DecisionLogEntry(
                step="normalise_ids",
                rows_affected=changed,
                rule=f"{col}: stripped surrounding whitespace and upper-cased",
            )
        )
    return df


def deduplicate_txn_id(df: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    """Drop exact duplicate txn_id rows (post ID-normalisation), keeping
    the first occurrence. Only exact duplicates are touched here - near
    duplicates are the job of flag_customer_name_variants, and only for
    customer_id, not txn_id.
    """
    dup_mask = df[schema.TXN_ID].duplicated(keep="first")
    rows_dropped = int(dup_mask.sum())
    cleaned = df.loc[~dup_mask].reset_index(drop=True)
    log.append(
        DecisionLogEntry(
            step="deduplicate_txn_id",
            rows_affected=rows_dropped,
            rule="exact duplicate txn_id (after normalisation) - kept first occurrence, dropped the rest",
        )
    )
    return cleaned


def flag_customer_name_variants(
    df: pd.DataFrame, log: DecisionLog, similarity_threshold: float = 0.9
) -> pd.DataFrame:
    """Find pairs of distinct customer_id values that look like the same
    customer under a typo or partial-edit (e.g. a transposed character),
    and flag them in the decision log. Never merges automatically - a
    human decides whether flagged pairs are really the same customer.

    Pairwise comparison over unique customer_id values: fine at the
    hundreds of customers this project deals with, but would need
    blocking or embeddings to scale to a real export with thousands of
    distinct customers.
    """
    df = df.copy()
    if schema.CUSTOMER_ID not in df.columns:
        log.append(
            DecisionLogEntry(
                step="flag_customer_name_variants",
                rows_affected=0,
                rule=f"skipped - no {schema.CUSTOMER_ID} column present",
            )
        )
        return df

    unique_ids = sorted(df[schema.CUSTOMER_ID].dropna().unique())
    flagged_pairs = []
    for i, a in enumerate(unique_ids):
        for b in unique_ids[i + 1 :]:
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            if ratio >= similarity_threshold:
                flagged_pairs.append((a, b, round(ratio, 3)))
    log.append(
        DecisionLogEntry(
            step="flag_customer_name_variants",
            rows_affected=len(flagged_pairs),
            rule=(
                f"pairs of distinct customer_id values with similarity >= {similarity_threshold} "
                f"flagged for human review, not auto-merged: {flagged_pairs}"
            ),
        )
    )
    return df


def handle_nulls(df: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    """Record the stated null-handling policy per nullable column. Every
    policy here is "leave it null" - this step never imputes - but each
    column's reason is logged so the decision is visible, not assumed.
    """
    df = df.copy()
    for col, policy in NULL_POLICY.items():
        if col not in df.columns:
            continue
        null_count = int(df[col].isna().sum())
        log.append(
            DecisionLogEntry(
                step="handle_nulls",
                rows_affected=null_count,
                rule=f"{col}: {policy}",
            )
        )
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, DecisionLog]:
    """Run the full ledger cleaning pipeline in order, returning the
    cleaned DataFrame alongside the full decision log.
    """
    log: DecisionLog = []
    df = normalise_dates(df, log)
    df = normalise_ids(df, log)
    df = deduplicate_txn_id(df, log)
    df = flag_customer_name_variants(df, log)
    df = handle_nulls(df, log)
    return df, log


def clean_trips(df: pd.DataFrame) -> tuple[pd.DataFrame, DecisionLog]:
    """Cleaning pipeline for the routing dataset (schema.TRIPS_SCHEMA).

    Reuses the shared date/ID/dedup steps above; skips
    flag_customer_name_variants (no customer_id in this schema) and
    handle_nulls (TRIPS_SCHEMA has no nullable columns - a null here is
    a validation failure downstream, not a policy decision to log).
    """
    log: DecisionLog = []
    df = normalise_dates(df, log, date_columns=(schema.TRIP_DATE,))
    df = normalise_ids(df, log)
    df = deduplicate_txn_id(df, log)
    return df, log
