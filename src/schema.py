"""Canonical schema for the transaction ledger.

This module is the single source of truth for column names and expected
types. Every other module must reference columns through the constants
below (e.g. ``schema.REVENUE``), never as hardcoded string literals, so
that a schema change - which will happen once real data arrives - is a
one-file edit.

``validate(df)`` inspects a DataFrame against the schema and returns a
report. It never mutates or coerces the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --- Column name constants -------------------------------------------------

TXN_ID = "txn_id"
TXN_DATE = "txn_date"
CUSTOMER_ID = "customer_id"
CUSTOMER_SEGMENT = "customer_segment"
ROUTE_ID = "route_id"
LINE_ITEM = "line_item"
QUANTITY = "quantity"
REVENUE = "revenue"
DIRECT_COST = "direct_cost"
COST_CATEGORY = "cost_category"
INVOICE_ID = "invoice_id"

ALL_COLUMNS = [
    TXN_ID,
    TXN_DATE,
    CUSTOMER_ID,
    CUSTOMER_SEGMENT,
    ROUTE_ID,
    LINE_ITEM,
    QUANTITY,
    REVENUE,
    DIRECT_COST,
    COST_CATEGORY,
    INVOICE_ID,
]

# Coarse type categories used for comparison. Real exports load through
# pandas as plain objects, so "string" accepts object dtype rather than
# demanding pandas' dedicated StringDtype.
STRING = "string"
FLOAT = "float"
DATETIME = "datetime"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str  # one of STRING, FLOAT, DATETIME
    nullable: bool
    unique: bool = False
    notes: str = ""


SCHEMA: list[ColumnSpec] = [
    ColumnSpec(TXN_ID, STRING, nullable=False, unique=True),
    ColumnSpec(TXN_DATE, DATETIME, nullable=False),
    ColumnSpec(CUSTOMER_ID, STRING, nullable=False, notes="anonymised"),
    ColumnSpec(CUSTOMER_SEGMENT, STRING, nullable=True),
    ColumnSpec(ROUTE_ID, STRING, nullable=True, notes="origin/destination or product line"),
    ColumnSpec(LINE_ITEM, STRING, nullable=False),
    ColumnSpec(QUANTITY, FLOAT, nullable=False),
    ColumnSpec(REVENUE, FLOAT, nullable=False, notes="scaled currency"),
    ColumnSpec(DIRECT_COST, FLOAT, nullable=True, notes="may be missing in real data"),
    ColumnSpec(COST_CATEGORY, STRING, nullable=True),
    ColumnSpec(INVOICE_ID, STRING, nullable=False, notes="groups lines"),
]

SCHEMA_BY_NAME: dict[str, ColumnSpec] = {col.name: col for col in SCHEMA}

REQUIRED_COLUMNS = [col.name for col in SCHEMA if not col.nullable]
NULLABLE_COLUMNS = [col.name for col in SCHEMA if col.nullable]
UNIQUE_COLUMNS = [col.name for col in SCHEMA if col.unique]


# --- Validation report -------------------------------------------------------


@dataclass
class ValidationReport:
    row_count: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    type_mismatches: dict[str, str] = field(default_factory=dict)
    null_rates: dict[str, float] = field(default_factory=dict)
    duplicate_key_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True only if every required column is present with a matching
        type and no duplicate unique keys. Nulls in nullable columns do
        not affect validity - callers should look at null_rates
        themselves for required columns.
        """
        return (
            not self.missing_columns
            and not self.type_mismatches
            and not self.duplicate_key_counts
        )

    def summary(self) -> str:
        lines = [f"rows: {self.row_count}", f"valid: {self.is_valid}"]
        if self.missing_columns:
            lines.append(f"missing columns: {self.missing_columns}")
        if self.unexpected_columns:
            lines.append(f"unexpected columns: {self.unexpected_columns}")
        if self.type_mismatches:
            lines.append(f"type mismatches: {self.type_mismatches}")
        if self.duplicate_key_counts:
            lines.append(f"duplicate keys: {self.duplicate_key_counts}")
        lines.append(f"null rates: {self.null_rates}")
        return "\n".join(lines)


def _observed_kind(dtype) -> str:
    kind = dtype.kind
    if kind in ("i", "u", "f"):
        return FLOAT
    if kind == "M":
        return DATETIME
    if kind in ("O", "U", "S") or str(dtype) == "string":
        return STRING
    return f"unknown({dtype})"


def validate(df: pd.DataFrame) -> ValidationReport:
    """Compare df against SCHEMA and report discrepancies.

    Never coerces or mutates df. Callers decide what to do with a report;
    this function only observes.
    """
    report = ValidationReport(row_count=len(df))

    present = set(df.columns)
    expected = set(ALL_COLUMNS)
    report.missing_columns = [c for c in ALL_COLUMNS if c not in present]
    report.unexpected_columns = sorted(present - expected)

    for col in SCHEMA:
        if col.name not in present:
            continue
        series = df[col.name]
        report.null_rates[col.name] = float(series.isna().mean())

        observed = _observed_kind(series.dtype)
        if observed != col.kind:
            report.type_mismatches[col.name] = f"expected {col.kind}, got {series.dtype}"

    for col_name in UNIQUE_COLUMNS:
        if col_name not in present:
            continue
        dup_count = int(df[col_name].duplicated(keep=False).sum())
        if dup_count:
            report.duplicate_key_counts[col_name] = dup_count

    return report
