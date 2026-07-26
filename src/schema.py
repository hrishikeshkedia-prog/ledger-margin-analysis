"""Canonical schemas for the ledger analysis project.

This module is the single source of truth for column names and expected
types, across input files for two independent, dormant-or-not-by-choice
pipelines:

- the cost/margin pipeline: the transaction ledger (``LEDGER_SCHEMA``) -
  one row per invoice line - and the period overhead file
  (``OVERHEAD_SCHEMA``) - one row per month per cost category, covering
  costs that are not attributable to a single transaction (driver
  salaries, EMIs, rent, office). ``margin.py``'s allocation strategies
  distribute this pool across transactions.
- the routing/empty-running pipeline: the trip ledger (``TRIPS_SCHEMA``)
  - one row per vehicle trip, no cost data - and the distance matrix
  (``DISTANCE_SCHEMA``). ``routing.py`` detects dead-head legs and
  quantifies utilisation from these.

Only one of these pipelines may have data available on a given real
export (freight ledgers so far have had either costed invoice lines or
vehicle trip logs, not both) - each schema and its pipeline stands
alone and neither assumes the other ran.

Every other module must reference columns through the constants below
(e.g. ``schema.REVENUE``), never as hardcoded string literals, so that a
schema change - which will happen once real data arrives - is a
one-file edit.

``validate(df, schema)`` inspects a DataFrame against a schema and
returns a report. It never mutates or coerces the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --- Ledger column name constants ------------------------------------------

TXN_ID = "txn_id"
TXN_DATE = "txn_date"
TXN_TYPE = "txn_type"
CUSTOMER_ID = "customer_id"
CUSTOMER_SEGMENT = "customer_segment"
ORIGIN_ID = "origin_id"
DESTINATION_ID = "destination_id"
VENDOR_ID = "vendor_id"
LINE_ITEM = "line_item"
QUANTITY = "quantity"
UOM = "uom"
REVENUE = "revenue"
DIRECT_COST = "direct_cost"
COST_CATEGORY = "cost_category"
INVOICE_ID = "invoice_id"
PAYMENT_DATE = "payment_date"

# Allowed values for txn_type. A row outside this set (or missing it
# entirely) is not a plain sale and must not be summed into topline
# revenue without a deliberate decision in clean.py.
TXN_TYPE_SALE = "sale"
TXN_TYPE_CREDIT_NOTE = "credit_note"
TXN_TYPE_CANCELLED = "cancelled"
TXN_TYPE_VALUES = frozenset({TXN_TYPE_SALE, TXN_TYPE_CREDIT_NOTE, TXN_TYPE_CANCELLED})

# --- Overhead column name constants ----------------------------------------

PERIOD_MONTH = "period_month"
# COST_CATEGORY is shared with the ledger schema: same concept (a label
# like "driver_salaries", "emi", "rent", "office"), same column name.
AMOUNT = "amount"
DESCRIPTION = "description"


# --- Type categories ---------------------------------------------------------

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


@dataclass(frozen=True)
class Schema:
    name: str
    columns: tuple[ColumnSpec, ...]

    @property
    def by_name(self) -> dict[str, ColumnSpec]:
        return {col.name: col for col in self.columns}

    @property
    def all_columns(self) -> list[str]:
        return [col.name for col in self.columns]

    @property
    def required_columns(self) -> list[str]:
        return [col.name for col in self.columns if not col.nullable]

    @property
    def nullable_columns(self) -> list[str]:
        return [col.name for col in self.columns if col.nullable]

    @property
    def unique_columns(self) -> list[str]:
        return [col.name for col in self.columns if col.unique]


LEDGER_SCHEMA = Schema(
    name="ledger",
    columns=(
        ColumnSpec(TXN_ID, STRING, nullable=False, unique=True),
        ColumnSpec(TXN_DATE, DATETIME, nullable=False),
        ColumnSpec(
            TXN_TYPE,
            STRING,
            nullable=False,
            notes=f"one of {sorted(TXN_TYPE_VALUES)}; without this, credit notes "
            "and cancellations silently inflate topline",
        ),
        ColumnSpec(CUSTOMER_ID, STRING, nullable=False, notes="anonymised"),
        ColumnSpec(CUSTOMER_SEGMENT, STRING, nullable=True),
        ColumnSpec(ORIGIN_ID, STRING, nullable=True, notes="origin of route or product line"),
        ColumnSpec(DESTINATION_ID, STRING, nullable=True, notes="destination of route or product line"),
        ColumnSpec(VENDOR_ID, STRING, nullable=True, notes="subcontractor/supplier fulfilling this line, if any"),
        ColumnSpec(LINE_ITEM, STRING, nullable=False),
        ColumnSpec(QUANTITY, FLOAT, nullable=False),
        ColumnSpec(UOM, STRING, nullable=True, notes="unit of quantity, e.g. tonnes/trips/cartons/litres"),
        ColumnSpec(REVENUE, FLOAT, nullable=False, notes="scaled currency"),
        ColumnSpec(DIRECT_COST, FLOAT, nullable=True, notes="may be missing in real data"),
        ColumnSpec(COST_CATEGORY, STRING, nullable=True),
        ColumnSpec(INVOICE_ID, STRING, nullable=False, notes="groups lines"),
        ColumnSpec(PAYMENT_DATE, DATETIME, nullable=True, notes="settlement date; needed to test payment-timing confounders"),
    ),
)

OVERHEAD_SCHEMA = Schema(
    name="overhead",
    columns=(
        ColumnSpec(PERIOD_MONTH, DATETIME, nullable=False, notes="first day of the month this cost applies to"),
        ColumnSpec(COST_CATEGORY, STRING, nullable=False, notes="e.g. driver_salaries, emi, rent, office"),
        ColumnSpec(AMOUNT, FLOAT, nullable=False),
        ColumnSpec(DESCRIPTION, STRING, nullable=True),
    ),
)

# --- Trip column name constants ---------------------------------------------
# TXN_ID, ORIGIN_ID, DESTINATION_ID, and REVENUE are shared with the ledger
# schema above: same concept, same column name, different file.

TRIP_DATE = "date"  # deliberately not TXN_DATE - the real trip export's own column name
VEHICLE_ID = "vehicle_id"

TRIPS_SCHEMA = Schema(
    name="trips",
    columns=(
        ColumnSpec(TXN_ID, STRING, nullable=False, unique=True),
        ColumnSpec(TRIP_DATE, DATETIME, nullable=False),
        ColumnSpec(VEHICLE_ID, STRING, nullable=False),
        ColumnSpec(ORIGIN_ID, STRING, nullable=False, notes="every trip has an origin - unlike the ledger schema, not nullable here"),
        ColumnSpec(DESTINATION_ID, STRING, nullable=False, notes="every trip has a destination - unlike the ledger schema, not nullable here"),
        ColumnSpec(REVENUE, FLOAT, nullable=False, notes="rupees; no cost columns exist in this pipeline"),
    ),
)

# --- Distance matrix column name constants ----------------------------------

DISTANCE_KM = "km"

DISTANCE_SCHEMA = Schema(
    name="distances",
    columns=(
        ColumnSpec(ORIGIN_ID, STRING, nullable=False),
        ColumnSpec(DESTINATION_ID, STRING, nullable=False),
        ColumnSpec(DISTANCE_KM, FLOAT, nullable=False),
    ),
)

# --- Column name aliases -----------------------------------------------------
# Maps a lowercased, whitespace-stripped real-world header to its
# canonical column constant above. src/detect.py uses this to recognize
# common variants (a different freight operator's export won't use our
# exact header names) without ever guessing a column's meaning from its
# data - an unrecognized header is reported as unmapped, never silently
# dropped or matched by a fuzzy guess.
COLUMN_ALIASES: dict[str, str] = {
    # vehicle_id
    "veh": VEHICLE_ID,
    "truck": VEHICLE_ID,
    "truck_id": VEHICLE_ID,
    "truck_no": VEHICLE_ID,
    "vehicle_no": VEHICLE_ID,
    "vehicle_number": VEHICLE_ID,
    # origin_id
    "origin": ORIGIN_ID,
    "from": ORIGIN_ID,
    "from_location": ORIGIN_ID,
    "source": ORIGIN_ID,
    "start_location": ORIGIN_ID,
    # destination_id
    "destination": DESTINATION_ID,
    "dest": DESTINATION_ID,
    "to": DESTINATION_ID,
    "to_location": DESTINATION_ID,
    "end_location": DESTINATION_ID,
    # revenue
    "amount": REVENUE,
    "fare": REVENUE,
    "freight_amount": REVENUE,
    "freight_rate": REVENUE,
    # direct_cost
    "cost": DIRECT_COST,
    "fuel_cost": DIRECT_COST,
    "trip_cost": DIRECT_COST,
    # txn_id / trip id
    "trip_id": TXN_ID,
    "trip_no": TXN_ID,
    # customer_id
    "customer": CUSTOMER_ID,
    "client": CUSTOMER_ID,
    "client_id": CUSTOMER_ID,
    # km
    "distance": DISTANCE_KM,
    "distance_km": DISTANCE_KM,
    "kms": DISTANCE_KM,
}

# Backwards-compatible flat aliases for the ledger schema, since it is the
# schema most other modules validate against by default.
ALL_COLUMNS = LEDGER_SCHEMA.all_columns
REQUIRED_COLUMNS = LEDGER_SCHEMA.required_columns
NULLABLE_COLUMNS = LEDGER_SCHEMA.nullable_columns
UNIQUE_COLUMNS = LEDGER_SCHEMA.unique_columns
SCHEMA_BY_NAME = LEDGER_SCHEMA.by_name


# --- Validation report -------------------------------------------------------


@dataclass
class ValidationReport:
    schema_name: str = ""
    row_count: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    type_mismatches: dict[str, str] = field(default_factory=dict)
    null_rates: dict[str, float] = field(default_factory=dict)
    null_violations: dict[str, float] = field(default_factory=dict)
    duplicate_key_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True only if every required column is present, matches its
        expected type, has no nulls, and unique columns have no duplicates.
        """
        return (
            not self.missing_columns
            and not self.type_mismatches
            and not self.null_violations
            and not self.duplicate_key_counts
        )

    def summary(self) -> str:
        lines = [f"schema: {self.schema_name}", f"rows: {self.row_count}", f"valid: {self.is_valid}"]
        if self.missing_columns:
            lines.append(f"missing columns: {self.missing_columns}")
        if self.unexpected_columns:
            lines.append(f"unexpected columns: {self.unexpected_columns}")
        if self.type_mismatches:
            lines.append(f"type mismatches: {self.type_mismatches}")
        if self.null_violations:
            lines.append(f"nulls in required columns: {self.null_violations}")
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


def validate(df: pd.DataFrame, schema: Schema = LEDGER_SCHEMA) -> ValidationReport:
    """Compare df against a Schema and report discrepancies.

    Never coerces or mutates df. Callers decide what to do with a report;
    this function only observes. Defaults to LEDGER_SCHEMA; pass
    schema=OVERHEAD_SCHEMA for the period overhead file.
    """
    report = ValidationReport(schema_name=schema.name, row_count=len(df))

    present = set(df.columns)
    expected = set(schema.all_columns)
    report.missing_columns = [c for c in schema.all_columns if c not in present]
    report.unexpected_columns = sorted(present - expected)

    by_name = schema.by_name
    for col_name, col in by_name.items():
        if col_name not in present:
            continue
        series = df[col_name]
        null_rate = float(series.isna().mean())
        report.null_rates[col_name] = null_rate

        if not col.nullable and null_rate > 0:
            report.null_violations[col_name] = null_rate

        observed = _observed_kind(series.dtype)
        if observed != col.kind:
            report.type_mismatches[col_name] = f"expected {col.kind}, got {series.dtype}"

    for col_name in schema.unique_columns:
        if col_name not in present:
            continue
        dup_count = int(df[col_name].duplicated(keep=False).sum())
        if dup_count:
            report.duplicate_key_counts[col_name] = dup_count

    return report
