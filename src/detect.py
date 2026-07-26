"""Schema detection: decide which pipeline(s) a given file's columns
support - never what conclusion to draw from them.

detect_capabilities(df) inspects the columns actually present (after
mapping known real-world header variants through
schema.COLUMN_ALIASES) and reports:

- whether the margin pipeline is available (direct_cost present, or an
  overhead file supplied separately)
- whether the routing pipeline is available (vehicle_id, origin_id,
  destination_id, and date all present)
- which columns it couldn't map to anything canonical (reported, never
  dropped)

It never fabricates or estimates a missing column to make a pipeline
"available" - absent cost data means the margin pipeline is skipped,
stated plainly, not imputed as zero or guessed at.
"""

from __future__ import annotations

import pandas as pd

from src import schema

ROUTING_REQUIRED = [schema.VEHICLE_ID, schema.ORIGIN_ID, schema.DESTINATION_ID, schema.TRIP_DATE]


def _map_aliases(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    """Returns (mapping, unmapped): mapping is original header -> canonical
    name for every header already canonical or recognized via
    schema.COLUMN_ALIASES; unmapped lists headers that matched neither.
    """
    canonical_set = set(schema.TRIPS_SCHEMA.all_columns) | set(schema.LEDGER_SCHEMA.all_columns)
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for col in columns:
        key = str(col).strip().lower()
        if col in canonical_set:
            mapping[col] = col
        elif key in schema.COLUMN_ALIASES:
            mapping[col] = schema.COLUMN_ALIASES[key]
        else:
            unmapped.append(col)
    return mapping, unmapped


def detect_capabilities(df: pd.DataFrame, overhead_available: bool = False) -> dict:
    """Inspect df's columns and decide which analysis pipeline(s) are
    runnable. overhead_available should be True if a separate overhead
    file was also supplied alongside df - the margin pipeline can run
    off overhead data alone even without direct_cost on every line.

    Returns a dict with: canonical_columns (original -> canonical
    mapping), unmapped_columns, margin_available, routing_available,
    missing_for_margin, missing_for_routing, and a plain-language
    message describing what was found and what will run.
    """
    mapping, unmapped = _map_aliases(list(df.columns))
    canonical_present = set(mapping.values())

    margin_available = schema.DIRECT_COST in canonical_present or overhead_available
    missing_for_margin = [] if margin_available else [schema.DIRECT_COST]

    missing_for_routing = [c for c in ROUTING_REQUIRED if c not in canonical_present]
    routing_available = not missing_for_routing

    if margin_available and routing_available:
        message = (
            "Found cost data and vehicle_id + origin_id + destination_id + date -> "
            "running both margin and routing analysis."
        )
    elif routing_available:
        message = (
            "Found vehicle_id + origin_id + destination_id + date, no cost columns -> "
            "running empty-running/utilisation analysis. Margin analysis skipped (no cost data)."
        )
    elif margin_available:
        message = (
            "Found cost data (direct_cost or an overhead file), missing "
            f"{', '.join(missing_for_routing)} for routing -> running margin analysis. "
            "Routing analysis skipped (missing those columns)."
        )
    else:
        message = (
            "Cannot analyse: missing required columns for either pipeline. "
            f"Margin needs: {schema.DIRECT_COST} (or a separate overhead file). "
            f"Routing needs: {', '.join(ROUTING_REQUIRED)}."
        )

    if unmapped:
        message += f" Unmapped columns (left as-is, not used by either pipeline): {unmapped}."

    return {
        "canonical_columns": mapping,
        "unmapped_columns": unmapped,
        "margin_available": margin_available,
        "routing_available": routing_available,
        "missing_for_margin": missing_for_margin,
        "missing_for_routing": missing_for_routing,
        "message": message,
    }
