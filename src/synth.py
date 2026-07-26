"""Synthetic data generator for both pipelines this project supports.

Two independent modes:

- ``--mode ledger`` (default): two CSVs matching ``schema.LEDGER_SCHEMA``
  and ``schema.OVERHEAD_SCHEMA``, seeded with the kind of mess a real
  accounting-software export tends to have: inconsistent date formats,
  whitespace/casing variants in IDs, a few negative and zero-revenue
  rows, ~5% missing direct_cost, and occasional duplicate txn_id.
- ``--mode trips``: one CSV matching ``schema.TRIPS_SCHEMA`` (no cost
  columns) for the routing/empty-running pipeline - a vehicle ledger
  with the same category of mess (inconsistent dates, whitespace/
  casing in IDs, a few duplicates), scaled to a few hundred trips
  rather than tens of thousands of invoice lines.

Neither mode plants a discoverable pattern or anomaly - variation is
pure noise (plus mild seasonality in ledger mode). They exist to
exercise the rest of each pipeline's code paths, not to produce
insights. In trips mode in particular: origin and destination are
drawn independently and uniformly at random for every trip, with no
memory of where the vehicle's previous trip ended - any empty running
that shows up is an accident of the random draw, never designed in.

CLI:
    python -m src.synth --rows 20000 --seed 42 --out data/synthetic/ledger.csv
    python -m src.synth --mode trips --rows 400 --seed 42 --out data/synthetic/trips.csv
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import schema

# (line_item, unit of measure) - uom is None for fee lines that have no
# physical unit, which is a legitimate case clean.py must handle, not a bug.
LINE_ITEMS = [
    ("freight_charge", "trips"),
    ("fuel_surcharge", "trips"),
    ("handling_fee", "cartons"),
    ("bulk_transport", "tonnes"),
    ("tanker_transport", "litres"),
    ("storage_fee", None),
    ("documentation_fee", None),
]

COST_CATEGORIES = ["fuel", "tolls", "labor", "packaging", "maintenance"]
OVERHEAD_CATEGORIES = {
    "driver_salaries": 40000.0,
    "emi": 15000.0,
    "rent": 8000.0,
    "office": 4000.0,
}
CUSTOMER_SEGMENTS = ["enterprise", "mid_market", "smb", None]

TXN_TYPE_WEIGHTS = {
    schema.TXN_TYPE_SALE: 0.94,
    schema.TXN_TYPE_CREDIT_NOTE: 0.04,
    schema.TXN_TYPE_CANCELLED: 0.02,
}


@dataclass
class SynthConfig:
    rows: int = 20000
    seed: int = 42
    n_customers: int = 40
    n_locations: int = 12  # pool of origin/destination codes
    n_routes: int = 30  # distinct origin->destination lanes actually used
    n_vendors: int = 15
    n_months: int = 20
    start_month: date = date(2024, 1, 1)

    def __post_init__(self) -> None:
        max_possible_routes = self.n_locations * (self.n_locations - 1)
        if self.n_routes > max_possible_routes:
            raise ValueError(
                f"n_routes ({self.n_routes}) exceeds the number of distinct "
                f"origin->destination pairs available from {self.n_locations} "
                f"locations ({max_possible_routes})"
            )


@dataclass
class TripsConfig:
    rows: int = 400
    seed: int = 42
    n_vehicles: int = 12
    n_locations: int = 10
    n_months: int = 4
    start_month: date = date(2024, 1, 1)


def _month_range(start: date, n_months: int) -> list[date]:
    months = []
    year, month = start.year, start.month
    for _ in range(n_months):
        months.append(date(year, month, 1))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def _build_routes(rng: random.Random, locations: list[str], n_routes: int) -> list[tuple[str, str]]:
    routes: list[tuple[str, str]] = []
    seen = set()
    attempts = 0
    max_attempts = n_routes * 200
    while len(routes) < n_routes:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("could not sample enough distinct routes; check n_locations/n_routes")
        origin, destination = rng.choice(locations), rng.choice(locations)
        if origin == destination or (origin, destination) in seen:
            continue
        seen.add((origin, destination))
        routes.append((origin, destination))
    return routes


def _messy_id(rng: random.Random, value: str) -> str:
    """Occasionally mangle an ID with whitespace or casing - the exact
    mess a hand-edited spreadsheet or inconsistent export produces."""
    roll = rng.random()
    if roll < 0.05:
        return f" {value}"
    if roll < 0.10:
        return f"{value} "
    if roll < 0.15:
        return value.lower()
    if roll < 0.18:
        return value.swapcase()
    return value


def _messy_date(rng: random.Random, d: date | None) -> str | None:
    """Render a date in one of several inconsistent formats."""
    if d is None:
        return None
    fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y"])
    return d.strftime(fmt)


def generate_ledger(cfg: SynthConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    customers = [f"CUST{idx:04d}" for idx in range(1, cfg.n_customers + 1)]
    locations = [f"LOC{idx:03d}" for idx in range(1, cfg.n_locations + 1)]
    vendors = [f"VEND{idx:03d}" for idx in range(1, cfg.n_vendors + 1)]
    months = _month_range(cfg.start_month, cfg.n_months)
    routes = _build_routes(rng, locations, cfg.n_routes)
    txn_types, txn_weights = zip(*TXN_TYPE_WEIGHTS.items())

    rows = []
    for i in range(cfg.rows):
        month = rng.choice(months)
        day = rng.randint(1, 28)
        txn_date = date(month.year, month.month, day)

        # Mild seasonality only - a smooth wave, no planted anomaly.
        seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month.month / 12))

        origin_id, destination_id = rng.choice(routes)
        line_item, uom = rng.choice(LINE_ITEMS)
        if uom is not None and rng.random() < 0.03:
            uom = None  # occasionally dropped in the export even when it applies

        quantity = round(max(0.1, np_rng.normal(loc=10 * seasonal_factor, scale=4)), 2)
        base_rate = np_rng.uniform(5, 50)
        revenue = round(quantity * base_rate, 2)
        if rng.random() < 0.01:
            revenue = -abs(revenue)
        elif rng.random() < 0.01:
            revenue = 0.0

        direct_cost = round(revenue * np_rng.uniform(0.4, 0.8), 2)
        cost_category = rng.choice(COST_CATEGORIES)
        if rng.random() < 0.05:
            direct_cost = None
            cost_category = None

        payment_lag_days = int(np_rng.uniform(0, 90))
        payment_date = txn_date + timedelta(days=payment_lag_days)
        if rng.random() < 0.08:
            payment_date = None  # still outstanding

        rows.append(
            {
                schema.TXN_ID: f"TXN{i:07d}",
                schema.TXN_DATE: txn_date,
                schema.TXN_TYPE: rng.choices(txn_types, weights=txn_weights)[0],
                schema.CUSTOMER_ID: rng.choice(customers),
                schema.CUSTOMER_SEGMENT: rng.choice(CUSTOMER_SEGMENTS),
                schema.ORIGIN_ID: origin_id,
                schema.DESTINATION_ID: destination_id,
                schema.VENDOR_ID: rng.choice(vendors) if rng.random() < 0.4 else None,
                schema.LINE_ITEM: line_item,
                schema.QUANTITY: quantity,
                schema.UOM: uom,
                schema.REVENUE: revenue,
                schema.DIRECT_COST: direct_cost,
                schema.COST_CATEGORY: cost_category,
                schema.INVOICE_ID: f"INV{i // 3:07d}",
                schema.PAYMENT_DATE: payment_date,
            }
        )

    df = pd.DataFrame(rows)

    # Occasional duplicate txn_id: clone a small number of existing rows
    # verbatim (same txn_id), the way a re-exported or double-posted
    # invoice line would show up.
    n_dupes = max(1, int(cfg.rows * 0.002))
    dupe_positions = rng.sample(range(len(df)), min(n_dupes, len(df)))
    df = pd.concat([df, df.iloc[dupe_positions]], ignore_index=True)

    df[schema.CUSTOMER_ID] = df[schema.CUSTOMER_ID].map(lambda v: _messy_id(rng, v))
    df[schema.TXN_DATE] = df[schema.TXN_DATE].map(lambda d: _messy_date(rng, d))
    df[schema.PAYMENT_DATE] = df[schema.PAYMENT_DATE].map(lambda d: _messy_date(rng, d))

    return df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)


def generate_overhead(cfg: SynthConfig) -> pd.DataFrame:
    # Separate seed offset so overhead noise is independent of ledger noise
    # but still fully determined by cfg.seed.
    rng = random.Random(cfg.seed + 1)
    months = _month_range(cfg.start_month, cfg.n_months)

    rows = []
    for month in months:
        for category, base_amount in OVERHEAD_CATEGORIES.items():
            noise = rng.uniform(0.9, 1.1)
            rows.append(
                {
                    schema.PERIOD_MONTH: month,
                    schema.COST_CATEGORY: category,
                    schema.AMOUNT: round(base_amount * noise, 2),
                    schema.DESCRIPTION: None,
                }
            )
    return pd.DataFrame(rows)


def generate_trips(cfg: TripsConfig) -> pd.DataFrame:
    """Vehicle trip ledger for the routing pipeline: txn_id, date,
    vehicle_id, origin_id, destination_id, revenue - no cost columns,
    matching the real freight export's shape.

    origin_id and destination_id are drawn independently and uniformly
    at random for every trip (rejecting only origin == destination,
    since a trip has to go somewhere) - deliberately with no memory of
    where the same vehicle's previous trip ended. Any empty running
    that emerges from that is an accident of the random draw, not a
    planted pattern.
    """
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    vehicles = [f"VEH{idx:03d}" for idx in range(1, cfg.n_vehicles + 1)]
    locations = [f"LOC{idx:03d}" for idx in range(1, cfg.n_locations + 1)]
    day_span = cfg.n_months * 30

    rows = []
    for i in range(cfg.rows):
        trip_date = cfg.start_month + timedelta(days=rng.randint(0, day_span - 1))
        origin, destination = rng.sample(locations, 2)
        revenue = round(np_rng.uniform(5000, 40000), 2)

        rows.append(
            {
                schema.TXN_ID: f"TRIP{i:04d}",
                schema.TRIP_DATE: trip_date,
                schema.VEHICLE_ID: rng.choice(vehicles),
                schema.ORIGIN_ID: origin,
                schema.DESTINATION_ID: destination,
                schema.REVENUE: revenue,
            }
        )

    df = pd.DataFrame(rows)

    # Occasional duplicate txn_id, same reasoning as generate_ledger.
    n_dupes = max(1, int(cfg.rows * 0.01))
    dupe_positions = rng.sample(range(len(df)), min(n_dupes, len(df)))
    df = pd.concat([df, df.iloc[dupe_positions]], ignore_index=True)

    df[schema.VEHICLE_ID] = df[schema.VEHICLE_ID].map(lambda v: _messy_id(rng, v))
    df[schema.TRIP_DATE] = df[schema.TRIP_DATE].map(lambda d: _messy_date(rng, d))

    return df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ledger", "trips"], default="ledger")
    parser.add_argument("--seed", type=int, default=42)
    # Shared numeric knobs - default is None here and resolved per-mode in
    # main(), since "rows"/"locations"/"months" mean sensible but different
    # defaults for a 20k-line invoice ledger vs. a few-hundred-trip log.
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--locations", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    # ledger-mode only
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--routes", type=int, default=30)
    parser.add_argument("--vendors", type=int, default=15)
    # trips-mode only
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--overhead-out",
        type=Path,
        default=None,
        help="ledger mode only; defaults to overhead.csv next to --out",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.mode == "trips":
        cfg = TripsConfig(
            rows=args.rows if args.rows is not None else 400,
            seed=args.seed,
            n_vehicles=args.vehicles,
            n_locations=args.locations if args.locations is not None else 10,
            n_months=args.months if args.months is not None else 4,
        )
        trips_df = generate_trips(cfg)

        out_path: Path = args.out or Path("data/synthetic/trips.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trips_df.to_csv(out_path, index=False)
        print(f"wrote {len(trips_df)} trip rows to {out_path}")
        return

    cfg = SynthConfig(
        rows=args.rows if args.rows is not None else 20000,
        seed=args.seed,
        n_customers=args.customers,
        n_locations=args.locations if args.locations is not None else 12,
        n_routes=args.routes,
        n_vendors=args.vendors,
        n_months=args.months if args.months is not None else 20,
    )

    ledger_df = generate_ledger(cfg)
    overhead_df = generate_overhead(cfg)

    out_path: Path = args.out or Path("data/synthetic/ledger.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_df.to_csv(out_path, index=False)

    overhead_path: Path = args.overhead_out or (out_path.parent / "overhead.csv")
    overhead_path.parent.mkdir(parents=True, exist_ok=True)
    overhead_df.to_csv(overhead_path, index=False)

    print(f"wrote {len(ledger_df)} ledger rows to {out_path}")
    print(f"wrote {len(overhead_df)} overhead rows to {overhead_path}")


if __name__ == "__main__":
    main()
