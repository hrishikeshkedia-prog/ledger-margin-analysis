"""Pipeline entry point for both the cost/margin and routing pipelines.

Cost/margin (unchanged): reads a raw (or synthetic) ledger export and
the period overhead file, runs the ledger through src/clean.py,
validates both against src/schema.py, and writes the results to
data/processed/ for src/margin.py, src/slice.py, src/confounder.py, and
the dashboard to read.

Routing (additive): reads a trips export and a distance matrix, runs
the trips through src/clean.py's clean_trips, validates both against
src/schema.py, and writes empty-leg/utilisation results to
data/processed/ for the dashboard and notebook to read.

Both fail loudly (raise) if their inputs don't pass schema.validate()
rather than writing output that looks fine but isn't.

CLI:
    python -m src.pipeline --in data/synthetic/ledger.csv \\
        --overhead-in data/synthetic/overhead.csv --out-dir data/processed

    python -m src.pipeline --trips data/synthetic/trips.csv \\
        --distances data/reference/distances.csv --out-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import clean, routing, schema


def _string_dtypes(schema_obj: schema.Schema) -> dict[str, str]:
    # A column that's entirely null (e.g. an empty description column)
    # loads as float64 by pandas' inference rather than string, which
    # would then fail schema.validate() for a reason that has nothing
    # to do with real data quality. Pin string-kind columns explicitly
    # at read time instead of loosening the validator itself.
    return {col.name: "string" for col in schema_obj.columns if col.kind == schema.STRING}


def run(ledger_path: Path, overhead_path: Path, out_dir: Path) -> None:
    raw = pd.read_csv(ledger_path, dtype=_string_dtypes(schema.LEDGER_SCHEMA))
    cleaned, decision_log = clean.clean(raw)

    ledger_report = schema.validate(cleaned)
    if not ledger_report.is_valid:
        raise ValueError(f"cleaned ledger failed schema validation:\n{ledger_report.summary()}")

    overhead = pd.read_csv(
        overhead_path, parse_dates=[schema.PERIOD_MONTH], dtype=_string_dtypes(schema.OVERHEAD_SCHEMA)
    )
    overhead_report = schema.validate(overhead, schema=schema.OVERHEAD_SCHEMA)
    if not overhead_report.is_valid:
        raise ValueError(f"overhead file failed schema validation:\n{overhead_report.summary()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_dir / "ledger.csv", index=False)
    overhead.to_csv(out_dir / "overhead.csv", index=False)
    clean.decision_log_to_frame(decision_log).to_csv(out_dir / "decision_log.csv", index=False)

    print(f"wrote {len(cleaned)} cleaned ledger rows to {out_dir / 'ledger.csv'}")
    print(f"wrote {len(overhead)} overhead rows to {out_dir / 'overhead.csv'}")
    print(f"wrote {len(decision_log)} decision log entries to {out_dir / 'decision_log.csv'}")


def run_routing(trips_path: Path, distances_path: Path, out_dir: Path) -> None:
    raw = pd.read_csv(trips_path, dtype=_string_dtypes(schema.TRIPS_SCHEMA))
    cleaned, decision_log = clean.clean_trips(raw)

    trips_report = schema.validate(cleaned, schema=schema.TRIPS_SCHEMA)
    if not trips_report.is_valid:
        raise ValueError(f"cleaned trips failed schema validation:\n{trips_report.summary()}")

    distances = pd.read_csv(distances_path, dtype=_string_dtypes(schema.DISTANCE_SCHEMA))
    distances_report = schema.validate(distances, schema=schema.DISTANCE_SCHEMA)
    if not distances_report.is_valid:
        raise ValueError(f"distance matrix failed schema validation:\n{distances_report.summary()}")

    empty_legs = routing.detect_empty_legs(cleaned, distances)
    vehicle_km = routing.summarize_vehicle_km(cleaned, distances)
    util_df = routing.utilisation(cleaned)
    util_summary = routing.utilisation_summary(util_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_dir / "trips.csv", index=False)
    empty_legs.to_csv(out_dir / "empty_legs.csv", index=False)
    vehicle_km.to_csv(out_dir / "vehicle_km_summary.csv", index=False)
    util_summary.to_csv(out_dir / "utilisation_summary.csv", index=False)
    clean.decision_log_to_frame(decision_log).to_csv(out_dir / "trips_decision_log.csv", index=False)

    print(f"wrote {len(cleaned)} cleaned trip rows to {out_dir / 'trips.csv'}")
    print(f"wrote {len(empty_legs)} empty legs to {out_dir / 'empty_legs.csv'}")
    print(f"wrote per-vehicle km summary ({len(vehicle_km)} vehicles) to {out_dir / 'vehicle_km_summary.csv'}")
    print(f"wrote utilisation summary ({len(util_summary)} vehicles) to {out_dir / 'utilisation_summary.csv'}")
    print(f"wrote {len(decision_log)} decision log entries to {out_dir / 'trips_decision_log.csv'}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="ledger_path", type=Path, default=None)
    parser.add_argument("--overhead-in", dest="overhead_path", type=Path, default=None)
    parser.add_argument("--trips", dest="trips_path", type=Path, default=None)
    parser.add_argument("--distances", dest="distances_path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.trips_path is not None:
        distances_path = args.distances_path or Path("data/reference/distances.csv")
        run_routing(args.trips_path, distances_path, args.out_dir)
        return

    ledger_path = args.ledger_path or Path("data/synthetic/ledger.csv")
    overhead_path = args.overhead_path or Path("data/synthetic/overhead.csv")
    run(ledger_path, overhead_path, args.out_dir)


if __name__ == "__main__":
    main()
