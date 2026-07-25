"""Cleaning pipeline entry point.

Reads a raw (or synthetic) ledger export and the period overhead file,
runs the ledger through src/clean.py, validates both against
src/schema.py, and writes the results to data/processed/ for
src/margin.py, src/slice.py, src/confounder.py, and the dashboard to
read. Fails loudly (raises) if the cleaned ledger or the overhead file
doesn't pass schema.validate() rather than writing output that looks
fine but isn't.

CLI:
    python -m src.pipeline --in data/synthetic/ledger.csv \\
        --overhead-in data/synthetic/overhead.csv --out-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import clean, schema


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="ledger_path", type=Path, default=Path("data/synthetic/ledger.csv"))
    parser.add_argument(
        "--overhead-in", dest="overhead_path", type=Path, default=Path("data/synthetic/overhead.csv")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run(args.ledger_path, args.overhead_path, args.out_dir)


if __name__ == "__main__":
    main()
