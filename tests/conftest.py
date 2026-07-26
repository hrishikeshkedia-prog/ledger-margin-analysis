import itertools

import pandas as pd
import pytest

from src import clean, schema
from src.synth import SynthConfig, TripsConfig, generate_ledger, generate_overhead, generate_trips


@pytest.fixture(scope="session")
def synth_config() -> SynthConfig:
    # Small but non-trivial: enough rows/months/customers to exercise
    # dedup, null handling, and multi-period grouping without a slow suite.
    return SynthConfig(rows=500, seed=7, n_customers=10, n_locations=6, n_routes=12, n_vendors=5, n_months=6)


@pytest.fixture(scope="session")
def raw_ledger(synth_config: SynthConfig) -> pd.DataFrame:
    return generate_ledger(synth_config)


@pytest.fixture(scope="session")
def raw_overhead(synth_config: SynthConfig) -> pd.DataFrame:
    return generate_overhead(synth_config)


@pytest.fixture(scope="session")
def cleaned_ledger(raw_ledger: pd.DataFrame) -> pd.DataFrame:
    df, _ = clean.clean(raw_ledger)
    return df


@pytest.fixture(scope="session")
def trips_config() -> TripsConfig:
    return TripsConfig(rows=200, seed=11, n_vehicles=6, n_locations=6, n_months=3)


@pytest.fixture(scope="session")
def raw_trips(trips_config: TripsConfig) -> pd.DataFrame:
    return generate_trips(trips_config)


@pytest.fixture(scope="session")
def distances_fixture(trips_config: TripsConfig) -> pd.DataFrame:
    # Full matrix over the fixture's own location pool, so the routing
    # integration test never trips over a missing pair by accident.
    locations = [f"LOC{idx:03d}" for idx in range(1, trips_config.n_locations + 1)]
    rows = [
        {schema.ORIGIN_ID: a, schema.DESTINATION_ID: b, schema.DISTANCE_KM: 50.0 + 10 * i}
        for i, (a, b) in enumerate(itertools.permutations(locations, 2))
    ]
    return pd.DataFrame(rows)
