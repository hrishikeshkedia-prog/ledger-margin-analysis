import pandas as pd
import pytest

from src import clean
from src.synth import SynthConfig, generate_ledger, generate_overhead


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
