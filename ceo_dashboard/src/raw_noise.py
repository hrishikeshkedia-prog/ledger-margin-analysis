"""
Simulates the messiness of a real raw export, for demonstrating that
`clean.py` does real work rather than operating on already-perfect data.

The Stage 1 generator produces clean data by construction (there's no
reason to synthesize noise into the canonical dataset every other stage
relies on). But a real order export always has some mess: duplicate rows
from a webhook retry, a few missing cost fields from a data-entry gap,
inconsistent text casing from a platform migration, a handful of clearly
wrong values (negative quantity, zero price) from an integration bug.

`make_messy(orders_df, rng)` takes the clean canonical `orders` table and
returns a copy with exactly these issues introduced, at realistic (small)
rates, so `clean.py`'s pipeline has genuine defects to find, fix, and log.
This function is never used to build the canonical dataset -- only to
demonstrate cleaning in the notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema


def make_messy(orders_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = orders_df.copy()

    # 1. Duplicate rows (~0.4%): simulates a webhook/export retry writing
    #    the same order line twice.
    n_dupe = max(1, int(len(df) * 0.004))
    dupe_rows = df.sample(n=n_dupe, random_state=rng.integers(0, 2**31 - 1))
    df = pd.concat([df, dupe_rows], ignore_index=True)

    # 2. Missing unit_cogs (~0.5%): simulates a product-catalog gap where
    #    cost wasn't entered for a line.
    n_missing_cogs = int(len(df) * 0.005)
    missing_idx = rng.choice(df.index, size=n_missing_cogs, replace=False)
    df.loc[missing_idx, schema.UNIT_COGS] = np.nan

    # 3. Inconsistent category casing/whitespace (~2%): simulates a platform
    #    migration or manual CSV edit.
    n_casing = int(len(df) * 0.02)
    casing_idx = rng.choice(df.index, size=n_casing, replace=False)
    df.loc[casing_idx, schema.CATEGORY] = df.loc[casing_idx, schema.CATEGORY].str.upper() + "  "

    # 4. Invalid economics (~0.2%): simulates an integration bug -- a
    #    negative quantity (return processed as a negative-qty order line
    #    instead of via the return flag) or a zero price (free-gift line
    #    mis-tagged as a sale).
    n_invalid = max(1, int(len(df) * 0.002))
    invalid_idx = rng.choice(df.index, size=n_invalid, replace=False)
    half = len(invalid_idx) // 2
    df.loc[invalid_idx[:half], schema.QUANTITY] = -1
    df.loc[invalid_idx[half:], schema.UNIT_PRICE] = 0.0

    return df.sample(frac=1, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)
