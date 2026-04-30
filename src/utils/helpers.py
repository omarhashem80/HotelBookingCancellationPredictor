import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """Set random seed for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)


def dataframe_overview(df: pd.DataFrame) -> dict:
    """Return a compact dataframe summary for logging or debugging."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Measure execution time for a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label} took {elapsed:.2f}s")
