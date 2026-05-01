from pathlib import Path

import pandas as pd
from loguru import logger


def load_primary_dataset(path: str | Path) -> pd.DataFrame:
    """Load the main hotel bookings dataset."""
    df = pd.read_csv(path)
    logger.info("Loaded primary dataset: rows={}, cols={}, path={}", df.shape[0], df.shape[1], path)
    return df


def load_secondary_dataset(path: str | Path) -> pd.DataFrame:
    """Load an optional secondary dataset for enrichment."""
    df = pd.read_csv(path)
    logger.info(
        "Loaded secondary dataset: rows={}, cols={}, path={}", df.shape[0], df.shape[1], path
    )
    return df


def merge_datasets(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    on: list[str] | str,
    how: str = "left",
) -> pd.DataFrame:
    """Merge primary and secondary sources."""
    merged = primary_df.merge(secondary_df, on=on, how=how)
    logger.info("Merged datasets: rows={}, cols={}, how={}", merged.shape[0], merged.shape[1], how)
    return merged


def load_and_merge(
    primary_path: str | Path,
    secondary_path: str | Path | None = None,
    merge_on: list[str] | str | None = None,
) -> pd.DataFrame:
    """Load main data and optionally merge with a second source."""
    primary_df = load_primary_dataset(primary_path)
    if not secondary_path:
        return primary_df

    secondary_df = load_secondary_dataset(secondary_path)
    if merge_on is None:
        common_cols = sorted(set(primary_df.columns).intersection(secondary_df.columns))
        if not common_cols:
            raise ValueError("No common columns found for dataset merge.")
        merge_on = common_cols[0]
    logger.info("Merging datasets on column={}", merge_on)
    return merge_datasets(primary_df, secondary_df, on=merge_on)
