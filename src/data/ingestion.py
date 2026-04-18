from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_primary_dataset(path: str | Path) -> pd.DataFrame:
	"""Load the main hotel bookings dataset."""
	return pd.read_csv(path)


def load_secondary_dataset(path: str | Path) -> pd.DataFrame:
	"""Load an optional secondary dataset for enrichment."""
	return pd.read_csv(path)


def merge_datasets(
	primary_df: pd.DataFrame,
	secondary_df: pd.DataFrame,
	on: list[str] | str,
	how: str = "left",
) -> pd.DataFrame:
	"""Merge primary and secondary sources."""
	return primary_df.merge(secondary_df, on=on, how=how)


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
	return merge_datasets(primary_df, secondary_df, on=merge_on)

