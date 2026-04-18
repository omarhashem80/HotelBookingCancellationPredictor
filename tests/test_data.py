from __future__ import annotations

import pandas as pd

from src.data.cleaning import clean_data
from src.data.ingestion import merge_datasets
from src.data.validation import validate_dataframe


def test_merge_datasets_left_join() -> None:
	left = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
	right = pd.DataFrame({"id": [1], "y": [99]})
	merged = merge_datasets(left, right, on="id", how="left")
	assert "y" in merged.columns
	assert pd.isna(merged.loc[1, "y"])


def test_clean_data_removes_invalid_guest_rows() -> None:
	df = pd.DataFrame(
		{
			"adults": [0, 2],
			"children": [0, 0],
			"babies": [0, 0],
			"lead_time": [1, 2],
			"is_canceled": [0, 1],
		}
	)
	cleaned = clean_data(df)
	assert len(cleaned) == 1


def test_validate_dataframe_contains_required_sections() -> None:
	df = pd.DataFrame({"is_canceled": [0, 1, 1], "lead_time": [10, 12, 11]})
	report = validate_dataframe(df)
	assert "missing_values" in report
	assert "duplicates" in report
	assert "schema_issues" in report
	assert "class_balance" in report

