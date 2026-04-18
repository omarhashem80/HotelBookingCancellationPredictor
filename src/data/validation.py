from __future__ import annotations

from typing import Any

import pandas as pd


def _outlier_summary(df: pd.DataFrame) -> dict[str, int]:
	summary: dict[str, int] = {}
	numeric_cols = df.select_dtypes(include=["number"]).columns
	for col in numeric_cols:
		q1 = df[col].quantile(0.25)
		q3 = df[col].quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		outliers = ((df[col] < lower) | (df[col] > upper)).sum()
		summary[col] = int(outliers)
	return summary


def validate_dataframe(df: pd.DataFrame, target_col: str = "is_canceled") -> dict[str, Any]:
	"""Run a compact set of data quality checks."""
	schema_issues: list[str] = []
	if target_col not in df.columns:
		schema_issues.append(f"Missing target column: {target_col}")

	result = {
		"missing_values": df.isna().sum().to_dict(),
		"duplicates": int(df.duplicated().sum()),
		"schema_issues": schema_issues,
		"class_balance": df[target_col].value_counts(normalize=True).to_dict()
		if target_col in df.columns
		else {},
		"outlier_summary": _outlier_summary(df),
	}
	return result

