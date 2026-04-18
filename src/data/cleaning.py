from __future__ import annotations

import pandas as pd


NUMERIC_CANDIDATES = [
	"lead_time",
	"stays_in_weekend_nights",
	"stays_in_week_nights",
	"adults",
	"children",
	"babies",
	"booking_changes",
	"days_in_waiting_list",
	"adr",
	"required_car_parking_spaces",
	"total_of_special_requests",
	"days_to_next_holiday",
	"days_from_last_holiday",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Clean raw dataframe while preserving modeling signal."""
	cleaned = df.copy()
	cleaned = cleaned.drop_duplicates().reset_index(drop=True)

	for col in NUMERIC_CANDIDATES:
		if col in cleaned.columns:
			cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

	numeric_cols = cleaned.select_dtypes(include=["number"]).columns
	for col in numeric_cols:
		cleaned[col] = cleaned[col].fillna(cleaned[col].median())

	cat_cols = cleaned.select_dtypes(include=["object", "category"]).columns
	for col in cat_cols:
		cleaned[col] = cleaned[col].astype(str).str.strip().str.lower().fillna("unknown")
		cleaned[col] = cleaned[col].replace({"": "unknown", "nan": "unknown"})

	if {"adults", "children", "babies"}.issubset(cleaned.columns):
		total_guests = cleaned["adults"] + cleaned["children"] + cleaned["babies"]
		cleaned = cleaned[total_guests > 0].copy()

	return cleaned.reset_index(drop=True)

