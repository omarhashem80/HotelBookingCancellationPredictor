from __future__ import annotations

import pandas as pd


def _add_total_guests(df: pd.DataFrame) -> pd.DataFrame:
	if {"adults", "children", "babies"}.issubset(df.columns):
		df["total_guests"] = df["adults"] + df["children"] + df["babies"]
	return df


def _add_total_nights(df: pd.DataFrame) -> pd.DataFrame:
	if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
		df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
	return df


def _add_room_change_flag(df: pd.DataFrame) -> pd.DataFrame:
	if {"reserved_room_type", "assigned_room_type"}.issubset(df.columns):
		df["room_change"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
	return df


def _add_lead_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
	if "lead_time" in df.columns:
		df["lead_time_bucket"] = pd.cut(
			df["lead_time"],
			bins=[-1, 7, 30, 90, 365, float("inf")],
			labels=["0_7", "8_30", "31_90", "91_365", "365_plus"],
		).astype(str)
	return df


def _add_holiday_distance(df: pd.DataFrame) -> pd.DataFrame:
	if {"days_to_next_holiday", "days_from_last_holiday"}.issubset(df.columns):
		df["holiday_distance_min"] = df[["days_to_next_holiday", "days_from_last_holiday"]].min(
			axis=1
		)
	return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Create reproducible features used by downstream models."""
	featured = df.copy()
	featured = _add_total_guests(featured)
	featured = _add_total_nights(featured)
	featured = _add_room_change_flag(featured)
	featured = _add_lead_time_bucket(featured)
	featured = _add_holiday_distance(featured)
	return featured

