from __future__ import annotations

import pandas as pd

from src.features.build_features import build_features


def test_total_guests_feature() -> None:
	df = pd.DataFrame(
		{
			"adults": [2],
			"children": [1],
			"babies": [0],
			"stays_in_weekend_nights": [1],
			"stays_in_week_nights": [2],
			"reserved_room_type": ["A"],
			"assigned_room_type": ["B"],
			"lead_time": [40],
			"days_to_next_holiday": [3],
			"days_from_last_holiday": [6],
		}
	)
	out = build_features(df)
	assert out.loc[0, "total_guests"] == 3


def test_total_nights_feature() -> None:
	df = pd.DataFrame(
		{
			"adults": [1],
			"children": [0],
			"babies": [0],
			"stays_in_weekend_nights": [2],
			"stays_in_week_nights": [3],
			"lead_time": [10],
			"reserved_room_type": ["A"],
			"assigned_room_type": ["A"],
			"days_to_next_holiday": [2],
			"days_from_last_holiday": [4],
		}
	)
	out = build_features(df)
	assert out.loc[0, "total_nights"] == 5
	assert out.loc[0, "room_change"] == 0

