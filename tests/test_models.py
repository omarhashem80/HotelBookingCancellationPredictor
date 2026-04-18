from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.trainer import train_single_model


def _toy_dataset(n: int = 120) -> pd.DataFrame:
	rng = np.random.default_rng(42)
	df = pd.DataFrame(
		{
			"lead_time": rng.integers(1, 200, size=n),
			"adr": rng.normal(100, 20, size=n),
			"hotel": rng.choice(["resort hotel", "city hotel"], size=n),
			"adults": rng.integers(1, 3, size=n),
			"children": rng.integers(0, 2, size=n),
			"babies": 0,
			"stays_in_weekend_nights": rng.integers(0, 3, size=n),
			"stays_in_week_nights": rng.integers(1, 5, size=n),
			"reserved_room_type": rng.choice(["A", "B"], size=n),
			"assigned_room_type": rng.choice(["A", "B"], size=n),
			"days_to_next_holiday": rng.integers(0, 20, size=n),
			"days_from_last_holiday": rng.integers(0, 20, size=n),
			"is_canceled": rng.integers(0, 2, size=n),
		}
	)
	return df


def test_model_training_runs_baseline() -> None:
	df = _toy_dataset()
	result = train_single_model(df, model_name="baseline", cv=2)
	assert "accuracy" in result.metrics
	assert len(result.predictions) > 0


def test_prediction_shape_matches_holdout() -> None:
	df = _toy_dataset()
	result = train_single_model(df, model_name="logistic", cv=2)
	assert len(result.predictions) == len(result.y_true)

