from __future__ import annotations

import pandas as pd

from src.data.cleaning import clean_data
from src.data.validation import validate_dataframe
from src.features.build_features import build_features
from src.models.trainer import train_single_model


def test_end_to_end_pipeline_small_dataframe() -> None:
    df = pd.DataFrame(
        {
            "hotel": ["resort", "city", "city", "resort", "city", "resort"],
            "is_canceled": [0, 1, 0, 1, 0, 1],
            "lead_time": [10, 100, 30, 90, 12, 75],
            "stays_in_weekend_nights": [1, 2, 1, 2, 0, 1],
            "stays_in_week_nights": [2, 3, 2, 2, 2, 3],
            "adults": [2, 2, 1, 2, 1, 2],
            "children": [0, 1, 0, 0, 1, 0],
            "babies": [0, 0, 0, 0, 0, 0],
            "reserved_room_type": ["A", "A", "B", "A", "B", "A"],
            "assigned_room_type": ["A", "B", "B", "A", "A", "A"],
            "days_to_next_holiday": [1, 5, 3, 10, 2, 8],
            "days_from_last_holiday": [6, 2, 4, 5, 3, 1],
            "adr": [80, 140, 100, 120, 90, 130],
        }
    )

    report = validate_dataframe(df)
    assert "class_balance" in report

    clean = clean_data(df)
    featured = build_features(clean)

    result = train_single_model(featured, model_name="baseline", cv=2)
    assert result.metrics["accuracy"] >= 0.0
    assert len(result.predictions) == len(result.y_true)
