import pytest
import pandas as pd
import numpy as np


@pytest.fixture(params=[60, 120])
def toy_dataset(request) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "lead_time": rng.integers(1, 200, size=request.param),
            "adr": rng.normal(100, 20, size=request.param),
            "hotel": rng.choice(["resort hotel", "city hotel"], size=request.param),
            "adults": rng.integers(1, 3, size=request.param),
            "children": rng.integers(0, 2, size=request.param),
            "babies": 0,
            "stays_in_weekend_nights": rng.integers(0, 3, size=request.param),
            "stays_in_week_nights": rng.integers(1, 5, size=request.param),
            "reserved_room_type": rng.choice(["A", "B"], size=request.param),
            "assigned_room_type": rng.choice(["A", "B"], size=request.param),
            "days_to_next_holiday": rng.integers(0, 20, size=request.param),
            "days_from_last_holiday": rng.integers(0, 20, size=request.param),
            "is_canceled": rng.integers(0, 2, size=request.param),
            "reservation_status_date": "2015-07-02",
        }
    )
    return df
