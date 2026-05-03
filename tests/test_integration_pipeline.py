import pandas as pd
import numpy as np

from src.data.cleaning import clean_data
from src.features.build_features import build_features
from src.models.trainer import spliter, train_single_model

n = 200
rng = np.random.default_rng(0)
rows = np.arange(n)

def test_end_to_end_pipeline_small_dataframe() -> None:

    df = pd.DataFrame(
        {
            "hotel": np.where(rows % 2 == 0, "City Hotel", "Resort Hotel"),
            "meal": np.resize(["BB", "HB", "SC", "FB"], n),
            "country": np.resize(["PRT", "GBR", "ESP", "FRA", "DEU"], n),
            "market_segment": np.resize(["Online TA", "Offline TA/TO", "Direct", "Corporate"], n),
            "distribution_channel": np.resize(["TA/TO", "Direct", "Corporate"], n),
            "reserved_room_type": np.resize(["A", "B", "C", "D"], n),
            "assigned_room_type": np.resize(["A", "B", "C", "D"], n),
            "deposit_type": np.resize(["No Deposit", "Non Refund", "Refundable"], n),
            "customer_type": np.resize(["Transient", "Contract", "Transient-Party", "Group"], n),
            "agent": np.resize(["9", "240", "None", "14"], n),
            "is_repeated_guest": rows % 2,
            "is_holiday": rows % 3 == 0,
            "is_canceled": rows % 2,
            "lead_time": rng.integers(1, 300, size=n),
            "stays_in_weekend_nights": rng.integers(0, 4, size=n),
            "stays_in_week_nights": rng.integers(1, 6, size=n),
            "adults": rng.integers(1, 4, size=n),
            "children": rng.integers(0, 3, size=n),
            "babies": [0] * n,
            "previous_cancellations": rng.integers(0, 5, size=n),
            "previous_bookings_not_canceled": rng.integers(0, 10, size=n),
            "booking_changes": rng.integers(0, 4, size=n),
            "days_in_waiting_list": rng.integers(0, 50, size=n),
            "required_car_parking_spaces": rng.integers(0, 2, size=n),
            "total_of_special_requests": rng.integers(0, 5, size=n),
            "days_to_next_holiday": rng.integers(1, 30, size=n),
            "days_from_last_holiday": rng.integers(1, 30, size=n),
            "adr": 80.0 + rng.integers(0, 120, size=n),
            "arrival_date": pd.date_range("2015-01-01", periods=n, freq="D"),
            "reservation_status_date": pd.date_range("2015-01-02", periods=n, freq="D"),
            "reservation_status": np.resize(["Check-Out", "Canceled"], n),
        }
    )

    clean = clean_data(df)
    featured = build_features(clean)
    # ensure is_canceled is int so group_rare_categories (min_freq=900) doesn't
    # collapse both classes to "Other" on this small dataset
    featured["is_canceled"] = featured["is_canceled"].astype(int)

    X_train, X_test, y_train, y_test = spliter(featured)
    result = train_single_model(X_train, y_train, X_test, y_test, model_name="baseline", cv_splits=2)
    assert result.metrics["accuracy"] >= 0.0
    assert len(result.predictions) == len(result.y_true)
