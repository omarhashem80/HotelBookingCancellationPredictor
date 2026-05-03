import numpy as np
import pandas as pd
import pytest

from src.models.trainer import (
    TrainingResult,
    enforce_schema,
    get_model_registry,
    run_model_pipeline,
    train_single_model,
)


def _repeated(values, n):
    return np.resize(np.array(values), n)


@pytest.fixture
def hotel_dataset():
    n = 60
    rows = np.arange(n)

    return pd.DataFrame(
        {
            "hotel": ["City Hotel"] * n,
            "meal": ["BB"] * n,
            "country": ["PRT"] * n,
            "market_segment": ["Online TA"] * n,
            "distribution_channel": ["TA/TO"] * n,
            "reserved_room_type": ["A"] * n,
            "assigned_room_type": ["A"] * n,
            "deposit_type": ["No Deposit"] * n,
            "customer_type": ["Transient"] * n,
            "agent": ["9"] * n,
            "is_repeated_guest": _repeated([0, 1], n),
            "is_holiday": _repeated([0, 1], n),
            "is_canceled": _repeated([0, 1], n),
            "arrival_date": pd.date_range("2015-07-01", periods=n, freq="D"),
            "reservation_status_date": pd.date_range("2015-07-02", periods=n, freq="D"),
            "arrival_date_year": [2015] * n,
            "arrival_date_month": _repeated([7, 8], n),
            "arrival_date_week_number": (rows % 52) + 1,
            "arrival_date_day_of_month": (rows % 28) + 1,
            "reservation_status_year": [2015] * n,
            "reservation_status_month": _repeated([7, 8], n),
            "reservation_status_day": ((rows + 1) % 28) + 1,
            "lead_time": rows + 1,
            "stays_in_weekend_nights": rows % 3,
            "stays_in_week_nights": (rows % 5) + 1,
            "adults": _repeated([1, 2], n),
            "children": rows % 2,
            "babies": [0] * n,
            "previous_cancellations": rows % 2,
            "previous_bookings_not_canceled": rows % 4,
            "booking_changes": rows % 3,
            "days_in_waiting_list": rows % 7,
            "required_car_parking_spaces": rows % 2,
            "total_of_special_requests": rows % 5,
            "days_to_next_holiday": (rows % 10) + 1,
            "days_from_last_holiday": (rows % 8) + 1,
            "adr": 80.0 + rows,
        }
    )


def test_enforce_schema_casts_configured_columns(hotel_dataset):
    df = hotel_dataset.astype({"lead_time": "string", "is_canceled": "string"})

    result = enforce_schema(df, supress_rare=True)

    assert pd.api.types.is_numeric_dtype(result["lead_time"])
    assert result["is_canceled"].dtype == "int8"
    assert pd.api.types.is_datetime64_any_dtype(result["arrival_date"])
    assert set(result["hotel"].astype(str)) == {"Other"}


# def test_group_rare_categories_keeps_frequent_values_and_groups_rare_values():
#     df = pd.DataFrame({"agent": ["1", "1", "2", "3"]})
#
#     result = group_rare_categories(df, "agent", min_freq=2)
#
#     assert result["agent"].tolist() == ["1", "1", "Other", "Other"]
#

def test_model_registry_contains_expected_models():
    registry = get_model_registry(random_state=7)

    assert {"baseline", "logistic", "xgboost", "catboost", "histboost", "ada_boost"} <= set(
        registry
    )
    assert registry["baseline"][1] == {}
    assert "C" in registry["logistic"][1]


def test_train_single_model_baseline_runs(hotel_dataset):
    result = train_single_model(hotel_dataset, model_name="baseline", cv_splits=2)

    assert isinstance(result, TrainingResult)
    assert result.model_name == "baseline"
    assert {"accuracy", "precision", "recall", "f1"} <= set(result.metrics)
    assert len(result.predictions) == len(result.y_true)


def test_train_single_model_logistic_returns_pipeline_and_test_split(hotel_dataset):
    result = train_single_model(hotel_dataset, model_name="logistic", cv_splits=2)

    assert hasattr(result.best_model, "named_steps")
    assert "preprocessing" in result.best_model.named_steps
    assert "model" in result.best_model.named_steps
    assert len(result.X_test) == len(result.y_true) == len(result.predictions)
    assert result.best_params


def test_metrics_are_valid(hotel_dataset):
    result = train_single_model(hotel_dataset, model_name="logistic", cv_splits=2)

    for key in ["accuracy", "f1", "precision", "recall"]:
        assert 0.0 <= result.metrics[key] <= 1.0


def test_unknown_model_raises_error(hotel_dataset):
    with pytest.raises(ValueError, match="Unknown model"):
        train_single_model(hotel_dataset, model_name="invalid_model")


def test_run_model_pipeline_direct(hotel_dataset):
    from sklearn.linear_model import LogisticRegression

    train = hotel_dataset.iloc[:40]
    test = hotel_dataset.iloc[40:]
    X_train = train.drop(columns=["is_canceled"])
    y_train = train["is_canceled"]
    X_test = test.drop(columns=["is_canceled"])
    y_test = test["is_canceled"]

    best_model, metrics, preds = run_model_pipeline(
        name="logistic",
        model=LogisticRegression(max_iter=500),
        param_grid={},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_splits=2,
    )

    assert "model" in best_model.named_steps
    assert isinstance(metrics, dict)
    assert len(preds) == len(y_test)


def test_pipeline_with_optional_selector_and_sampler(hotel_dataset):
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LogisticRegression

    train = hotel_dataset.iloc[:40]
    test = hotel_dataset.iloc[40:]

    best_model, metrics, preds = run_model_pipeline(
        name="logistic",
        model=LogisticRegression(max_iter=500),
        param_grid={},
        X_train=train.drop(columns=["is_canceled"]),
        y_train=train["is_canceled"],
        X_test=test.drop(columns=["is_canceled"]),
        y_test=test["is_canceled"],
        selector=VarianceThreshold(),
        sampler=RandomOverSampler(random_state=42),
        cv_splits=2,
    )

    assert "feature_selection" in best_model.named_steps
    assert "sampler" in best_model.named_steps
    assert len(preds) == len(test)
    assert 0.0 <= metrics["f1"] <= 1.0
