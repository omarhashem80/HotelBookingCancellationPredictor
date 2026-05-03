from typing import Tuple

import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from loguru import logger


def extract_month(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.dt.month)


def split_features_target(
    df: pd.DataFrame, target_col: str = "is_canceled"
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.debug(
        "Split features/target: rows={}, features={}, target={}",
        len(df),
        X.shape[1],
        target_col,
    )
    return X, y


def get_types() -> tuple[list, list, list]:
    numerical_cols = [
        'lead_time',
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'adults',
        'children',
        'babies',
        'previous_cancellations',
        'previous_bookings_not_canceled',
        'booking_changes',
        'days_in_waiting_list',
        'adr',
        'required_car_parking_spaces',
        'total_of_special_requests',
        'days_to_next_holiday',
        'days_from_last_holiday',
        'arrival_date_year',
        'arrival_date_week_number',
        'arrival_date_day_of_month',
        'reservation_status_year',
        'reservation_status_day'
    ]

    categorical_cols = [
        'hotel',
        'meal',
        'market_segment',
        'distribution_channel',
        'reserved_room_type',
        'assigned_room_type',
        'deposit_type',
        'customer_type',
        'agent',
        'is_repeated_guest',
        'is_holiday'
    ]

    date_cols = ['arrival_date_month', 'reservation_status_month']

    return numerical_cols, categorical_cols, date_cols


def build_preprocessor() -> ColumnTransformer:
    numerical_cols, categorical_cols, date_cols = get_types()
    logger.debug("Building preprocessor with {} features", len(numerical_cols) + len(categorical_cols) + len(date_cols))
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="error", sparse_output=False, drop="first")),
        ]
    )

    month_pipeline = Pipeline(
        steps=[
            # (
            #     "month",
            #     FunctionTransformer(extract_month, validate=False),
            # ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    transformers = [
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
        ("mon", month_pipeline, date_cols),
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor
