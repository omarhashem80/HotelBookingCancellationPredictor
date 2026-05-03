import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def dataframe_overview(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


@contextmanager
def timer(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("{} took {:.2f}s", label, elapsed)


SCHEMA = {
    # categorical
    "hotel": "category",
    "meal": "category",
    "country": "category",
    "market_segment": "category",
    "distribution_channel": "category",
    "reserved_room_type": "category",
    "assigned_room_type": "category",
    "deposit_type": "category",
    "customer_type": "category",
    "agent": "category",

    # binary
    "is_repeated_guest": "binary",
    "is_holiday": "binary",
    "is_canceled": "binary",

    # numeric (no need to split int types)
    "arrival_date_year": "numeric",
    "arrival_date_week_number": "numeric",
    "arrival_date_day_of_month": "numeric",
    "arrival_date_month": "numeric",
    "lead_time": "numeric",
    "stays_in_weekend_nights": "numeric",
    "stays_in_week_nights": "numeric",
    "adults": "numeric",
    "children": "numeric",
    "babies": "numeric",
    "previous_cancellations": "numeric",
    "previous_bookings_not_canceled": "numeric",
    "booking_changes": "numeric",
    "days_in_waiting_list": "numeric",
    "required_car_parking_spaces": "numeric",
    "total_of_special_requests": "numeric",
    "days_to_next_holiday": "numeric",
    "days_from_last_holiday": "numeric",

    # float
    "adr": "numeric",

    # datetime
    "arrival_date": "datetime",
    "reservation_status_date": "datetime"
}


def group_rare_categories(df, col, min_freq=100):
    counts = df[col].value_counts()
    rare = counts[counts < min_freq].index
    df[col] = df[col].replace(rare, "Other")
    return df


def enforce_schema(df, schema=SCHEMA, supress_rare=False):
    df = df.copy()
    for col, dtype in schema.items():

        if dtype == "category":
            df[col] = df[col].astype("string").astype("category")

        elif dtype == "binary":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int8")

        elif dtype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if supress_rare:
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df = group_rare_categories(df, col, min_freq=900)

    return df


