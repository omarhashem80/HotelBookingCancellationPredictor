import pandas as pd

CATEGORICAL_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "reservation_status",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "reservation_status_year",
    "reservation_status_month",
    "reservation_status_day",
]

BINARY_COLS = ["is_canceled", "is_repeated_guest", "is_holiday"]

INT_COLS = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "children",
    "days_to_next_holiday",
    "days_from_last_holiday",
]

FLOAT_COLS = ["adr"]

DATE_COLS = ["arrival_date", "reservation_status_date"]


def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype("int8").astype("category")
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "agent" in df.columns.tolist():
        df["agent"] = df["agent"].astype("str")

    return df


def reduce_cardinality(
    df: pd.DataFrame, col_name: str, top_k: int = 5
) -> pd.DataFrame:
    top_k_categories = df[col_name].value_counts().head(top_k).index
    df[col_name] = df[col_name].apply(
        lambda x: x if x in top_k_categories else "Other"
    )
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_fill = [
        "children",
        "country",
        "agent",
        "is_holiday",
        "days_to_next_holiday",
        "days_from_last_holiday",
    ]
    if cols_to_fill not in df.columns.tolist():
        return df
    df["children"].fillna(0, inplace=True)
    df["country"].fillna("Unknown", inplace=True)
    df["agent"].fillna(0, inplace=True)
    df["is_holiday"].fillna(0, inplace=True)
    df["days_to_next_holiday"].fillna(-1, inplace=True)
    df["days_from_last_holiday"].fillna(-1, inplace=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_check = ["adults", "children", "babies"]
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    for col, k in (("agent", 10), ("country", 5)):
        if col in df.columns:
            cleaned = reduce_cardinality(cleaned, col, k)
    cleaned = fill_missing_values(cleaned)
    cleaned = clean_dtypes(cleaned)
    columns_exist = all([_ in df.columns.tolist() for _ in cols_to_check])
    if columns_exist:
        cleaned = cleaned[
            ~(
                (cleaned["adults"] == 0)
                & (cleaned["children"] == 0)
                & (cleaned["babies"] == 0)
            )
        ]  # drop rows where no guests.
    return cleaned.reset_index(drop=True)
