import pandas as pd

CATEGORICAL_COLS = [
    'hotel',
    'arrival_date_month',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'customer_type',
    'reservation_status',
    'arrival_date_year',
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'reservation_status_year',
    'reservation_status_month',
    'reservation_status_day',
]

BINARY_COLS = ['is_canceled', 'is_repeated_guest', 'is_holiday']

INT_COLS = [
    'lead_time',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'babies',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'booking_changes',
    'days_in_waiting_list',
    'required_car_parking_spaces',
    'total_of_special_requests',
    'children',
    'days_to_next_holiday',
    'days_from_last_holiday',
]

FLOAT_COLS = ['adr']

DATE_COLS = ['arrival_date', 'reservation_status_date']


def clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype('int8').astype('category')
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    df['agent'] = df['agent'].astype('str')
    df = df[
        ~((df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0))
    ]  # drop rows where no guests.
    return df


def reduce_cardinality(
    df: pd.DataFrame, col_name: str, top_k: int = 5
) -> pd.DataFrame:
    top_k_categories = df[col_name].value_counts().head(top_k).index
    df[col_name] = df[col_name].apply(
        lambda x: x if x in top_k_categories else 'Other'
    )
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df["children"].fillna(0, inplace=True)
    df["country"].fillna("Unknown", inplace=True)
    df["agent"].fillna(0, inplace=True)
    df["is_holiday"].fillna(0, inplace=True)
    df["days_to_next_holiday"].fillna(-1, inplace=True)
    df["days_from_last_holiday"].fillna(-1, inplace=True)
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"]
    )
    df['reservation_status_year'] = df['reservation_status_date'].dt.year
    df['reservation_status_month'] = df['reservation_status_date'].dt.month
    df['reservation_status_day'] = df['reservation_status_date'].dt.day
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df['arrival_date_year'] = df['arrival_date'].dt.year
    df['arrival_date_month'] = df['arrival_date'].dt.month
    df['arrival_date_week_number'] = df['arrival_date'].dt.isocalendar().week
    df['arrival_date_day_of_month'] = df['arrival_date'].dt.day
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    for col, k in (('agent', 10), ('country', 5)):
        cleaned = reduce_cardinality(cleaned, col, k)
    cleaned = fill_missing_values(cleaned)
    cleaned = add_datetime_features(cleaned)
    cleaned = clean_dtypes(cleaned)
    return cleaned.reset_index(drop=True)
