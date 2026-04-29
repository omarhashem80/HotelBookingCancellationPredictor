import pandas as pd

# def _add_total_guests(df: pd.DataFrame) -> pd.DataFrame:
#     if {"adults", "children", "babies"}.issubset(df.columns):
#         df["total_guests"] = df["adults"] + df["children"] + df["babies"]
#     return df


# def _add_total_nights(df: pd.DataFrame) -> pd.DataFrame:
#     if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(
#         df.columns
#     ):
#         df["total_nights"] = (
#             df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
#         )
#     return df


# def _add_room_change_flag(df: pd.DataFrame) -> pd.DataFrame:
#     if {"reserved_room_type", "assigned_room_type"}.issubset(df.columns):
#         df["room_change"] = (
#             df["reserved_room_type"] != df["assigned_room_type"]
#         ).astype(int)
#     return df


# def _add_lead_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
#     if "lead_time" in df.columns:
#         df["lead_time_bucket"] = pd.cut(
#             df["lead_time"],
#             bins=[-1, 7, 30, 90, 365, float("inf")],
#             labels=["0_7", "8_30", "31_90", "91_365", "365_plus"],
#         ).astype(str)
#     return df


# def _add_holiday_distance(df: pd.DataFrame) -> pd.DataFrame:
#     if {"days_to_next_holiday", "days_from_last_holiday"}.issubset(df.columns):
#         df["holiday_distance_min"] = df[
#             ["days_to_next_holiday", "days_from_last_holiday"]
#         ].min(axis=1)
#     return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
    df["reservation_status_year"] = df["reservation_status_date"].dt.year
    df["reservation_status_month"] = df["reservation_status_date"].dt.month
    df["reservation_status_day"] = df["reservation_status_date"].dt.day
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["arrival_date_year"] = df["arrival_date"].dt.year
    df["arrival_date_month"] = df["arrival_date"].dt.month
    df["arrival_date_week_number"] = df["arrival_date"].dt.isocalendar().week
    df["arrival_date_day_of_month"] = df["arrival_date"].dt.day
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured.drop("reservation_status", inplace=True)
    featured = add_datetime_features(featured)
    # featured = _add_total_guests(featured)
    # featured = _add_total_nights(featured)
    # featured = _add_room_change_flag(featured)
    # featured = _add_lead_time_bucket(featured)
    # featured = _add_holiday_distance(featured)
    return featured
