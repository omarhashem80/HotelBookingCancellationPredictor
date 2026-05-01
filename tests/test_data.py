import pandas as pd

from src.data.cleaning import clean_data, clean_dtypes
from src.data.ingestion import merge_datasets
from src.data.preprocess import cols_grouped_by_type
from src.data.validation import validate_dataframe


def test_merge_datasets_left_join() -> None:
    left = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    right = pd.DataFrame({"id": [1], "y": [99]})
    merged = merge_datasets(left, right, on="id", how="left")
    assert "y" in merged.columns
    assert pd.isna(merged.loc[1, "y"])


def test_clean_data_removes_invalid_guest_rows() -> None:
    df = pd.DataFrame(
        {
            "adults": [0, 2],
            "children": [0, 0],
            "babies": [0, 0],
            "lead_time": [1, 2],
            "is_canceled": [0, 1],
        }
    )
    cleaned = clean_data(df)
    assert len(cleaned) == 1


def test_validate_dataframe_contains_required_sections() -> None:
    df = pd.DataFrame({"is_canceled": [0, 1, 1], "lead_time": [10, 12, 11]})
    report = validate_dataframe(df)
    assert "missing_values" in report
    assert "duplicates" in report
    assert "schema_issues" in report
    assert "class_balance" in report


def test_cols_grouped_by_type(toy_dataset):
    cleaned_types_df = clean_dtypes(toy_dataset)
    numerical_cols, categorical_cols, date_cols = cols_grouped_by_type(cleaned_types_df)
    assert len(numerical_cols) == 9
    assert len(categorical_cols) == 4
    assert "is_canceled" not in numerical_cols
    assert "adr" in numerical_cols
    assert "reserved_room_type" in categorical_cols
    assert "reservation_status_date" in date_cols
