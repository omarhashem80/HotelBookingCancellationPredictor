import numpy as np
import pandas as pd
import pytest
import pycountry
from src.data.cleaning import clean_data, clean_dtypes, reduce_cardinality
from src.data.ingestion import merge_datasets
from src.data.preprocess import cols_grouped_by_type


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


def test_cols_grouped_by_type(toy_dataset):
    cleaned_types_df = clean_dtypes(toy_dataset)
    numerical_cols, categorical_cols, date_cols = cols_grouped_by_type(
        cleaned_types_df
    )
    assert len(numerical_cols) == 9
    assert len(categorical_cols) == 4
    assert 'is_canceled' not in numerical_cols
    assert 'adr' in numerical_cols
    assert 'reserved_room_type' in categorical_cols
    assert 'reservation_status_date' in date_cols


@pytest.mark.parametrize(
    "size,k_agent,k_country", [(10, 2, 4), (50, 4, 4), (200, 5, 5)]
)
def test_reduce_cardinality(size, k_agent, k_country):
    rng = np.random.default_rng(42)
    all_countries = np.array(list(pycountry.countries))
    random_countries = rng.choice(all_countries, size=k_country * 4)
    country_names = [country.name for country in random_countries]
    df = pd.DataFrame(
        {
            "agent": rng.integers(50, 600, size=size),
            "country": rng.choice(country_names, size=size),
        }
    )
    cols = ["agent", "country"]
    for col, k_col in zip(cols, [k_agent, k_country]):
        df = reduce_cardinality(df, col, k_col)
    agent_cardi = len(df['agent'].unique())
    country_cardi = len(df['country'].unique())
    # plus 1 -> to consider Other Category
    assert agent_cardi <= k_agent + 1
    assert country_cardi <= k_country + 1
