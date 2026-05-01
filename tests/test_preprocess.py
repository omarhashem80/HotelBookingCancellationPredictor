import pytest
import pandas as pd

from src.data.preprocess import (
    split_features_target,
    cols_grouped_by_type,
    build_preprocessor,
)



@pytest.fixture
def toy_df():
    return pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [10.0, 20.0, 30.0, 40.0],
        "cat1": ["a", "b", "a", "c"],
        "date1": pd.to_datetime([
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01"
        ]),
        "is_canceled": [0, 1, 0, 1],
    })


def test_split_features_target(toy_df):
    X, y = split_features_target(toy_df)

    assert "is_canceled" not in X.columns
    assert len(X) == len(y)
    assert y.name == "is_canceled"



def test_split_missing_target_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(KeyError):
        split_features_target(df, target_col="is_canceled")



def test_cols_grouping(toy_df):
    num, cat, date = cols_grouped_by_type(toy_df)

    assert "num1" in num
    assert "num2" in num

    assert "cat1" in cat
    assert "date1" in date

    assert isinstance(num, list)
    assert isinstance(cat, list)
    assert isinstance(date, list)



def test_build_preprocessor_structure(toy_df):
    X = toy_df.drop(columns=["is_canceled"])

    preprocessor = build_preprocessor(X)

    assert preprocessor is not None
    assert hasattr(preprocessor, "transformers")

    transformer_names = [t[0] for t in preprocessor.transformers]

    assert "num" in transformer_names
    assert "cat" in transformer_names
    assert "month" in transformer_names



def test_preprocessor_fit_transform(toy_df):
    X = toy_df.drop(columns=["is_canceled"])

    preprocessor = build_preprocessor(X)

    Xt = preprocessor.fit_transform(X)

    assert Xt is not None
    assert Xt.shape[0] == len(X)



def test_empty_dataframe():
    df = pd.DataFrame({"is_canceled": [0, 1]})

    X, y = split_features_target(df)

    assert X.empty
    assert len(y) == 2