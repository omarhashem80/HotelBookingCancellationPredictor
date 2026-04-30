from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def split_features_target(
    df: pd.DataFrame, target_col: str = "is_canceled"
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def cols_grouped_by_type(df: pd.DataFrame) -> tuple[list, list, list]:
    numerical_cols = df.select_dtypes(["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(["int8", "category"]).columns.tolist()
    date_cols = df.select_dtypes(["datetime"]).columns.tolist()
    return numerical_cols, categorical_cols, date_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical_cols, categorical_cols, date_cols = cols_grouped_by_type(X)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    month_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
            ("month", month_pipeline, date_cols),
        ],
        remainder="drop",
    )

    return preprocessor
