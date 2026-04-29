from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data.cleaning import CATEGORICAL_COLS, DATE_COLS, FLOAT_COLS, INT_COLS


def split_features_target(
    df: pd.DataFrame, target_col: str = "is_canceled"
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
            ("num", numeric_pipeline, [*INT_COLS, *FLOAT_COLS]),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ("month", month_pipeline, DATE_COLS),
        ],
        remainder="drop",
    )

    return preprocessor
