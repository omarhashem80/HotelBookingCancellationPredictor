from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
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
        "Split features/target: rows={}, features={}, target={}", len(df), X.shape[1], target_col
    )
    return X, y


def cols_grouped_by_type(df: pd.DataFrame) -> tuple[list, list, list]:
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols = df.select_dtypes(
        include=["datetime", "datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()
    logger.debug(
        "Grouped columns: numerical={}, categorical={}, datetime={}",
        len(numerical_cols),
        len(categorical_cols),
        len(date_cols),
    )
    return numerical_cols, categorical_cols, date_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical_cols, categorical_cols, date_cols = cols_grouped_by_type(X)
    logger.debug("Building preprocessor with {} features", X.shape[1])
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
            (
                "month",
                FunctionTransformer(
                    extract_month,
                    validate=False
                ),
            ),
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
