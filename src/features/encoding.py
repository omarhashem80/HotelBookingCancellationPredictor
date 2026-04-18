from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target_col: str = "is_canceled") -> Tuple[pd.DataFrame, pd.Series]:
	X = df.drop(columns=[target_col])
	y = df[target_col]
	return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
	"""Create preprocessing pipeline: one-hot for categorical + scaling for numeric."""
	cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
	num_cols = [c for c in X.columns if c not in cat_cols]

	return ColumnTransformer(
		transformers=[
			(
				"cat",
				OneHotEncoder(handle_unknown="ignore"),
				cat_cols,
			),
			(
				"num",
				Pipeline(steps=[("scaler", StandardScaler())]),
				num_cols,
			),
		]
	)

