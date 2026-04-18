from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_numeric_features(
	df: pd.DataFrame,
	numeric_columns: Iterable[str],
) -> Tuple[pd.DataFrame, StandardScaler]:
	"""Scale selected numeric columns and return transformed dataframe + fitted scaler."""
	scaled = df.copy()
	scaler = StandardScaler()
	cols = list(numeric_columns)
	if cols:
		scaled[cols] = scaler.fit_transform(scaled[cols])
	return scaled, scaler

