from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: str | Path) -> pd.DataFrame:
	return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
	out_path = Path(path)
	ensure_parent_dir(out_path)
	df.to_csv(out_path, index=index)


def save_model(model: Any, path: str | Path) -> None:
	out_path = Path(path)
	ensure_parent_dir(out_path)
	joblib.dump(model, out_path)


def load_model(path: str | Path) -> Any:
	return joblib.load(path)

