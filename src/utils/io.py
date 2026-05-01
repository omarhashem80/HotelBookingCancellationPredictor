from pathlib import Path
from typing import Any

import joblib
import pandas as pd
# import tomllib
import tomli as tomllib  # Python 3.10 fallback
from functools import lru_cache
from loguru import logger


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded CSV: rows={}, cols={}, path={}", df.shape[0], df.shape[1], path)
    return df


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    out_path = Path(path)
    ensure_parent_dir(out_path)
    df.to_csv(out_path, index=index)
    logger.info("Saved CSV: rows={}, cols={}, path={}", df.shape[0], df.shape[1], out_path)


def save_model(model: Any, path: str | Path) -> None:
    out_path = Path(path)
    ensure_parent_dir(out_path)
    joblib.dump(model, out_path)
    logger.info("Saved model artifact to {}", out_path)


def load_model(path: str | Path) -> Any:
    model = joblib.load(path)
    logger.info("Loaded model artifact from {}", path)
    return model


@lru_cache(maxsize=1)
def read_config() -> dict:
    with open(Path("configs/config.toml"), "rb") as f:
        config = tomllib.load(f)
    logger.info("Loaded config from configs/config.toml")
    return config
