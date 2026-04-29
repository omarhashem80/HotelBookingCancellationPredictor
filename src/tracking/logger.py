from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


def log_params(params: dict[str, Any]) -> None:
    clean_params = {k: str(v) for k, v in params.items()}
    mlflow.log_params(clean_params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))


def log_model(model: Any, artifact_path: str) -> None:
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)
