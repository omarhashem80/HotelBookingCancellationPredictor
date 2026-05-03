from pathlib import Path
from typing import Any

import mlflow
from mlflow.models import infer_signature


def log_params(params: dict[str, Any]) -> None:
    clean_params = {k: str(v) for k, v in params.items()}
    mlflow.log_params(clean_params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))


def log_model(model: Any, artifact_path: str, input_example: Any | None = None) -> None:
    signature = None

    if input_example is not None:
        signature = infer_signature(input_example, model.predict(input_example))

    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
    )
