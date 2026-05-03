from pathlib import Path
from typing import Any

import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.models import infer_signature


def log_params(params: dict[str, Any]) -> None:
    clean_params = {k: str(v) for k, v in params.items()}
    mlflow.log_params(clean_params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_tags(tags: dict[str, Any]) -> None:
    mlflow.set_tags({k: str(v) for k, v in tags.items()})


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))


def log_artifact_dir(path: str | Path, artifact_path: str | None = None) -> None:
    """Log an entire directory as MLflow artifacts."""
    mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def log_figure(fig: Any, filename: str) -> None:
    """Log a matplotlib or plotly figure directly to the active MLflow run."""
    mlflow.log_figure(fig, filename)


def log_model(model: Any, artifact_path: str, input_example: Any | None = None) -> None:
    if input_example is not None:
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow_sklearn.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
        )
    else:
        mlflow_sklearn.log_model(
            model,
            artifact_path=artifact_path,
        )
