import os
import re
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

_WINDOWS_DRIVE_URI_RE = re.compile(r"^file:/+([A-Za-z]):/")
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")

def _normalize_tracking_uri(tracking_uri: str) -> str:
    tracking_uri = tracking_uri.strip()

    if not tracking_uri:
        raise ValueError("MLflow tracking URI must not be empty")

    parsed = urlparse(tracking_uri)
    is_windows_path = bool(_WINDOWS_DRIVE_PATH_RE.match(tracking_uri))

    if parsed.scheme and not is_windows_path:
        return tracking_uri

    return Path(tracking_uri).expanduser().resolve().as_uri()

def _is_cross_os_windows_uri(uri: str) -> bool:
    return os.name != "nt" and bool(_WINDOWS_DRIVE_URI_RE.match(uri))


def _repair_experiment_if_needed(experiment_name: str) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return

    if not _is_cross_os_windows_uri(experiment.artifact_location):
        return

    logger.warning(
        "Detected incompatible MLflow experiment artifact location. Recreating experiment: {}",
        experiment_name,
    )

    client.delete_experiment(experiment.experiment_id)
    client.create_experiment(experiment_name)


def setup_mlflow(tracking_uri: str, experiment_name: str = "hotel-cancellation") -> None:

    mlflow.set_tracking_uri(_normalize_tracking_uri(tracking_uri))
    logger.info("Configured MLflow tracking URI: {}", mlflow.get_tracking_uri())
    _repair_experiment_if_needed(experiment_name)
    mlflow.set_experiment(experiment_name)
    logger.info("Using MLflow experiment: {}", experiment_name)