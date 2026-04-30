import os
import re
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

_WINDOWS_DRIVE_URI_RE = re.compile(r"^file:/+([A-Za-z]):/")


def _normalize_tracking_uri(tracking_uri: str) -> str:
    """Return a normalized tracking URI, converting local paths to file URIs."""
    if tracking_uri.startswith(("file:", "http://", "https://", "databricks")):
        return tracking_uri

    return Path(tracking_uri).expanduser().resolve().as_uri()


def _is_cross_os_windows_uri(uri: str) -> bool:
    """Detect a Windows drive-style file URI when running on non-Windows hosts."""
    return os.name != "nt" and bool(_WINDOWS_DRIVE_URI_RE.match(uri))


def _repair_experiment_if_needed(experiment_name: str) -> None:
    """Rotate incompatible experiment metadata and recreate with a valid artifact root."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    if not _is_cross_os_windows_uri(experiment.artifact_location):
        return

    # Artifact locations are immutable; rotate stale experiment and recreate it.
    legacy_name = f"{experiment_name}-legacy-os-mismatch"
    if client.get_experiment_by_name(legacy_name) is None:
        client.rename_experiment(experiment.experiment_id, legacy_name)
    else:
        client.delete_experiment(experiment.experiment_id)

    client.create_experiment(experiment_name)


def setup_mlflow(
    tracking_uri: str, experiment_name: str = "hotel-cancellation"
) -> None:
    """Configure MLflow tracking destination and experiment."""
    mlflow.set_tracking_uri(_normalize_tracking_uri(tracking_uri))
    _repair_experiment_if_needed(experiment_name)
    mlflow.set_experiment(experiment_name)
