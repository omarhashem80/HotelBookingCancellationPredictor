from __future__ import annotations

import mlflow


def setup_mlflow(tracking_uri: str, experiment_name: str = "hotel-cancellation") -> None:
	"""Configure MLflow tracking destination and experiment."""
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(experiment_name)

