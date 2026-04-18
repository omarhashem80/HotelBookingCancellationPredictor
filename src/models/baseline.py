from __future__ import annotations

from sklearn.dummy import DummyClassifier


def get_baseline_estimator() -> DummyClassifier:
	"""Simple baseline classifier for performance benchmarking."""
	return DummyClassifier(strategy="most_frequent")

