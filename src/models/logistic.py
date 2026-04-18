from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def get_logistic_estimator(random_state: int = 42) -> LogisticRegression:
	return LogisticRegression(max_iter=3000, random_state=random_state, solver="liblinear")


def logistic_param_grid() -> dict:
	return {
		"model__C": [0.01, 0.1, 1.0, 10.0],
		"model__penalty": ["l1", "l2"],
		"model__class_weight": [None, "balanced"],
	}

