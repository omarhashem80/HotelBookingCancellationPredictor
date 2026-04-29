from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def get_random_forest_estimator(
    random_state: int = 42,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def random_forest_param_grid() -> dict:
    return {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
