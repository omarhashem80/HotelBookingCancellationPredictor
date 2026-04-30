from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import calculate_classification_metrics
from src.data.preprocess import (
    build_preprocessor,
    split_features_target,
)
from src.models.baseline import get_baseline_estimator
from src.models.catboost import (
    HAS_CATBOOST,
    catboost_param_grid,
    get_catboost_estimator,
)
from src.models.logistic import get_logistic_estimator, logistic_param_grid
from src.models.random_forest import (
    get_random_forest_estimator,
    random_forest_param_grid,
)
from src.models.xgboost import get_xgboost_estimator, xgboost_param_grid


@dataclass
class TrainingResult:
    model_name: str
    best_model: Any
    metrics: dict[str, float]
    predictions: list[int]
    best_params: dict[str, Any]
    y_true: list[int]
    X_test: pd.DataFrame


def _model_registry(random_state: int) -> dict[str, tuple[Any, dict]]:
    return {
        "baseline": (get_baseline_estimator(), {}),
        "logistic": (
            get_logistic_estimator(random_state=random_state),
            logistic_param_grid(),
        ),
        "random_forest": (
            get_random_forest_estimator(random_state=random_state),
            random_forest_param_grid(),
        ),
        "xgboost": (
            get_xgboost_estimator(random_state=random_state),
            xgboost_param_grid(),
        ),
        "catboost": (
            get_catboost_estimator(random_state=random_state),
            catboost_param_grid(),
        ),
    }


def _get_param_grid(
    model_name: str, tune_profile: str
) -> dict[str, list[Any]]:
    base_grids = {
        "baseline": {},
        "logistic": logistic_param_grid(),
        "random_forest": random_forest_param_grid(),
        "xgboost": xgboost_param_grid(),
        "catboost": catboost_param_grid(),
    }
    grid = base_grids[model_name]

    if tune_profile == "fast" and grid:
        reduced: dict[str, list[Any]] = {}
        for key, values in grid.items():
            if len(values) <= 2:
                reduced[key] = values
            else:
                mid = len(values) // 2
                reduced[key] = [values[0], values[mid]]
        return reduced

    return grid


def train_single_model(
    df: pd.DataFrame,
    model_name: str,
    target_col: str = "is_canceled",
    random_state: int = 42,
    test_size: float = 0.2,
    cv: int = 3,
    scoring: str = "f1",
    tune_profile: str = "full",
) -> TrainingResult:
    registry = _model_registry(random_state=random_state)
    if model_name not in registry:
        raise ValueError(f"Unknown model_name: {model_name}")

    X, y = split_features_target(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    estimator, _ = registry[model_name]
    param_grid = _get_param_grid(model_name, tune_profile=tune_profile)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    imbalance_ratio = float(neg_count / max(pos_count, 1))

    if model_name == "xgboost":
        estimator = get_xgboost_estimator(
            random_state=random_state,
            scale_pos_weight=imbalance_ratio,
        )

    if model_name == "catboost" and HAS_CATBOOST:
        catboost_params = estimator.get_params()
        catboost_params["class_weights"] = [1.0, imbalance_ratio]
        estimator.set_params(**catboost_params)

        cat_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_test, y_test),
            use_best_model=True,
            early_stopping_rounds=30,
            verbose=False,
        )
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        metrics = calculate_classification_metrics(
            y_test, predictions, probabilities
        )
        return TrainingResult(
            model_name=model_name,
            best_model=estimator,
            metrics=metrics,
            predictions=list(predictions),
            best_params=estimator.get_params(),
            y_true=list(y_test),
            X_test=X_test,
        )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    cv_strategy = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid or {},
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    probabilities = (
        best_model.predict_proba(X_test)[:, 1]
        if hasattr(best_model, "predict_proba")
        else None
    )
    metrics = calculate_classification_metrics(
        y_test, predictions, probabilities
    )

    return TrainingResult(
        model_name=model_name,
        best_model=best_model,
        metrics=metrics,
        predictions=list(predictions),
        best_params=search.best_params_,
        y_true=list(y_test),
        X_test=X_test,
    )
