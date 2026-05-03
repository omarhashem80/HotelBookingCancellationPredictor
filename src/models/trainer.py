from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from loguru import logger

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline

from src.evaluation.metrics import calculate_classification_metrics
from src.data.preprocess import build_preprocessor, split_features_target

from src.models.baseline import get_baseline_estimator
from src.models.catboost import catboost_param_grid, get_catboost_estimator
from src.models.histboost import get_histboost_estimator, histboost_param_grid
from src.models.logistic import get_logistic_estimator, logistic_param_grid
from src.models.adaboost import get_ada_boost_estimator, ada_boost_param_grid
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
            get_logistic_estimator(random_state),
            logistic_param_grid(),
        ),
        "xgboost": (get_xgboost_estimator(random_state), xgboost_param_grid()),
        "catboost": (
            get_catboost_estimator(random_state),
            catboost_param_grid(),
        ),
        "histboost": (
            get_histboost_estimator(random_state),
            histboost_param_grid(),
        ),
        "ada_boost": (
            get_ada_boost_estimator(random_state),
            ada_boost_param_grid(),
        ),
    }


def run_model_pipeline(
    name: str,
    model: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selector: Optional[Any] = None,
    sampler: Optional[Any] = None,
    random_state: int = 42,
    cv_splits: int = 5,
) -> tuple[float, Any, dict[str, float], list[int]]:

    PipelineClass = ImbPipeline if sampler is not None else Pipeline

    steps = [("preprocessing", build_preprocessor(X_train))]

    if selector is not None:
        steps.append(("feature_selection", selector))

    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(("model", model))

    pipeline = PipelineClass(steps=steps)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid or {},
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        error_score="raise",
    )

    logger.info("Fitting model pipeline with params={}", list(param_grid.keys()))

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = (
        best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    )

    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)

    logger.info(
        "Model pipeline complete: best_score={:.4f}, f1={:.4f}",
        search.best_score_,
        metrics.get("f1", 0.0),
    )

    return search.best_score_, best_model, metrics, list(y_pred)


def train_single_model(
    df: pd.DataFrame,
    model_name: str,
    target_col: str = "is_canceled",
    random_state: int = 42,
    test_size: float = 0.2,
    selector: Optional[Any] = None,
    sampler: Optional[Any] = None,
    cv_splits: int = 5,
) -> TrainingResult:

    registry = _model_registry(random_state)

    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info("Training single model: {}", model_name)

    X, y = split_features_target(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        "Train/test split: train_rows={}, test_rows={}, test_size={}",
        X_train.shape[0],
        X_test.shape[0],
        test_size,
    )

    model, param_grid = registry[model_name]

    best_score, best_model, metrics, predictions = run_model_pipeline(
        name=model_name,
        model=model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        selector=selector,
        sampler=sampler,
        random_state=random_state,
        cv_splits=cv_splits,
    )

    return TrainingResult(
        model_name=model_name,
        best_model=best_model,
        metrics=metrics,
        predictions=predictions,
        best_params=best_model.get_params(),
        y_true=list(y_test),
        X_test=X_test,
    )
