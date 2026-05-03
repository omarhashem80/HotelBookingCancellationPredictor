from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from loguru import logger

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline

from src.evaluation.metrics import calculate_classification_metrics
from src.data.preprocess import build_preprocessor, split_features_target

from src.models.baseline import get_baseline_estimator
from src.models.catboost import get_catboost_estimator, get_catboost_param_grid
from src.models.histboost import get_histboost_estimator, get_histboost_param_grid
from src.models.logistic import get_logistic_estimator, get_logistic_param_grid
from src.models.adaboost import get_ada_boost_estimator, get_ada_boost_param_grid
from src.models.xgboost import get_xgboost_estimator, get_xgboost_param_grid


SCHEMA = {
    # categorical
    "hotel": "category",
    "meal": "category",
    "country": "category",
    "market_segment": "category",
    "distribution_channel": "category",
    "reserved_room_type": "category",
    "assigned_room_type": "category",
    "deposit_type": "category",
    "customer_type": "category",
    "agent": "category",

    # binary
    "is_repeated_guest": "binary",
    "is_holiday": "binary",
    "is_canceled": "binary",

    # numeric (no need to split int types)
    "arrival_date_year": "numeric",
    "arrival_date_week_number": "numeric",
    "arrival_date_day_of_month": "numeric",
    "arrival_date_month": "numeric",
    "lead_time": "numeric",
    "stays_in_weekend_nights": "numeric",
    "stays_in_week_nights": "numeric",
    "adults": "numeric",
    "children": "numeric",
    "babies": "numeric",
    "previous_cancellations": "numeric",
    "previous_bookings_not_canceled": "numeric",
    "booking_changes": "numeric",
    "days_in_waiting_list": "numeric",
    "required_car_parking_spaces": "numeric",
    "total_of_special_requests": "numeric",
    "days_to_next_holiday": "numeric",
    "days_from_last_holiday": "numeric",

    # float
    "adr": "numeric",

    # datetime
    "arrival_date": "datetime",
    "reservation_status_date": "datetime"
}

@dataclass
class TrainingResult:
    model_name: str
    best_model: Any
    metrics: dict[str, float]
    predictions: list[int]
    best_params: dict[str, Any]
    y_true: list[int]
    X_test: pd.DataFrame


def enforce_schema(df, schema):
    df = df.copy()
    for col, dtype in schema.items():

        if dtype == "category":
            df[col] = df[col].astype("string").astype("category")

        elif dtype == "binary":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int8")

        elif dtype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def get_model_registry(random_state: int) -> dict[str, tuple[Any, dict]]:
    return {
        "baseline": (get_baseline_estimator(), {}),
        "logistic": (
            get_logistic_estimator(random_state),
            get_logistic_param_grid(),
        ),
        "xgboost": (
            get_xgboost_estimator(random_state),
            get_xgboost_param_grid(),
        ),
        "catboost": (
            get_catboost_estimator(random_state),
            get_catboost_param_grid(),
        ),
        "histboost": (
            get_histboost_estimator(random_state),
            get_histboost_param_grid(),
        ),
        "ada_boost": (
            get_ada_boost_estimator(random_state),
            get_ada_boost_param_grid(),
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
) -> tuple [Any, dict[str, float], list[int]]:

    PipelineClass = ImbPipeline if sampler else Pipeline

    steps = [("preprocessing", build_preprocessor())]

    if selector is not None:
        steps.append(("feature_selection", selector))

    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(("model", model))

    pipeline = PipelineClass(steps=steps)

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={f"model__{k}": v for k, v in (param_grid or {}).items()},
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        error_score="raise",
    )

    logger.info("Training pipeline: model={}, params={}", name, list(param_grid.keys()))

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)

    metrics = calculate_classification_metrics(y_test.values, y_pred)

    logger.info(
        "Done: model={}, best_cv_f1={:.4f}, test_f1={:.4f}",
        name,
        search.best_score_,
        metrics.get("f1", 0.0),
    )

    return best_model, metrics, list(y_pred)


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

    registry = get_model_registry(random_state)

    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info("Starting training: {}", model_name)

    df = enforce_schema(df, SCHEMA)

    X, y = split_features_target(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info(
        "Split done: train={}, test={}",
        X_train.shape[0],
        X_test.shape[0],
    )

    model, param_grid = registry[model_name]

    best_model, metrics, predictions = run_model_pipeline(
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