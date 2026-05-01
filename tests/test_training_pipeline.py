import pytest
import pandas as pd
import numpy as np

from src.models.trainer import (
    train_single_model,
    run_model_pipeline,
    TrainingResult,
)


@pytest.fixture
def toy_dataset():
    np.random.seed(42)

    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "is_canceled": np.random.randint(0, 2, 50)
    })

    return df


def test_train_single_model_baseline_runs(toy_dataset):
    result = train_single_model(toy_dataset, model_name="baseline")

    assert isinstance(result, TrainingResult)
    assert result.model_name == "baseline"

    assert isinstance(result.metrics, dict)
    assert "f1" in result.metrics or "accuracy" in result.metrics

    assert len(result.predictions) == len(result.y_true)


def test_predictions_match_test_set_size(toy_dataset):
    result = train_single_model(toy_dataset, model_name="logistic")

    assert len(result.predictions) == len(result.y_true)
    assert len(result.X_test) == len(result.y_true)



def test_pipeline_contains_model(toy_dataset):
    result = train_single_model(toy_dataset, model_name="logistic")

    assert hasattr(result.best_model, "named_steps")
    assert "model" in result.best_model.named_steps
    assert "preprocessing" in result.best_model.named_steps


def test_metrics_are_valid(toy_dataset):
    result = train_single_model(toy_dataset, model_name="logistic")

    metrics = result.metrics

    assert isinstance(metrics, dict)

    # check common classification metrics
    for key in ["f1", "precision", "recall"]:
        if key in metrics:
            assert 0.0 <= metrics[key] <= 1.0


def test_unknown_model_raises_error(toy_dataset):
    with pytest.raises(ValueError, match="Unknown model"):
        train_single_model(toy_dataset, model_name="invalid_model")



def test_run_model_pipeline_direct(toy_dataset):
    X = toy_dataset.drop(columns=["is_canceled"])
    y = toy_dataset["is_canceled"]

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=100)

    score, best_model, metrics, preds = run_model_pipeline(
        name="logistic",
        model=model,
        param_grid={},
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
    )

    assert isinstance(score, float)
    assert isinstance(metrics, dict)
    assert len(preds) == len(y)


def test_pipeline_with_optional_components(toy_dataset):
    result = train_single_model(
        toy_dataset,
        model_name="logistic",
        selector=None,
        sampler=None,
    )

    assert isinstance(result.best_model, object)
    assert result.predictions is not None