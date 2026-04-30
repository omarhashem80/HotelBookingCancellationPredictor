from src.models.trainer import train_single_model


def test_model_training_runs_baseline(toy_dataset) -> None:
    df = toy_dataset
    result = train_single_model(df, model_name="baseline", cv=2)
    assert "accuracy" in result.metrics
    assert len(result.predictions) > 0


def test_prediction_shape_matches_holdout(toy_dataset) -> None:
    df = toy_dataset
    result = train_single_model(df, model_name="logistic", cv=2)
    assert len(result.predictions) == len(result.y_true)
