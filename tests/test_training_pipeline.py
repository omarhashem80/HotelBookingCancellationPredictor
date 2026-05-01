# import pytest
#
# from src.data.cleaning import clean_dtypes
# from src.models.trainer import TrainingResult, train_single_model
#
#
# def test_model_training_runs_baseline(toy_dataset) -> None:
#     df = clean_dtypes(toy_dataset.copy())
#     result = train_single_model(df, model_name="baseline")
#     assert isinstance(result, TrainingResult)
#     assert result.model_name == "baseline"
#     assert "accuracy" in result.metrics
#     assert len(result.predictions) == len(result.y_true)
#
#
# def test_prediction_shape_matches_holdout(toy_dataset) -> None:
#     df = clean_dtypes(toy_dataset.copy())
#     result = train_single_model(df, model_name="logistic")
#     assert len(result.predictions) == len(result.y_true)
#     assert "preprocessing" in result.best_model.named_steps
#
#
# def test_unknown_model_raises_value_error(toy_dataset) -> None:
#     df = clean_dtypes(toy_dataset.copy())
#     with pytest.raises(ValueError, match="Unknown model"):
#         train_single_model(df, model_name="not_a_model")
