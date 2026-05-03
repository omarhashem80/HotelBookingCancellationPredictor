import pytest
import numpy as np
import pandas as pd

from src.data.preprocess import (
    split_features_target,
    get_types,
    build_preprocessor,
)


def _get_transformer(preprocessor, name):
    return next(
        transformer
        for transformer_name, transformer, _ in preprocessor.transformers
        if transformer_name == name
    )


def _build_df(n: int) -> pd.DataFrame:
    numerical_cols, categorical_cols, date_cols = get_types()

    num_data = {col: np.arange(1, n + 1, dtype=float) for col in numerical_cols}

    cat_values = {
        "hotel":                ["City Hotel", "Resort Hotel"] * (n // 2),
        "meal":                 (["BB", "HB", "SC"] * ((n // 3) + 1))[:n],
        "market_segment":       ["Online TA"] * n,
        "distribution_channel": ["TA/TO"] * n,
        "reserved_room_type":   (["A", "B", "C"] * ((n // 3) + 1))[:n],
        "assigned_room_type":   (["A", "B", "C"] * ((n // 3) + 1))[:n],
        "deposit_type":         ["No Deposit"] * n,
        "customer_type":        ["Transient"] * n,
        "agent":                (["1", "2"] * ((n // 2) + 1))[:n],
        "is_repeated_guest":    (["0", "1"] * ((n // 2) + 1))[:n],
        "is_holiday":           (["0", "1"] * ((n // 2) + 1))[:n],
    }
    cat_data = {
        col: cat_values.get(col, (["val_a", "val_b"] * ((n // 2) + 1))[:n])
        for col in categorical_cols
    }

    date_data = {
        col: ([1, 2, 3, 4, 5, 6] * ((n // 6) + 1))[:n]
        for col in date_cols
    }

    target = {"is_canceled": ([0, 1] * ((n // 2) + 1))[:n]}

    return pd.DataFrame({**num_data, **cat_data, **date_data, **target})


@pytest.fixture(params=[60, 120])
def minimal_df(request) -> pd.DataFrame:
    return _build_df(request.param)


class TestSplitFeaturesTarget:
    def test_target_removed_from_features(self, minimal_df):
        X, _ = split_features_target(minimal_df)
        assert "is_canceled" not in X.columns

    def test_row_counts_match(self, minimal_df):
        X, y = split_features_target(minimal_df)
        assert len(X) == len(y)

    def test_target_series_name(self, minimal_df):
        _, y = split_features_target(minimal_df)
        assert y.name == "is_canceled"

    def test_feature_count(self, minimal_df):
        X, _ = split_features_target(minimal_df)
        assert X.shape[1] == minimal_df.shape[1] - 1

    def test_missing_target_raises_key_error(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(KeyError):
            split_features_target(df, target_col="is_canceled")

    def test_custom_target_col(self):
        df = pd.DataFrame({"feat": [1, 2, 3], "label": [0, 1, 0]})
        X, y = split_features_target(df, target_col="label")
        assert "label" not in X.columns
        assert y.name == "label"

    def test_empty_features_after_split(self):
        """Only the target column present → X should be empty."""
        df = pd.DataFrame({"is_canceled": [0, 1]})
        X, y = split_features_target(df)
        assert X.empty
        assert len(y) == 2


class TestGetTypes:
    def test_returns_three_lists(self):
        result = get_types()
        assert len(result) == 3
        assert all(isinstance(r, list) for r in result)

    def test_known_numerical_cols_present(self):
        num, _, _ = get_types()
        expected = [
            "lead_time",
            "adr",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "booking_changes",
        ]
        for col in expected:
            assert col in num, f"Expected '{col}' in numerical_cols"

    def test_known_categorical_cols_present(self):
        _, cat, _ = get_types()
        expected = [
            "hotel",
            "meal",
            "market_segment",
            "distribution_channel",
            "reserved_room_type",
            "deposit_type",
            "customer_type",
        ]
        for col in expected:
            assert col in cat, f"Expected '{col}' in categorical_cols"

    def test_known_date_cols_present(self):
        _, _, date = get_types()
        assert "arrival_date_month" in date
        assert "reservation_status_month" in date

    def test_no_overlap_between_groups(self):
        num, cat, date = get_types()
        assert not set(num) & set(cat),  "num and cat share columns"
        assert not set(num) & set(date), "num and date share columns"
        assert not set(cat) & set(date), "cat and date share columns"

    def test_non_empty_groups(self):
        num, cat, date = get_types()
        assert len(num) > 0
        assert len(cat) > 0
        assert len(date) > 0


class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        assert isinstance(build_preprocessor(), ColumnTransformer)

    def test_has_required_transformer_names(self):
        names = [t[0] for t in build_preprocessor().transformers]
        assert "num" in names, "Missing 'num' transformer"
        assert "cat" in names, "Missing 'cat' transformer"
        assert "mon" in names, "Missing 'mon' transformer"

    def test_transformer_count(self):
        assert len(build_preprocessor().transformers) == 3

    def test_remainder_is_drop(self):
        assert build_preprocessor().remainder == "drop"

    def test_numeric_pipeline_steps(self):
        num_pipeline = _get_transformer(build_preprocessor(), "num")
        step_names = [name for name, _ in num_pipeline.steps]
        assert "imputer" in step_names
        assert "scaler" in step_names

    def test_categorical_pipeline_steps(self):
        cat_pipeline = _get_transformer(build_preprocessor(), "cat")
        step_names = [name for name, _ in cat_pipeline.steps]
        assert "imputer" in step_names
        assert "onehot" in step_names

    def test_month_pipeline_has_imputer(self):
        mon_pipeline = _get_transformer(build_preprocessor(), "mon")
        step_names = [name for name, _ in mon_pipeline.steps]
        assert "imputer" in step_names

    def test_fit_transform_output_shape(self, minimal_df):
        X, _ = split_features_target(minimal_df)
        Xt = build_preprocessor().fit_transform(X)
        assert Xt.shape[0] == len(X), "Row count changed after transform"
        assert Xt.shape[1] > 0,       "No output columns produced"

    def test_fit_transform_no_nulls_in_output(self, minimal_df):
        df = minimal_df.copy()
        df.iloc[0, 0] = np.nan
        X, _ = split_features_target(df)
        Xt = build_preprocessor().fit_transform(X)
        assert not np.isnan(Xt).any(), "NaNs remain after preprocessing"

    def test_transform_unseen_categorical_raises(self, minimal_df):
        X_train, _ = split_features_target(minimal_df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)

        X_test, _ = split_features_target(minimal_df.copy())
        X_test = X_test.copy()
        X_test.loc[X_test.index[0], "hotel"] = "Space Hotel"  # unseen category

        with pytest.raises(ValueError):
            preprocessor.transform(X_test)

    def test_repeated_fit_transform_consistent(self, minimal_df):
        X, _ = split_features_target(minimal_df)

        Xt1 = build_preprocessor().fit_transform(X)
        Xt2 = build_preprocessor().fit_transform(X)

        np.testing.assert_array_almost_equal(Xt1, Xt2)
