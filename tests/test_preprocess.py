# import pandas as pd
#
# from src.data.cleaning import clean_dtypes
# from src.data.preprocess import build_preprocessor, split_features_target
#
#
# def _make_preprocess_df() -> pd.DataFrame:
#     return pd.DataFrame(
#         {
#             "lead_time": [10, 40, 15],
#             "adr": [80.0, 120.0, 95.0],
#             "hotel": ["resort hotel", "city hotel", "resort hotel"],
#             "reserved_room_type": ["A", "B", "A"],
#             "assigned_room_type": ["A", "B", "B"],
#             "arrival_date": ["2015-07-01", "2015-07-02", "2015-07-03"],
#             "reservation_status_date": [
#                 "2015-07-02",
#                 "2015-07-03",
#                 "2015-07-04",
#             ],
#             "is_canceled": [0, 1, 0],
#         }
#     )
#
#
# def test_split_features_target_separates_target() -> None:
#     df = _make_preprocess_df()
#     X, y = split_features_target(df)
#     assert "is_canceled" not in X.columns
#     assert y.name == "is_canceled"
#     assert len(X) == len(y)
#
#
# def test_build_preprocessor_fit_transform_shape() -> None:
#     df = clean_dtypes(_make_preprocess_df())
#     X, _ = split_features_target(df)
#     preprocessor = build_preprocessor(X)
#     transformed = preprocessor.fit_transform(X)
#     transformer_names = [name for name, _, _ in preprocessor.transformers]
#
#     assert transformed.shape[0] == len(X)
#     assert transformed.shape[1] > 0
#     assert set(["num", "cat", "month"]).issubset(transformer_names)
