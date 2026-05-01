import pandas as pd
import pytest

from src.features.build_features import add_datetime_features, build_features


@pytest.fixture()
def datetime_df() -> pd.DataFrame:
      return pd.DataFrame(
              {
                      "reservation_status_date": ["2015-07-02", "2015-07-05"],
                      "arrival_date": ["2015-07-01", "2015-07-03"],
                      "reservation_status": ["Check-Out", "Canceled"],
                      "lead_time": [10, 40],
              }
      )


def test_add_datetime_features_creates_components(datetime_df) -> None:
      out = add_datetime_features(datetime_df.copy())
      assert "reservation_status_year" in out.columns
      assert "reservation_status_month" in out.columns
      assert "reservation_status_day" in out.columns
      assert "arrival_date_year" in out.columns
      assert "arrival_date_month" in out.columns
      assert "arrival_date_week_number" in out.columns
      assert "arrival_date_day_of_month" in out.columns
      assert out.loc[0, "arrival_date_month"] == 7
      assert out.loc[1, "reservation_status_day"] == 5


def test_build_features_drops_reservation_status_and_is_copy(
      datetime_df,
) -> None:
      original = datetime_df.copy()
      out = build_features(original)
      assert "reservation_status" not in out.columns
      assert "reservation_status" in original.columns
      assert "reservation_status_year" in out.columns
      assert "arrival_date_day_of_month" in out.columns


def test_build_features_no_datetime_column_is_noop() -> None:
      df = pd.DataFrame({"lead_time": [5], "is_canceled": [0]})
      out = build_features(df)
      assert out.columns.tolist() == df.columns.tolist()