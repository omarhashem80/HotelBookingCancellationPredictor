import pandas as pd
import pytest

from src.data.gx_validation import (
    compute_outliers_iqr,
    check_date_consistency,
    check_reservation_consistency,
    _append_value_counts,
)


class TestOutliersIQR:
    def test_no_outliers(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        q1, q3, iqr, lower, upper, count, pct = compute_outliers_iqr(s)
        assert count == 0
        assert pct == 0.0
        assert q1 < q3
        assert iqr == q3 - q1
        assert lower == q1 - 1.5 * iqr
        assert upper == q3 + 1.5 * iqr

    def test_with_outliers(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        _, _, _, _, upper, count, pct = compute_outliers_iqr(s)
        assert count > 0
        assert pct > 0.0
        assert 100 > upper


class TestDateConsistency:
    def test_clean(self):
        df = pd.DataFrame(
            {
                "arrival_date": ["2016-03-15", "2017-07-01"],
                "arrival_date_year": [2016, 2017],
                "arrival_date_month": ["March", "July"],
                "arrival_date_day_of_month": [15, 1],
            }
        )
        issues = check_date_consistency(df)
        assert len(issues) == 0

    def test_year_mismatch(self):
        df = pd.DataFrame(
            {
                "arrival_date": ["2016-03-15"],
                "arrival_date_year": [2017],
                "arrival_date_month": ["March"],
                "arrival_date_day_of_month": [15],
            }
        )
        issues = check_date_consistency(df)
        assert any("year" in i for i in issues)

    def test_month_mismatch(self):
        df = pd.DataFrame(
            {
                "arrival_date": ["2016-03-15"],
                "arrival_date_year": [2016],
                "arrival_date_month": ["April"],
                "arrival_date_day_of_month": [15],
            }
        )
        issues = check_date_consistency(df)
        assert any("month" in i for i in issues)

    def test_day_mismatch(self):
        df = pd.DataFrame(
            {
                "arrival_date": ["2016-03-15"],
                "arrival_date_year": [2016],
                "arrival_date_month": ["March"],
                "arrival_date_day_of_month": [20],
            }
        )
        issues = check_date_consistency(df)
        assert any("day" in i for i in issues)

    def test_missing_columns(self):
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        issues = check_date_consistency(df)
        assert len(issues) == 0


class TestReservationConsistency:
    def test_clean(self):
        df = pd.DataFrame(
            {
                "is_canceled": [1, 0, 0],
                "reservation_status": ["Canceled", "Check-Out", "No-Show"],
                "adults": [2, 1, 1],
                "children": [0, 1, 0],
                "babies": [0, 0, 0],
            }
        )
        issues = check_reservation_consistency(df)
        assert len(issues) == 0

    def test_canceled_but_checkout(self):
        df = pd.DataFrame(
            {
                "is_canceled": [1],
                "reservation_status": ["Check-Out"],
                "adults": [2],
                "children": [0],
                "babies": [0],
            }
        )
        issues = check_reservation_consistency(df)
        assert any("is_canceled=1 but reservation_status=Check-Out" in i for i in issues)

    def test_not_canceled_but_canceled_status(self):
        df = pd.DataFrame(
            {
                "is_canceled": [0],
                "reservation_status": ["Canceled"],
                "adults": [1],
                "children": [0],
                "babies": [0],
            }
        )
        issues = check_reservation_consistency(df)
        assert any("is_canceled=0 but reservation_status=Canceled" in i for i in issues)

    def test_zero_guests(self):
        df = pd.DataFrame(
            {
                "is_canceled": [0],
                "reservation_status": ["Check-Out"],
                "adults": [0],
                "children": [0],
                "babies": [0],
            }
        )
        issues = check_reservation_consistency(df)
        assert any("zero total guests" in i for i in issues)

    def test_holiday_inconsistency(self):
        df = pd.DataFrame(
            {
                "is_canceled": [0],
                "reservation_status": ["Check-Out"],
                "adults": [1],
                "children": [0],
                "babies": [0],
                "is_holiday": [1],
                "days_to_next_holiday": [5],
                "days_from_last_holiday": [3],
            }
        )
        issues = check_reservation_consistency(df)
        assert any("is_holiday=1" in i for i in issues)


class TestAppendValueCounts:
    def test_basic(self):
        lines = []
        s = pd.Series(["A", "A", "B"])
        _append_value_counts(lines, "Test", s, 3)
        text = "\n".join(lines)
        assert "Test" in text
        assert "A" in text
        assert "B" in text