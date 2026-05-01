import unittest
import pandas as pd

from src.data.gx_validation import (
    compute_outliers_iqr,
    check_date_consistency,
    check_reservation_consistency,
    _append_value_counts,
)


class TestOutliersIQR(unittest.TestCase):

    def test_no_outliers(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        q1, q3, iqr, lower, upper, count, pct = compute_outliers_iqr(s)
        self.assertEqual(count, 0)
        self.assertEqual(pct, 0.0)
        self.assertLess(q1, q3)
        self.assertEqual(iqr, q3 - q1)
        self.assertEqual(lower, q1 - 1.5 * iqr)
        self.assertEqual(upper, q3 + 1.5 * iqr)

    def test_with_outliers(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        _, _, _, lower, upper, count, pct = compute_outliers_iqr(s)
        self.assertGreater(count, 0)
        self.assertGreater(pct, 0.0)
        self.assertGreater(100, upper)


class TestDateConsistency(unittest.TestCase):

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
        self.assertEqual(len(issues), 0)

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
        self.assertTrue(any("year" in i for i in issues))

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
        self.assertTrue(any("month" in i for i in issues))

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
        self.assertTrue(any("day" in i for i in issues))

    def test_missing_columns(self):
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        issues = check_date_consistency(df)
        self.assertEqual(len(issues), 0)


class TestReservationConsistency(unittest.TestCase):

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
        self.assertEqual(len(issues), 0)

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
        self.assertTrue(any("is_canceled=1 but reservation_status=Check-Out" in i for i in issues))

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
        self.assertTrue(any("is_canceled=0 but reservation_status=Canceled" in i for i in issues))

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
        self.assertTrue(any("zero total guests" in i for i in issues))

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
        self.assertTrue(any("is_holiday=1" in i for i in issues))


class TestAppendValueCounts(unittest.TestCase):

    def test_basic(self):
        lines = []
        s = pd.Series(["A", "A", "B"])
        _append_value_counts(lines, "Test", s, 3)
        text = "\n".join(lines)
        self.assertIn("Test", text)
        self.assertIn("A", text)
        self.assertIn("B", text)
