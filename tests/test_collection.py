import unittest
from datetime import date, datetime
import numpy as np

from src.data.collection import (
    alpha3_to_alpha2,
    _parse_date,
    _extract_holidays,
    calc_holiday_features,
)


class TestAlpha3ToAlpha2(unittest.TestCase):
    def test_valid_code(self):
        self.assertEqual(alpha3_to_alpha2("USA"), "US")
        self.assertEqual(alpha3_to_alpha2("GBR"), "GB")
        self.assertEqual(alpha3_to_alpha2("DEU"), "DE")

    def test_lowercase_input(self):
        self.assertEqual(alpha3_to_alpha2("usa"), "US")

    def test_with_whitespace(self):
        self.assertEqual(alpha3_to_alpha2("  USA  "), "US")

    def test_none_input(self):
        self.assertIsNone(alpha3_to_alpha2(None))

    def test_empty_string(self):
        self.assertIsNone(alpha3_to_alpha2(""))

    def test_null_string(self):
        self.assertIsNone(alpha3_to_alpha2("NULL"))

    def test_invalid_code(self):
        self.assertIsNone(alpha3_to_alpha2("XYZ"))

    def test_nan_input(self):
        self.assertIsNone(alpha3_to_alpha2(float("nan")))


class TestParseDate(unittest.TestCase):

    def test_iso_string(self):
        entry = {"date": {"iso": "2023-12-25"}}
        self.assertEqual(_parse_date(entry), date(2023, 12, 25))

    def test_iso_with_time(self):
        entry = {"date": {"iso": "2023-12-25T00:00:00"}}
        self.assertEqual(_parse_date(entry), date(2023, 12, 25))

    def test_datetime_dict(self):
        entry = {"date": {"datetime": {"year": 2023, "month": 12, "day": 25}}}
        self.assertEqual(_parse_date(entry), date(2023, 12, 25))

    def test_empty_entry(self):
        self.assertIsNone(_parse_date({}))

    def test_missing_date_key(self):
        self.assertIsNone(_parse_date({"name": "Christmas"}))

    def test_invalid_iso(self):
        entry = {"date": {"iso": "not-a-date"}}
        self.assertIsNone(_parse_date(entry))

    def test_incomplete_datetime_dict(self):
        entry = {"date": {"datetime": {"year": 2023}}}
        self.assertIsNone(_parse_date(entry))


class TestExtractHolidays(unittest.TestCase):

    def test_normal_response(self):
        data = {
            "meta": {"code": 200},
            "response": {"holidays": [{"name": "Christmas"}]},
        }
        result = _extract_holidays(data, "US", 2023)
        self.assertEqual(result, [{"name": "Christmas"}])

    def test_error_code(self):
        data = {"meta": {"code": 401, "error_detail": "Unauthorized"}}
        result = _extract_holidays(data, "US", 2023)
        self.assertIsNone(result)

    def test_response_is_list(self):
        data = {"meta": {"code": 200}, "response": [{"name": "New Year"}]}
        result = _extract_holidays(data, "US", 2023)
        self.assertEqual(result, [{"name": "New Year"}])

    def test_empty_holidays(self):
        data = {"meta": {"code": 200}, "response": {"holidays": []}}
        result = _extract_holidays(data, "US", 2023)
        self.assertEqual(result, [])

    def test_missing_meta(self):
        data = {"response": {"holidays": [{"name": "Test"}]}}
        result = _extract_holidays(data, "US", 2023)
        self.assertEqual(result, [{"name": "Test"}])


class TestCalcHolidayFeatures(unittest.TestCase):

    def setUp(self):
        self.holiday_dict = {
            ("US", 2022): [date(2022, 12, 25), date(2022, 12, 31)],
            ("US", 2023): [date(2023, 1, 1), date(2023, 7, 4), date(2023, 12, 25)],
            ("US", 2024): [date(2024, 1, 1)],
        }

    def test_on_holiday(self):
        arrival = datetime(2023, 7, 4)
        is_hol, to_next, from_last = calc_holiday_features(arrival, "US", self.holiday_dict, 2023)
        self.assertEqual(is_hol, 1)
        self.assertEqual(from_last, 0)

    def test_not_on_holiday(self):
        arrival = datetime(2023, 7, 5)
        is_hol, to_next, from_last = calc_holiday_features(arrival, "US", self.holiday_dict, 2023)
        self.assertEqual(is_hol, 0)
        self.assertEqual(from_last, 1)

    def test_days_to_next(self):
        arrival = datetime(2023, 7, 3)
        is_hol, to_next, from_last = calc_holiday_features(arrival, "US", self.holiday_dict, 2023)
        self.assertEqual(to_next, 1)

    def test_none_arrival(self):
        result = calc_holiday_features(None, "US", self.holiday_dict, 2023)
        self.assertTrue(all(np.isnan(v) for v in result))

    def test_none_country(self):
        arrival = datetime(2023, 7, 4)
        result = calc_holiday_features(arrival, None, self.holiday_dict, 2023)
        self.assertTrue(all(np.isnan(v) for v in result))

    def test_no_holidays_for_country(self):
        arrival = datetime(2023, 7, 4)
        result = calc_holiday_features(arrival, "ZZ", self.holiday_dict, 2023)
        self.assertTrue(all(np.isnan(v) for v in result))

    def test_cross_year_boundary(self):
        arrival = datetime(2023, 12, 26)
        is_hol, to_next, from_last = calc_holiday_features(arrival, "US", self.holiday_dict, 2023)
        self.assertEqual(is_hol, 0)
        self.assertEqual(from_last, 1)  # day after Dec 25
        self.assertEqual(to_next, 6)    # Jan 1 2024


if __name__ == "__main__":
    unittest.main()