from datetime import date, datetime
import numpy as np
import pytest

from src.data.collection import (
    alpha3_to_alpha2,
    _parse_date,
    _extract_holidays,
    calc_holiday_features,
)


class TestAlpha3ToAlpha2:
    @pytest.mark.parametrize(
        "code,expected",
        [
            ("USA", "US"),
            ("GBR", "GB"),
            ("DEU", "DE"),
        ],
    )
    def test_valid_code(self, code, expected):
        assert alpha3_to_alpha2(code) == expected

    def test_lowercase_input(self):
        assert alpha3_to_alpha2("usa") == "US"

    def test_with_whitespace(self):
        assert alpha3_to_alpha2("  USA  ") == "US"

    def test_none_input(self):
        assert alpha3_to_alpha2(None) is None

    def test_empty_string(self):
        assert alpha3_to_alpha2("") is None

    def test_null_string(self):
        assert alpha3_to_alpha2("NULL") is None

    def test_invalid_code(self):
        assert alpha3_to_alpha2("XYZ") is None

    def test_nan_input(self):
        assert alpha3_to_alpha2(float("nan")) is None


class TestParseDate:
    def test_iso_string(self):
        entry = {"date": {"iso": "2023-12-25"}}
        assert _parse_date(entry) == date(2023, 12, 25)

    def test_iso_with_time(self):
        entry = {"date": {"iso": "2023-12-25T00:00:00"}}
        assert _parse_date(entry) == date(2023, 12, 25)

    def test_datetime_dict(self):
        entry = {"date": {"datetime": {"year": 2023, "month": 12, "day": 25}}}
        assert _parse_date(entry) == date(2023, 12, 25)

    def test_empty_entry(self):
        assert _parse_date({}) is None

    def test_missing_date_key(self):
        assert _parse_date({"name": "Christmas"}) is None

    def test_invalid_iso(self):
        entry = {"date": {"iso": "not-a-date"}}
        assert _parse_date(entry) is None

    def test_incomplete_datetime_dict(self):
        entry = {"date": {"datetime": {"year": 2023}}}
        assert _parse_date(entry) is None


class TestExtractHolidays:
    def test_normal_response(self):
        data = {
            "meta": {"code": 200},
            "response": {"holidays": [{"name": "Christmas"}]},
        }
        result = _extract_holidays(data, "US", 2023)
        assert result == [{"name": "Christmas"}]

    def test_error_code(self):
        data = {"meta": {"code": 401, "error_detail": "Unauthorized"}}
        result = _extract_holidays(data, "US", 2023)
        assert result is None

    def test_response_is_list(self):
        data = {"meta": {"code": 200}, "response": [{"name": "New Year"}]}
        result = _extract_holidays(data, "US", 2023)
        assert result == [{"name": "New Year"}]

    def test_empty_holidays(self):
        data = {"meta": {"code": 200}, "response": {"holidays": []}}
        result = _extract_holidays(data, "US", 2023)
        assert result == []

    def test_missing_meta(self):
        data = {"response": {"holidays": [{"name": "Test"}]}}
        result = _extract_holidays(data, "US", 2023)
        assert result == [{"name": "Test"}]


@pytest.fixture
def holiday_dict():
    return {
        ("US", 2022): [date(2022, 12, 25), date(2022, 12, 31)],
        ("US", 2023): [
            date(2023, 1, 1),
            date(2023, 7, 4),
            date(2023, 12, 25),
        ],
        ("US", 2024): [date(2024, 1, 1)],
    }


class TestCalcHolidayFeatures:
    def test_on_holiday(self, holiday_dict):
        arrival = datetime(2023, 7, 4)
        is_hol, to_next, from_last = calc_holiday_features(
            arrival, "US", holiday_dict, 2023
        )
        assert is_hol == 1
        assert from_last == 0

    def test_not_on_holiday(self, holiday_dict):
        arrival = datetime(2023, 7, 5)
        is_hol, to_next, from_last = calc_holiday_features(
            arrival, "US", holiday_dict, 2023
        )
        assert is_hol == 0
        assert from_last == 1

    def test_days_to_next(self, holiday_dict):
        arrival = datetime(2023, 7, 3)
        is_hol, to_next, from_last = calc_holiday_features(
            arrival, "US", holiday_dict, 2023
        )
        assert to_next == 1

    def test_none_arrival(self, holiday_dict):
        result = calc_holiday_features(None, "US", holiday_dict, 2023)
        assert all(np.isnan(v) for v in result)

    def test_none_country(self, holiday_dict):
        arrival = datetime(2023, 7, 4)
        result = calc_holiday_features(arrival, None, holiday_dict, 2023)
        assert all(np.isnan(v) for v in result)

    def test_no_holidays_for_country(self, holiday_dict):
        arrival = datetime(2023, 7, 4)
        result = calc_holiday_features(arrival, "ZZ", holiday_dict, 2023)
        assert all(np.isnan(v) for v in result)

    def test_cross_year_boundary(self, holiday_dict):
        arrival = datetime(2023, 12, 26)
        is_hol, to_next, from_last = calc_holiday_features(
            arrival, "US", holiday_dict, 2023
        )
        assert is_hol == 0
        assert from_last == 1  # day after Dec 25
        assert to_next == 6  # Jan 1 2024