import requests
import time
import json
import os
import bisect

import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv
import pycountry
from tqdm import tqdm
from loguru import logger

load_dotenv()


API_KEY = os.getenv("API_KEY_CALENDERIFIC")
API_BASE_URL = "https://calendarific.com/api/v2/holidays"
CACHE_DIR = Path("../holiday_cache")
RATE_LIMIT_SEC = 1.0


def alpha3_to_alpha2(code_3):
    if pd.isna(code_3) or str(code_3).strip() in ("", "NULL"):
        return None
    code_3 = str(code_3).strip().upper()
    try:
        country = pycountry.countries.get(alpha_3=code_3)
        if country:
            return country.alpha_2
    except Exception:
        pass
    return None


def load_and_prepare(path):
    df = pd.read_csv(path)
    month_map = {
        m: i
        for i, m in enumerate(
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ],
            1,
        )
    }
    df["arrival_date_month_num"] = df["arrival_date_month"].map(month_map)
    df["arrival_date"] = pd.to_datetime(
        df[
            [
                "arrival_date_year",
                "arrival_date_month_num",
                "arrival_date_day_of_month",
            ]
        ].rename(
            columns={
                "arrival_date_year": "year",
                "arrival_date_month_num": "month",
                "arrival_date_day_of_month": "day",
            }
        ),
        errors="coerce",
    )
    invalid_dates = df["arrival_date"].isna().sum()
    if invalid_dates > 0:
        logger.warning("{} rows could not parse arrival_date", invalid_dates)
    logger.info("Converting country codes (alpha-3 -> alpha-2)")
    df["country_alpha2"] = df["country"].apply(alpha3_to_alpha2)
    unmapped = df["country_alpha2"].isna().sum()
    logger.warning("{} rows have unmapped or missing country codes", unmapped)
    return df


def _write_cache(cache, dates):
    with open(cache, "w") as f:
        json.dump([d.isoformat() for d in dates], f)


def _read_cache(cache):
    if not cache.exists():
        return None
    with open(cache, "r") as f:
        return [date.fromisoformat(d) for d in json.load(f)]


def _parse_date(holiday_entry):
    date_obj = holiday_entry.get("date", {})
    iso_date = (
        date_obj.get("iso", "")
        if isinstance(date_obj, dict)
        else date_obj if isinstance(date_obj, str) else None
    )
    if iso_date:
        try:
            return date.fromisoformat(str(iso_date)[:10])
        except (ValueError, TypeError):
            pass
    if isinstance(date_obj, dict):
        dt = date_obj.get("datetime", {})
        if isinstance(dt, dict):
            try:
                return date(int(dt["year"]), int(dt["month"]), int(dt["day"]))
            except (KeyError, TypeError, ValueError):
                pass
    return None


def _extract_holidays(data, country_a2, year):
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    error_code = meta.get("code", 200) if isinstance(meta, dict) else 200
    if error_code != 200:
        error_detail = meta.get("error_detail", "Unknown error")
        logger.warning(
            "API error {} for {}/{}: {}",
            error_code,
            country_a2,
            year,
            error_detail,
        )
        return None
    response_body = data.get("response", {})
    if isinstance(response_body, dict):
        return response_body.get("holidays", [])
    if isinstance(response_body, list):
        return response_body

    logger.warning(
        "Unexpected response format for {}/{}: type={}",
        country_a2,
        year,
        type(response_body).__name__,
    )
    return None


def fetch_holidays(country_a2, year):
    cache = _cache_path(country_a2, year)
    cached = _read_cache(cache)
    if cached is not None:
        return cached
    params = {"api_key": API_KEY, "country": country_a2, "year": year}
    try:
        resp = requests.get(API_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning("API request error for {}/{}: {}", country_a2, year, e)
        _write_cache(cache, [])
        return []
    holidays_raw = _extract_holidays(data, country_a2, year)
    if not holidays_raw:
        _write_cache(cache, [])
        return []
    # parse dates
    holiday_dates = sorted(
        {
            d
            for h in holidays_raw
            if isinstance(h, dict) and (d := _parse_date(h)) is not None
        }
    )
    _write_cache(cache, holiday_dates)
    return holiday_dates


CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(country_a2, year):
    return CACHE_DIR / f"{country_a2}_{year}.json"


def fetch_all_holidays(df):
    pairs = set()
    for _, row in df.dropna(
        subset=["country_alpha2", "arrival_date"]
    ).iterrows():
        a2 = row["country_alpha2"]
        year = int(row["arrival_date"].year)
        pairs.add((a2, year))
        pairs.add((a2, year - 1))
        pairs.add((a2, year + 1))
    pairs = {(a2, y) for a2, y in pairs if 2000 <= y <= 2049}
    holiday_dict = {}
    iterator = tqdm(sorted(pairs), desc="Fetching")
    api_calls = 0
    errors = 0
    for a2, year in iterator:
        cache = _cache_path(a2, year)
        was_cached = cache.exists()
        try:
            holidays = fetch_holidays(a2, year)
            holiday_dict[(a2, year)] = holidays
        except Exception as e:
            logger.warning("Unexpected error for {}/{}: {}", a2, year, e)
            holiday_dict[(a2, year)] = []
            errors += 1
            with open(_cache_path(a2, year), "w") as f:
                json.dump([], f)
        if not was_cached:
            api_calls += 1
            time.sleep(RATE_LIMIT_SEC)
    logger.info("Holiday fetch done: pairs={}, api_calls={}, errors={}", len(pairs), api_calls, errors)
    return holiday_dict


def save_merged_dataset(df: pd.DataFrame, output_path: str):
    # intermediate columns
    cols_to_drop = ["arrival_date_month_num", "country_alpha2"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    if "arrival_date" in df.columns:
        df["arrival_date"] = df["arrival_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False)
    logger.info("Saved merged dataset: rows={}, cols={}, path={}", df.shape[0], df.shape[1], output_path)


def fill_new_df(df, holiday_dict):
    results = []
    total = len(df)
    iterator = tqdm(df.iterrows(), total=total, desc="Processing")
    for idx, row in iterator:
        arrival = row["arrival_date"]
        a2 = row["country_alpha2"]
        year = (
            int(row["arrival_date_year"])
            if pd.notna(row["arrival_date_year"])
            else None
        )
        if pd.isna(arrival) or a2 is None or year is None:
            results.append((np.nan, np.nan, np.nan))
        else:
            results.append(
                calc_holiday_features(arrival, a2, holiday_dict, year)
            )
    df["is_holiday"] = [r[0] for r in results]
    df["days_to_next_holiday"] = [r[1] for r in results]
    df["days_from_last_holiday"] = [r[2] for r in results]
    df["is_holiday"] = df["is_holiday"].astype("Int64")
    df["days_to_next_holiday"] = df["days_to_next_holiday"].astype("Int64")
    df["days_from_last_holiday"] = df["days_from_last_holiday"].astype("Int64")
    logger.info("Holiday features added: rows={}", len(df))
    return df


def calc_holiday_features(arrival, country_a2, holiday_dict, year):
    if pd.isna(arrival) or country_a2 is None:
        return np.nan, np.nan, np.nan
    all_holidays = []
    for y in [year - 1, year, year + 1]:
        all_holidays.extend(holiday_dict.get((country_a2, y), []))
    if not all_holidays:
        return np.nan, np.nan, np.nan
    all_holidays = sorted(set(all_holidays))
    arrival_d = (
        arrival.date()
        if isinstance(arrival, (datetime, pd.Timestamp))
        else arrival
    )
    is_hol = 1 if arrival_d in all_holidays else 0
    idx = bisect.bisect_right(all_holidays, arrival_d)
    if idx < len(all_holidays):
        days_to_next = (all_holidays[idx] - arrival_d).days
    else:
        days_to_next = np.nan
    idx_left = bisect.bisect_left(all_holidays, arrival_d) - 1
    if idx_left >= 0:
        days_from_last = (arrival_d - all_holidays[idx_left]).days
    else:
        days_from_last = np.nan
    if is_hol == 1:
        days_from_last = 0
    return is_hol, days_to_next, days_from_last
