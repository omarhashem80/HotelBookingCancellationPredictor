# flake8: noqa
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Captured from the proposal rubric so the report aligns with expected validation/testing scope.
PROPOSAL_ALIGNMENT_REQUIREMENTS = [
    "Document row/column counts, missing values, duplicates, dtypes, and quality issues.",
    "Cover validation dimensions comprehensively and justify quality issue handling.",
    "Validate split strategy and guard against leakage across pipeline stages.",
    "Track class distribution and compare technical and business-relevant impacts.",
]


NON_NEGATIVE_COLUMNS = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "days_to_next_holiday",
    "days_from_last_holiday",
    "total_guests",
    "total_nights",
    "holiday_distance_min",
]


RANGE_RULES: dict[str, tuple[float, float]] = {
    "is_canceled": (0, 1),
    "is_repeated_guest": (0, 1),
    "room_change": (0, 1),
    "arrival_date_week_number": (1, 53),
    "arrival_date_day_of_month": (1, 31),
}


LEAKY_NAME_PATTERNS = [
    "status",
    "cancel",
    "outcome",
    "target",
    "label",
    "result",
    "checkout",
]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _to_float(value: Any) -> float:
    """Normalize pandas/numpy scalar-like values into plain float."""
    array = np.asarray(value)
    if array.size == 0:
        return float("nan")
    return float(array.reshape(-1)[0])


def _binary_target_series(
    df: pd.DataFrame, target_col: str
) -> pd.Series | None:
    if target_col not in df.columns:
        return None

    series = df[target_col]
    series_numeric = pd.to_numeric(series, errors="coerce")
    unique_numeric = set(series_numeric.dropna().unique().tolist())

    if unique_numeric.issubset({0, 1}) and unique_numeric:
        return series_numeric.astype(float)

    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "not_canceled": 0,
        "canceled": 1,
    }
    mapped = cleaned.map(mapping)
    mapped_unique = set(mapped.dropna().unique().tolist())
    if mapped_unique.issubset({0, 1}) and mapped_unique:
        return mapped.astype(float)

    if len(series.dropna().unique()) == 2:
        as_category = series.astype("category")
        codes = as_category.cat.codes.astype(float)
        codes[codes < 0] = np.nan
        return codes

    return None


def schema_validation_profile(
    df: pd.DataFrame, target_col: str
) -> dict[str, Any]:
    dtype_map = {col: str(dtype) for col, dtype in df.dtypes.items()}
    type_counts = pd.Series(dtype_map.values()).value_counts().to_dict()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    datetime_like_columns: list[str] = []
    for col in df.columns:
        if "date" in col.lower() or col.lower().endswith("_at"):
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.8:
                datetime_like_columns.append(col)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": dtype_map,
        "dtype_counts": type_counts,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_like_columns": datetime_like_columns,
        "target_column_present": target_col in df.columns,
    }


def completeness_profile(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    row_count = len(df)
    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_counts / max(row_count, 1)).round(6)

    missing_columns = [col for col, cnt in missing_counts.items() if cnt > 0]
    rows_with_any_missing = int(df.isna().any(axis=1).sum())

    missing_by_target: dict[str, dict[str, float]] = {}
    if target_col in df.columns:
        y = _binary_target_series(df, target_col)
        if y is not None:
            grouped = df.assign(_target_binary=y).groupby(
                "_target_binary", dropna=True
            )
            for target_value, group in grouped:
                group_missing = group.isna().mean().round(6).to_dict()
                missing_by_target[str(target_value)] = {
                    str(col): float(val) for col, val in group_missing.items()
                }

    return {
        "total_missing_cells": int(df.isna().sum().sum()),
        "rows_with_any_missing": rows_with_any_missing,
        "rows_with_any_missing_ratio": _safe_ratio(
            rows_with_any_missing, row_count
        ),
        "missing_columns_count": len(missing_columns),
        "missing_by_column_count": missing_counts.to_dict(),
        "missing_by_column_ratio": missing_pct.to_dict(),
        "missing_by_target_ratio": missing_by_target,
    }


def uniqueness_profile(df: pd.DataFrame) -> dict[str, Any]:
    row_count = len(df)
    duplicated_rows = int(df.duplicated().sum())
    unique_rows = int(row_count - duplicated_rows)

    unique_ratios = {
        col: _safe_ratio(df[col].nunique(dropna=True), row_count)
        for col in df.columns
    }

    candidate_unique_columns = [
        col
        for col in df.columns
        if df[col].notna().all()
        and df[col].nunique(dropna=True) == row_count
        and row_count > 0
    ]
    high_cardinality_columns = [
        col
        for col, ratio in unique_ratios.items()
        if ratio >= 0.95 and row_count > 0
    ]

    return {
        "duplicate_row_count": duplicated_rows,
        "duplicate_row_ratio": _safe_ratio(duplicated_rows, row_count),
        "unique_row_count": unique_rows,
        "candidate_unique_columns": candidate_unique_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "column_unique_ratio": unique_ratios,
    }


def _range_invalid_count(series: pd.Series, lower: float, upper: float) -> int:
    values = pd.to_numeric(series, errors="coerce")
    invalid_mask = values.notna() & ((values < lower) | (values > upper))
    return int(invalid_mask.sum())


def validity_profile(df: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    row_count = len(df)

    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = int((values.notna() & (values < 0)).sum())
            checks[f"{col}_non_negative"] = {
                "failed_rows": invalid,
                "failed_ratio": _safe_ratio(invalid, row_count),
            }

    for col, (low, high) in RANGE_RULES.items():
        if col in df.columns:
            invalid = _range_invalid_count(df[col], low, high)
            checks[f"{col}_within_{int(low)}_{int(high)}"] = {
                "failed_rows": invalid,
                "failed_ratio": _safe_ratio(invalid, row_count),
            }

    if "children" in df.columns:
        child_values = pd.to_numeric(df["children"], errors="coerce")
        fractional = int(
            (child_values.notna() & ((child_values % 1) != 0)).sum()
        )
        checks["children_integer_like"] = {
            "failed_rows": fractional,
            "failed_ratio": _safe_ratio(fractional, row_count),
        }

    if "country" in df.columns:
        country_series = df["country"].astype(str).str.strip()
        blank = int((country_series == "").sum())
        checks["country_not_blank"] = {
            "failed_rows": blank,
            "failed_ratio": _safe_ratio(blank, row_count),
        }

    total_failed_rows_upper_bound = int(
        sum(v["failed_rows"] for v in checks.values())
    )
    return {
        "checks": checks,
        "checks_failed_count": int(
            sum(v["failed_rows"] > 0 for v in checks.values())
        ),
        "total_failed_rows_upper_bound": total_failed_rows_upper_bound,
    }


def consistency_profile(df: pd.DataFrame) -> dict[str, Any]:
    row_count = len(df)
    checks: dict[str, dict[str, Any]] = {}

    if {"adults", "children", "babies"}.issubset(df.columns):
        guests = (
            pd.to_numeric(df["adults"], errors="coerce").fillna(0)
            + pd.to_numeric(df["children"], errors="coerce").fillna(0)
            + pd.to_numeric(df["babies"], errors="coerce").fillna(0)
        )
        invalid_guest_count = int((guests <= 0).sum())
        checks["guest_count_positive"] = {
            "failed_rows": invalid_guest_count,
            "failed_ratio": _safe_ratio(invalid_guest_count, row_count),
        }

        if "total_guests" in df.columns:
            mismatch = int(
                (
                    pd.to_numeric(df["total_guests"], errors="coerce")
                    != guests
                ).sum()
            )
            checks["total_guests_matches_components"] = {
                "failed_rows": mismatch,
                "failed_ratio": _safe_ratio(mismatch, row_count),
            }

    if {
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "total_nights",
    }.issubset(df.columns):
        nights = pd.to_numeric(
            df["stays_in_weekend_nights"], errors="coerce"
        ).fillna(0) + pd.to_numeric(
            df["stays_in_week_nights"], errors="coerce"
        ).fillna(
            0
        )
        mismatch = int(
            (
                pd.to_numeric(df["total_nights"], errors="coerce") != nights
            ).sum()
        )
        checks["total_nights_matches_components"] = {
            "failed_rows": mismatch,
            "failed_ratio": _safe_ratio(mismatch, row_count),
        }

    if {"reserved_room_type", "assigned_room_type", "room_change"}.issubset(
        df.columns
    ):
        expected_room_change = (
            df["reserved_room_type"].astype(str)
            != df["assigned_room_type"].astype(str)
        ).astype(int)
        mismatch = int(
            (
                pd.to_numeric(df["room_change"], errors="coerce")
                != expected_room_change
            ).sum()
        )
        checks["room_change_consistency"] = {
            "failed_rows": mismatch,
            "failed_ratio": _safe_ratio(mismatch, row_count),
        }

    if {
        "days_to_next_holiday",
        "days_from_last_holiday",
        "holiday_distance_min",
    }.issubset(df.columns):
        expected_min = (
            df[["days_to_next_holiday", "days_from_last_holiday"]]
            .apply(pd.to_numeric, errors="coerce")
            .min(axis=1)
        )
        mismatch = int(
            (
                pd.to_numeric(df["holiday_distance_min"], errors="coerce")
                != expected_min
            ).sum()
        )
        checks["holiday_distance_min_consistency"] = {
            "failed_rows": mismatch,
            "failed_ratio": _safe_ratio(mismatch, row_count),
        }

    if {"is_repeated_guest", "previous_bookings_not_canceled"}.issubset(
        df.columns
    ):
        repeated = pd.to_numeric(
            df["is_repeated_guest"], errors="coerce"
        ).fillna(0)
        prev_noncanceled = pd.to_numeric(
            df["previous_bookings_not_canceled"], errors="coerce"
        ).fillna(0)
        inconsistent = int(((repeated == 0) & (prev_noncanceled > 0)).sum())
        checks["repeated_guest_history_consistency"] = {
            "failed_rows": inconsistent,
            "failed_ratio": _safe_ratio(inconsistent, row_count),
        }

    if {"reservation_status", "is_canceled"}.issubset(df.columns):
        status = df["reservation_status"].astype(str).str.strip().str.lower()
        y = _binary_target_series(df, "is_canceled")
        if y is not None:
            should_be_canceled = status.isin(
                {"canceled", "cancelled", "no-show", "no show"}
            )
            inconsistent = int(
                (
                    should_be_canceled.astype(int) != y.fillna(0).astype(int)
                ).sum()
            )
            checks["reservation_status_target_consistency"] = {
                "failed_rows": inconsistent,
                "failed_ratio": _safe_ratio(inconsistent, row_count),
            }

    return {
        "checks": checks,
        "checks_failed_count": int(
            sum(v["failed_rows"] > 0 for v in checks.values())
        ),
    }


def _numeric_distribution_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    non_na = values.dropna()
    if non_na.empty:
        return {"count": 0}

    quantiles = non_na.quantile(
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).to_dict()
    iqr = float(quantiles[0.75] - quantiles[0.25])
    lower = float(quantiles[0.25] - 1.5 * iqr)
    upper = float(quantiles[0.75] + 1.5 * iqr)
    outlier_rate = float(((non_na < lower) | (non_na > upper)).mean())

    return {
        "count": int(non_na.shape[0]),
        "mean": float(non_na.mean()),
        "std": float(non_na.std(ddof=1)) if len(non_na) > 1 else 0.0,
        "min": float(non_na.min()),
        "max": float(non_na.max()),
        "quantiles": {str(k): float(v) for k, v in quantiles.items()},
        "iqr": iqr,
        "outlier_rate_iqr": outlier_rate,
    }


def _categorical_distribution_summary(
    series: pd.Series, top_n: int
) -> dict[str, Any]:
    values = series.fillna("<MISSING>").astype(str)
    vc = values.value_counts(dropna=False)
    total = int(vc.sum())
    top_values = []
    for val, count in vc.head(top_n).items():
        top_values.append(
            {
                "value": str(val),
                "count": int(count),
                "ratio": _safe_ratio(float(count), float(total)),
            }
        )

    probs = (vc / max(total, 1)).to_numpy(dtype=float)
    entropy = float(-(probs * np.log2(np.clip(probs, 1e-12, None))).sum())

    return {
        "count": total,
        "unique_count": int(vc.shape[0]),
        "top_values": top_values,
        "entropy": entropy,
    }


def distribution_profile(
    df: pd.DataFrame, target_col: str, top_n: int
) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric = {
        col: _numeric_distribution_summary(df[col]) for col in numeric_cols
    }
    categorical = {
        col: _categorical_distribution_summary(df[col], top_n=top_n)
        for col in categorical_cols
    }

    target_distribution: dict[str, Any] = {}
    if target_col in df.columns:
        target_distribution = _categorical_distribution_summary(
            df[target_col], top_n=top_n
        )

    return {
        "numeric": numeric,
        "categorical": categorical,
        "target_distribution": target_distribution,
    }


def statistical_properties_profile(
    df: pd.DataFrame, target_col: str
) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_properties: dict[str, dict[str, float]] = {}
    for col in numeric_df.columns:
        series = pd.to_numeric(numeric_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        numeric_properties[col] = {
            "skewness": _to_float(series.skew()),
            "kurtosis": _to_float(series.kurt()),
            "zero_ratio": float((series == 0).mean()),
            "variance": float(series.var(ddof=1)) if len(series) > 1 else 0.0,
        }

    high_correlation_pairs: list[dict[str, Any]] = []
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr(numeric_only=True)
        cols = list(corr_matrix.columns)
        for i, col_i in enumerate(cols):
            for j in range(i + 1, len(cols)):
                col_j = cols[j]
                corr_val = corr_matrix.loc[col_i, col_j]
                corr_float = _to_float(corr_val)
                if pd.notna(corr_float) and abs(corr_float) >= 0.9:
                    high_correlation_pairs.append(
                        {
                            "col_a": col_i,
                            "col_b": col_j,
                            "abs_corr": abs(corr_float),
                        }
                    )

    target_corr: dict[str, float] = {}
    y = _binary_target_series(df, target_col)
    if y is not None and not numeric_df.empty:
        aligned = numeric_df.assign(_target_binary=y)
        for col in numeric_df.columns:
            pair = aligned[[col, "_target_binary"]].dropna()
            if len(pair) > 2:
                target_corr[col] = float(
                    pair[col].corr(pair["_target_binary"])
                )

    return {
        "numeric_properties": numeric_properties,
        "high_correlation_pairs_abs_gte_0_9": high_correlation_pairs,
        "target_numeric_correlation": target_corr,
    }


def _name_based_leakage_flags(
    columns: list[str], target_col: str
) -> list[str]:
    flags: list[str] = []
    for col in columns:
        lower = col.lower()
        if col == target_col:
            continue
        if any(pattern in lower for pattern in LEAKY_NAME_PATTERNS):
            flags.append(col)
    return flags


def _categorical_purity_flags(
    df: pd.DataFrame,
    target_col: str,
    purity_threshold: float,
    min_support: int,
) -> list[dict[str, Any]]:
    y = _binary_target_series(df, target_col)
    if y is None:
        return []

    flags: list[dict[str, Any]] = []
    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in cat_cols:
        if col == target_col:
            continue
        temp = pd.DataFrame(
            {"feature": df[col].astype(str), "target": y}
        ).dropna()
        if temp.empty:
            continue

        grouped = (
            temp.groupby("feature")["target"]
            .agg(["count", "mean"])
            .reset_index()
        )
        risky = grouped[
            (grouped["count"] >= min_support)
            & (
                (grouped["mean"] >= purity_threshold)
                | (grouped["mean"] <= 1 - purity_threshold)
            )
        ]
        if not risky.empty:
            rows = []
            for _, r in (
                risky.sort_values("count", ascending=False).head(5).iterrows()
            ):
                rows.append(
                    {
                        "category": str(r["feature"]),
                        "support": int(r["count"]),
                        "target_rate": float(r["mean"]),
                    }
                )
            flags.append({"column": col, "risky_categories": rows})
    return flags


def leakage_profile(
    df: pd.DataFrame,
    target_col: str,
    corr_threshold: float,
    purity_threshold: float,
    min_support: int,
) -> dict[str, Any]:
    row_count = len(df)
    y = _binary_target_series(df, target_col)

    suspicious_by_name = _name_based_leakage_flags(
        list(df.columns), target_col
    )

    near_perfect_numeric: list[dict[str, Any]] = []
    if y is not None:
        numeric_cols = [
            col
            for col in df.select_dtypes(include=["number"]).columns.tolist()
            if col != target_col
        ]
        for col in numeric_cols:
            pair = pd.DataFrame(
                {"x": pd.to_numeric(df[col], errors="coerce"), "y": y}
            ).dropna()
            if len(pair) < 3:
                continue
            corr = float(pair["x"].corr(pair["y"]))
            if pd.notna(corr) and abs(corr) >= corr_threshold:
                near_perfect_numeric.append(
                    {
                        "column": col,
                        "abs_corr_with_target": abs(corr),
                    }
                )

    id_like_columns = []
    for col in df.columns:
        unique_ratio = _safe_ratio(df[col].nunique(dropna=True), row_count)
        if unique_ratio >= 0.98:
            id_like_columns.append(
                {"column": col, "unique_ratio": unique_ratio}
            )

    categorical_purity = _categorical_purity_flags(
        df=df,
        target_col=target_col,
        purity_threshold=purity_threshold,
        min_support=min_support,
    )

    return {
        "suspicious_column_names": suspicious_by_name,
        "near_perfect_numeric_correlation": near_perfect_numeric,
        "high_purity_categorical_segments": categorical_purity,
        "id_like_columns": id_like_columns,
        "risk_note": (
            "Columns with status/outcome semantics or very strong target association should be excluded "
            "from feature sets before train/validation/test splitting."
        ),
    }


def _total_variation_distance(raw: pd.Series, processed: pd.Series) -> float:
    raw_probs = raw.value_counts(normalize=True)
    processed_probs = processed.value_counts(normalize=True)
    all_idx = raw_probs.index.union(processed_probs.index)
    raw_aligned = raw_probs.reindex(all_idx, fill_value=0.0)
    processed_aligned = processed_probs.reindex(all_idx, fill_value=0.0)
    tv = 0.5 * float(np.abs(raw_aligned - processed_aligned).sum())
    return tv


def raw_vs_processed_profile(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame | None,
    target_col: str,
) -> dict[str, Any]:
    if processed_df is None:
        return {
            "available": False,
            "reason": "processed dataset not provided/found",
        }

    raw_rows = len(raw_df)
    processed_rows = len(processed_df)
    raw_cols = set(raw_df.columns)
    processed_cols = set(processed_df.columns)

    added_columns = sorted(processed_cols - raw_cols)
    removed_columns = sorted(raw_cols - processed_cols)
    common_columns = sorted(raw_cols.intersection(processed_cols))

    target_rate_shift = None
    if target_col in raw_df.columns and target_col in processed_df.columns:
        target_rate_shift = {
            "raw_target_rate": float(
                pd.to_numeric(raw_df[target_col], errors="coerce").mean()
            ),
            "processed_target_rate": float(
                pd.to_numeric(processed_df[target_col], errors="coerce").mean()
            ),
        }
        target_rate_shift["absolute_shift"] = abs(
            target_rate_shift["processed_target_rate"]
            - target_rate_shift["raw_target_rate"]
        )

    numeric_drift: list[dict[str, Any]] = []
    categorical_drift: list[dict[str, Any]] = []
    for col in common_columns:
        raw_series = raw_df[col]
        proc_series = processed_df[col]
        if pd.api.types.is_numeric_dtype(
            raw_series
        ) and pd.api.types.is_numeric_dtype(proc_series):
            raw_num = pd.to_numeric(raw_series, errors="coerce")
            proc_num = pd.to_numeric(proc_series, errors="coerce")
            raw_mean = float(raw_num.mean())
            proc_mean = float(proc_num.mean())
            raw_std = (
                float(raw_num.std(ddof=1))
                if raw_num.notna().sum() > 1
                else 0.0
            )
            z_shift = abs(proc_mean - raw_mean) / max(raw_std, 1e-9)
            numeric_drift.append(
                {
                    "column": col,
                    "raw_mean": raw_mean,
                    "processed_mean": proc_mean,
                    "mean_shift_std_units": float(z_shift),
                }
            )
        else:
            tv = _total_variation_distance(
                raw_series.fillna("<MISSING>").astype(str),
                proc_series.fillna("<MISSING>").astype(str),
            )
            categorical_drift.append(
                {"column": col, "total_variation_distance": float(tv)}
            )

    numeric_drift = sorted(
        numeric_drift, key=lambda x: x["mean_shift_std_units"], reverse=True
    )
    categorical_drift = sorted(
        categorical_drift,
        key=lambda x: x["total_variation_distance"],
        reverse=True,
    )

    return {
        "available": True,
        "raw_rows": raw_rows,
        "processed_rows": processed_rows,
        "row_retention_ratio": _safe_ratio(processed_rows, raw_rows),
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "target_rate_shift": target_rate_shift,
        "top_numeric_distribution_shifts": numeric_drift[:15],
        "top_categorical_distribution_shifts": categorical_drift[:15],
    }


def profile_dataset(
    df: pd.DataFrame, dataset_name: str, target_col: str, top_n: int
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "schema_validation": schema_validation_profile(
            df, target_col=target_col
        ),
        "completeness_testing": completeness_profile(
            df, target_col=target_col
        ),
        "uniqueness_testing": uniqueness_profile(df),
        "validity_testing": validity_profile(df),
        "consistency_testing": consistency_profile(df),
        "distribution_testing": distribution_profile(
            df, target_col=target_col, top_n=top_n
        ),
        "statistical_properties_testing": statistical_properties_profile(
            df, target_col=target_col
        ),
    }


def testing_style_blueprint() -> dict[str, Any]:
    return {
        "recommended_style": "Contract-first data tests with severity tiers",
        "severity_tiers": {
            "blocking_tests": [
                "Schema contract (required columns, dtype family, target presence)",
                "Completeness thresholds on critical columns",
                "Validity rules (binary/range/non-negative constraints)",
                "Core consistency invariants (feature arithmetic and label consistency)",
                "Leakage guards (forbidden columns + suspicious target association)",
            ],
            "monitoring_tests": [
                "Distribution drift monitors against baseline snapshots",
                "Statistical stability monitors (variance/skew/correlation shifts)",
                "Class-balance drift alerts",
            ],
        },
        "how_to_use_this_report": [
            "Convert failed_ratio and drift values into explicit test thresholds in pytest.",
            "Keep hard thresholds strict for business-critical leakage/validity checks.",
            "Use warning thresholds for distribution/statistical drift and track trends over time.",
        ],
    }


def build_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame | None,
    target_col: str,
    top_n: int,
    corr_threshold: float,
    purity_threshold: float,
    min_support: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_column": target_col,
            "proposal_alignment_requirements": PROPOSAL_ALIGNMENT_REQUIREMENTS,
        },
        "raw_dataset_profile": profile_dataset(
            raw_df,
            dataset_name="raw",
            target_col=target_col,
            top_n=top_n,
        ),
        "leakage_risk_assessment": leakage_profile(
            processed_df if processed_df is not None else raw_df,
            target_col=target_col,
            corr_threshold=corr_threshold,
            purity_threshold=purity_threshold,
            min_support=min_support,
        ),
        "testing_style_blueprint": testing_style_blueprint(),
    }

    if processed_df is not None:
        report["processed_dataset_profile"] = profile_dataset(
            processed_df,
            dataset_name="processed",
            target_col=target_col,
            top_n=top_n,
        )
    else:
        report["processed_dataset_profile"] = {
            "available": False,
            "reason": "processed dataset not provided/found",
        }

    report["raw_vs_processed_comparison"] = raw_vs_processed_profile(
        raw_df=raw_df,
        processed_df=processed_df,
        target_col=target_col,
    )

    return _to_builtin(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile dataset properties to design robust data-quality and leakage tests."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/hotel_bookings_with_holidays.csv"),
        help="Path to the raw dataset CSV.",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=Path("data/processed/hotel_bookings_processed.csv"),
        help="Path to processed dataset CSV (optional, if exists).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="is_canceled",
        help="Target column name.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top categories to include in categorical distribution summaries.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold for leakage suspicion.",
    )
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=0.98,
        help="Target purity threshold for suspicious categorical segments.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=50,
        help="Minimum category support for purity-based leakage checks.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/dataset_quality_properties.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path = args.raw_path
    processed_path = args.processed_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    processed_df = (
        pd.read_csv(processed_path) if processed_path.exists() else None
    )

    report = build_report(
        raw_df=raw_df,
        processed_df=processed_df,
        target_col=args.target_col,
        top_n=args.top_n,
        corr_threshold=args.corr_threshold,
        purity_threshold=args.purity_threshold,
        min_support=args.min_support,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print(f"Saved dataset quality properties report to: {args.output_path}")
    print(f"Raw rows: {len(raw_df)} | Raw cols: {len(raw_df.columns)}")
    if processed_df is not None:
        print(
            f"Processed rows: {len(processed_df)} | Processed cols: {len(processed_df.columns)}"
        )
    else:
        print(
            "Processed dataset not found; comparison/leakage uses raw dataset only."
        )


if __name__ == "__main__":
    main()
