import pandas as pd
import numpy as np
import great_expectations as gx

DIST_COLS = [
    "adr",
    "lead_time",
    "adults",
    "children",
    "babies",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "days_to_next_holiday",
    "days_from_last_holiday",
]


def _describe_column(lines, col, series):
    lines.append("{}:".format(col))
    lines.append("  count:{}".format(len(series)))
    lines.append("  mean:{:.4f}".format(series.mean()))
    lines.append("  std:{:.4f}".format(series.std()))
    lines.append("  min:{:.4f}".format(series.min()))
    lines.append("  25th Pct:{:.4f}".format(series.quantile(0.25)))
    lines.append("  median:{:.4f}".format(series.median()))
    lines.append("  75th Pct:{:.4f}".format(series.quantile(0.75)))
    lines.append("  max:{:.4f}".format(series.max()))
    lines.append("  Skewness:{:.4f}".format(series.skew()))


def dist_met(lines, df):
    for col in DIST_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if not np.issubdtype(series.dtype, np.number):
            continue
        _describe_column(lines, col, series)


DIMENSION_MAP = {
    "expect_table_columns_to_match_set": "Consistency",
    "expect_table_column_count_to_equal": "Consistency",
    "expect_column_values_to_match_regex": "Consistency",
    "expect_column_values_to_not_be_null": "Completeness",
    "expect_column_values_to_be_between": "Accuracy",
    "expect_column_values_to_be_in_set": "Accuracy",
    "expect_column_distinct_values_to_be_in_set": "Accuracy",
    "expect_column_unique_value_count_to_be_between": "Uniqueness",
    "expect_column_proportion_of_unique_values_to_be_between": "Uniqueness",
    "expect_table_row_count_to_be_between": "Distribution Profile",
    "expect_column_mean_to_be_between": "Distribution Profile",
    "expect_column_median_to_be_between": "Distribution Profile",
    "expect_column_stdev_to_be_between": "Distribution Profile",
}

ORDERED_DIMENSIONS = [
    "Consistency",
    "Completeness",
    "Accuracy",
    "Uniqueness",
    "Distribution Profile",
]

ALL_DIMENSIONS = ORDERED_DIMENSIONS + [
    "Outlier Detection",
    "Relationships Profile",
]


def get_dimension(exp_type):
    return DIMENSION_MAP.get(exp_type, "Other")


OUTLIER_COLS = [
    "adr",
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
    "required_car_parking_spaces",
    "total_of_special_requests",
    "days_to_next_holiday",
    "days_from_last_holiday",
]


def compute_outliers_iqr(series):
    clean = series.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (clean < lower) | (clean > upper)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = outlier_count / len(clean) * 100 if len(clean) > 0 else 0.0
    return q1, q3, iqr, lower, upper, outlier_count, outlier_pct


def completeness_met(lines, df):
    null_counts = df.isnull().sum()
    total_cells = len(df) * len(df.columns)
    total_nulls = int(null_counts.sum())
    lines.append("Total cells: {}".format(total_cells))
    lines.append("Total null Cells: {}".format(total_nulls))
    lines.append(
        "Overall Completeness:{:.2%}".format(
            1 - total_nulls / total_cells if total_cells > 0 else 0
        )
    )

    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) == 0:
        lines.append("  No null values found in any column")
    else:
        lines.append("  Columns With Nulls:")
        for col_name, count in cols_with_nulls.items():
            pct = count / len(df) * 100
            lines.append(
                "    {:33s} : {} ({:.2f}%)".format(col_name, count, pct)
            )


def holiday_met(lines, df):
    holiday_count = int(df["is_holiday"].sum())
    non_holiday_count = int((df["is_holiday"] == 0).sum())

    lines.append("Holiday Booking Rate:{:.2%}".format(df["is_holiday"].mean()))
    lines.append("Holiday Bookings Count: {}".format(holiday_count))
    lines.append("NonHoliday Bookings:{}".format(non_holiday_count))

    for label, col in [
        ("Days To Next Holiday", "days_to_next_holiday"),
        ("Days From Last Holiday", "days_from_last_holiday"),
    ]:
        s = df[col]
        lines.append("  Mean {}   : {:.2f}".format(label, s.mean()))
        lines.append("  Median {} : {:.2f}".format(label, s.median()))
        lines.append("  Std {}    : {:.2f}".format(label, s.std()))
        lines.append("  Min {}    : {}".format(label, s.min()))
        lines.append("  Max {}    : {}".format(label, s.max()))
        lines.append("")

    return holiday_count, non_holiday_count


def general_met(lines, df):
    for label, col in [("ADR", "adr"), ("Lead Time", "lead_time")]:
        s = df[col]
        lines.append("Mean {}: {:.2f}".format(label, s.mean()))
        lines.append("Median {}: {:.2f}".format(label, s.median()))
        lines.append("Std {}: {:.2f}".format(label, s.std()))
        if label == "ADR":
            lines.append("Min {}: {:.2f}".format(label, s.min()))
        lines.append("Max {}: {:.2f}".format(label, s.max()))

    lines.append("Cancellation Rate: {:.2%}".format(df["is_canceled"].mean()))
    lines.append(
        "Repeated Guest Rate: {:.2%}".format(df["is_repeated_guest"].mean())
    )
    lines.append("Mean Adults: {:.2f}".format(df["adults"].mean()))
    lines.append("Median Adults: {:.2f}".format(df["adults"].median()))
    lines.append("Unique Countries: {}".format(df["country"].nunique()))
    lines.append("Unique Agents: {}".format(df["agent"].nunique()))
    lines.append("Unique ADR Values: {}".format(df["adr"].nunique()))


def uniqe_met(lines, df):
    exact_dupes = int(df.duplicated().sum())
    lines.append("Exact Duplicate Rows: {}".format(exact_dupes))
    lines.append(
        "Exact Duplicate Pct: {:.2%}".format(
            exact_dupes / len(df) if len(df) > 0 else 0
        )
    )

    uniqueness_cols = [
        "country",
        "agent",
        "adr",
        "lead_time",
        "reserved_room_type",
        "assigned_room_type",
    ]
    lines.append("Cardinality per Column:")
    for col in uniqueness_cols:
        if col in df.columns:
            nuniq = df[col].nunique()
            prop = nuniq / len(df) if len(df) > 0 else 0
            lines.append(
                "  {:33s} :{} distinct ({:.4f} proportion)".format(
                    col, nuniq, prop
                )
            )


def _format_outlier_detail(lines, d, df):
    lines.append("[{}] {}".format(d["flag"], d["col"]))
    lines.append("  Q1  : {:.2f}".format(d["q1"]))
    lines.append("  Q3  : {:.2f}".format(d["q3"]))
    lines.append("  IQR : {:.2f}".format(d["iqr"]))
    lines.append("  Lower Fence : {:.2f}".format(d["lower"]))
    lines.append("  Upper Fence : {:.2f}".format(d["upper"]))
    lines.append("  Outliers    : {} ({:.2f}%)".format(d["count"], d["pct"]))
    if d["count"] > 0:
        col_data = df[d["col"]].dropna()
        outlier_vals = col_data[
            (col_data < d["lower"]) | (col_data > d["upper"])
        ]
        lines.append("  Min Outlier : {:.2f}".format(outlier_vals.min()))
        lines.append("  Max Outlier : {:.2f}".format(outlier_vals.max()))
        lines.append("  Mean Outlier: {:.2f}".format(outlier_vals.mean()))


def outlier_met(lines, df):
    total_outlier_flags = 0
    outlier_details = []

    for col in OUTLIER_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        if not np.issubdtype(series.dtype, np.number):
            continue
        q1, q3, iqr, lower, upper, out_count, out_pct = compute_outliers_iqr(
            series
        )
        total_outlier_flags += out_count
        outlier_details.append(
            {
                "col": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
                "count": out_count,
                "pct": out_pct,
                "flag": "OUTLIERS FOUND" if out_count > 0 else "CLEAN",
            }
        )

    lines.append("Total Outlier Flags: {}".format(total_outlier_flags))
    for d in outlier_details:
        _format_outlier_detail(lines, d, df)


def _append_value_counts(lines, label, series, total):
    lines.append("{}:".format(label))
    for name, count in series.value_counts().items():
        pct = count / total * 100
        lines.append("  {:33s} : {} ({:.2f}%)".format(name, count, pct))


def cat_met(lines, df):
    n = len(df)
    for label, col in [
        ("Hotel Type", "hotel"),
        ("Reservation Status", "reservation_status"),
        ("Deposit Type", "deposit_type"),
        ("Customer Type", "customer_type"),
        ("Market Segment", "market_segment"),
        ("Meal Type", "meal"),
    ]:
        _append_value_counts(lines, label, df[col], n)


def _append_corr_pairs(lines, label, pairs):
    lines.append(label)
    if not pairs:
        lines.append("  None found")
    else:
        for col_a, col_b, val in sorted(
            pairs, key=lambda x: abs(x[2]), reverse=True
        ):
            lines.append("  {} <-> {} : {:.4f}".format(col_a, col_b, val))


def corr_met(lines, df):
    corr_cols = [
        "is_canceled",
        "lead_time",
        "adr",
        "adults",
        "children",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "is_repeated_guest",
        "is_holiday",
        "days_to_next_holiday",
        "days_from_last_holiday",
    ]
    available = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[available].corr(method="spearman")

    high_pairs, moderate_pairs = [], []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            val = corr_matrix.loc[available[i], available[j]]
            pair = (available[i], available[j], val)
            if abs(val) >= 0.7:
                high_pairs.append(pair)
            elif abs(val) >= 0.4:
                moderate_pairs.append(pair)

    _append_corr_pairs(lines, "High Correlations (|r| >= 0.7)", high_pairs)
    _append_corr_pairs(
        lines, "Moderate Correlations (0.4 <= |r| < 0.7)", moderate_pairs
    )

    lines.append("Key Target Correlations (vs is_canceled)")
    if "is_canceled" in corr_matrix.columns:
        cancel_corr = (
            corr_matrix["is_canceled"]
            .drop("is_canceled")
            .sort_values(key=lambda x: abs(x), ascending=False)
        )
        for col_name, val in cancel_corr.items():
            lines.append("    {:40s} : {:.4f}".format(col_name, val))

    lines.append("Holiday Feature Correlations")
    holiday_features = [
        "is_holiday",
        "days_to_next_holiday",
        "days_from_last_holiday",
    ]
    target_features = [
        "is_canceled",
        "lead_time",
        "adr",
        "adults",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
    ]
    for hf in holiday_features:
        if hf not in corr_matrix.columns:
            continue
        lines.append("  {}:".format(hf))
        for tf in target_features:
            if tf not in corr_matrix.columns:
                continue
            lines.append(
                "    vs {:35s} : {:.4f}".format(tf, corr_matrix.loc[hf, tf])
            )
        lines.append("")


MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def check_date_consistency(df):
    issues = []
    required = [
        "arrival_date",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    ]
    if not all(c in df.columns for c in required):
        return issues

    parsed = pd.to_datetime(df["arrival_date"], errors="coerce")

    checks = [
        (
            parsed.dt.year != df["arrival_date_year"],
            "arrival_date year vs arrival_date_year mismatch",
        ),
        (
            parsed.dt.month != df["arrival_date_month"].map(MONTH_MAP),
            "arrival_date month vs arrival_date_month mismatch",
        ),
        (
            parsed.dt.day != df["arrival_date_day_of_month"],
            "arrival_date day vs arrival_date_day_of_month mismatch",
        ),
    ]

    if "arrival_date_week_number" in df.columns:
        checks.append(
            (
                parsed.dt.isocalendar().week.astype(int)
                != df["arrival_date_week_number"],
                "arrival_date week vs arrival_date_week_number mismatch",
            )
        )

    for mask, msg in checks:
        count = mask.sum()
        if count > 0:
            issues.append("{}: {} rows".format(msg, count))

    invalid_dates = parsed.isna().sum() - df["arrival_date"].isna().sum()
    if invalid_dates > 0:
        issues.append(
            "arrival_date contains {} unparseable date strings".format(
                invalid_dates
            )
        )

    return issues


def check_reservation_consistency(df):
    issues = []

    if "is_canceled" in df.columns and "reservation_status" in df.columns:
        cross_checks = [
            (
                (df["is_canceled"] == 1)
                & (df["reservation_status"] == "Check-Out"),
                "is_canceled=1 but reservation_status=Check-Out",
            ),
            (
                (df["is_canceled"] == 0)
                & (df["reservation_status"] == "Canceled"),
                "is_canceled=0 but reservation_status=Canceled",
            ),
        ]
        for mask, msg in cross_checks:
            count = mask.sum()
            if count > 0:
                issues.append("{}: {} rows".format(msg, count))

    if all(c in df.columns for c in ["adults", "children", "babies"]):
        zero_guests = (
            (df["adults"].fillna(0) == 0)
            & (df["children"].fillna(0) == 0)
            & (df["babies"].fillna(0) == 0)
        ).sum()
        if zero_guests > 0:
            issues.append(
                "Bookings with zero total guests (adults+children+babies=0): {} rows".format(
                    zero_guests
                )
            )

    if all(
        c in df.columns for c in ["arrival_date", "reservation_status_date"]
    ):
        arr = pd.to_datetime(df["arrival_date"], errors="coerce")
        res = pd.to_datetime(df["reservation_status_date"], errors="coerce")
        checkout_mask = (
            df["reservation_status"] == "Check-Out"
            if "reservation_status" in df.columns
            else pd.Series(False, index=df.index)
        )
        status_before = (checkout_mask & (res < arr)).sum()
        if status_before > 0:
            issues.append(
                "Check-Out status date before arrival date: {} rows".format(
                    status_before
                )
            )

    if all(
        c in df.columns
        for c in [
            "is_holiday",
            "days_to_next_holiday",
            "days_from_last_holiday",
        ]
    ):
        bad = (
            (df["is_holiday"] == 1)
            & (df["days_to_next_holiday"] != 0)
            & (df["days_from_last_holiday"] != 0)
        ).sum()
        if bad > 0:
            issues.append(
                "is_holiday=1 but both days_to_next and days_from_last are nonzero: {} rows".format(
                    bad
                )
            )

    return issues


def _add_regex_expectations(suite):
    regex_specs = [
        ("arrival_date", r"^\d{4}-\d{2}-\d{2}$"),
        ("reservation_status_date", r"^\d{4}-\d{2}-\d{2}$"),
        ("arrival_date_month", r"^[A-Z][a-z]+$"),
        ("hotel", r"^[A-Z][a-z]+ Hotel$"),
        ("country", r"^[A-Z]{2,3}$"),
    ]
    for col, regex in regex_specs:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToMatchRegex(
                column=col, regex=regex
            )
        )


def _add_set_expectations(suite):
    binary_cols = ["is_canceled", "is_repeated_guest", "is_holiday"]
    for col in binary_cols:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=col, value_set=[0, 1]
            )
        )

    set_specs = {
        "arrival_date_year": [2015, 2016, 2017],
        "arrival_date_month": [
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
        "meal": ["BB", "FB", "HB", "SC", "Undefined"],
        "market_segment": [
            "Direct",
            "Corporate",
            "Online TA",
            "Offline TA/TO",
            "Complementary",
            "Groups",
            "Undefined",
            "Aviation",
        ],
        "distribution_channel": [
            "Direct",
            "Corporate",
            "TA/TO",
            "Undefined",
            "GDS",
        ],
        "deposit_type": ["No Deposit", "Refundable", "Non Refund"],
        "customer_type": ["Transient", "Contract", "Transient-Party", "Group"],
        "reservation_status": ["Check-Out", "Canceled", "No-Show"],
        "hotel": ["Resort Hotel", "City Hotel"],
    }
    for col, values in set_specs.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=col, value_set=values
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="hotel", value_set=["Resort Hotel", "City Hotel"]
        )
    )


def _add_range_expectations(suite):
    range_specs = [
        ("lead_time", 0, 800),
        ("arrival_date_week_number", 1, 53),
        ("arrival_date_day_of_month", 1, 31),
        ("adults", 0, 100),
        ("children", 0, 10),
        ("babies", 0, 10),
        ("days_to_next_holiday", 0, 365),
        ("days_from_last_holiday", 0, 365),
    ]
    for col, mn, mx in range_specs:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col, min_value=mn, max_value=mx
            )
        )


def _add_statistical_expectations(suite):
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=119_000, max_value=120_000
        )
    )

    mean_specs = [
        ("adr", 80, 130),
        ("lead_time", 80, 130),
        ("is_canceled", 0.25, 0.45),
        ("is_holiday", 0.0, 0.5),
        ("days_to_next_holiday", 0, 180),
        ("days_from_last_holiday", 0, 180),
    ]
    for col, mn, mx in mean_specs:
        suite.add_expectation(
            gx.expectations.ExpectColumnMeanToBeBetween(
                column=col, min_value=mn, max_value=mx
            )
        )

    median_specs = [
        ("adults", 1, 3),
        ("days_to_next_holiday", 0, 180),
        ("days_from_last_holiday", 0, 180),
    ]
    for col, mn, mx in median_specs:
        suite.add_expectation(
            gx.expectations.ExpectColumnMedianToBeBetween(
                column=col, min_value=mn, max_value=mx
            )
        )

    stdev_specs = [
        ("adr", 30, 150),
        ("lead_time", 50, 150),
        ("days_to_next_holiday", 1, 200),
        ("days_from_last_holiday", 1, 200),
    ]
    for col, mn, mx in stdev_specs:
        suite.add_expectation(
            gx.expectations.ExpectColumnStdevToBeBetween(
                column=col, min_value=mn, max_value=mx
            )
        )


def _add_null_expectations(suite):
    critical_not_null = [
        "hotel",
        "is_canceled",
        "lead_time",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "arrival_date",
        "adults",
        "meal",
        "market_segment",
        "distribution_channel",
        "is_repeated_guest",
        "deposit_type",
        "customer_type",
        "adr",
        "reservation_status",
        "reservation_status_date",
        "is_holiday",
        "days_to_next_holiday",
        "days_from_last_holiday",
    ]
    for col in critical_not_null:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    for col in ["country", "children", "agent"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=col, mostly=0.95
            )
        )


def _add_non_negative_expectations(suite):
    non_neg_cols = [
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
        "required_car_parking_spaces",
        "total_of_special_requests",
        "days_to_next_holiday",
        "days_from_last_holiday",
    ]
    for col in non_neg_cols:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col, min_value=0
            )
        )


def _add_uniqueness_expectations(suite):
    unique_count_specs = [
        ("country", 50, 250),
        ("agent", 50, 600),
        ("adr", 1000, 10000),
        ("reserved_room_type", 5, 15),
        ("assigned_room_type", 5, 15),
    ]
    for col, mn, mx in unique_count_specs:
        suite.add_expectation(
            gx.expectations.ExpectColumnUniqueValueCountToBeBetween(
                column=col, min_value=mn, max_value=mx
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnProportionOfUniqueValuesToBeBetween(
            column="adr", min_value=0.01, max_value=0.10
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnProportionOfUniqueValuesToBeBetween(
            column="lead_time", min_value=0.001, max_value=0.01
        )
    )


EXPECTED_COLUMNS = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "arrival_date",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
    "is_holiday",
    "days_to_next_holiday",
    "days_from_last_holiday",
]


def _build_suite(context):
    suite = context.suites.add(
        gx.ExpectationSuite(name="hotel_validation_suite")
    )

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToEqual(value=36)
    )

    _add_regex_expectations(suite)
    _add_null_expectations(suite)
    _add_non_negative_expectations(suite)
    _add_range_expectations(suite)
    _add_set_expectations(suite)
    _add_uniqueness_expectations(suite)
    _add_statistical_expectations(suite)

    return suite


def _report_header(lines, results):
    lines.append("VALIDATION REPORT")
    passed = sum(1 for r in results.results if r.success)
    failed = sum(1 for r in results.results if not r.success)
    lines.append(
        "expectations: {} total | {} passed | {} failed".format(
            len(results.results), passed, failed
        )
    )


def _group_by_dimension(results):
    grouped = {}
    for exp_result in results.results:
        dim = get_dimension(exp_result.expectation_config.type)
        grouped.setdefault(dim, []).append(exp_result)
    return grouped


def _report_dimension_results(lines, grouped):
    for dim in ORDERED_DIMENSIONS:
        if dim not in grouped:
            continue
        dim_results = grouped[dim]
        passes = sum(1 for r in dim_results if r.success)
        fails = len(dim_results) - passes
        lines.append("DIMENSION: {}".format(dim.upper()))
        lines.append(
            "  {} checks | {} passed | {} failed".format(
                len(dim_results), passes, fails
            )
        )

        for exp_result in dim_results:
            _report_single_expectation(lines, exp_result)

    if "Other" in grouped:
        lines.append("DIMENSION: UNCATEGORIZED")
        for exp_result in grouped["Other"]:
            _report_single_expectation(lines, exp_result)


def _report_single_expectation(lines, exp_result):
    exp_type = exp_result.expectation_config.type
    col = exp_result.expectation_config.kwargs.get("column", "table-level")
    status = "PASS" if exp_result.success else "FAIL"
    lines.append("  [{}] {}".format(status, exp_type))
    lines.append("    column: {}".format(col))

    if not exp_result.success and exp_result.result:
        r = exp_result.result
        if r.get("unexpected_count"):
            lines.append(
                "    Issues: {} unexpected values".format(
                    r["unexpected_count"]
                )
            )
        if r.get("partial_unexpected_list"):
            lines.append(
                "    Sample: {}".format(r["partial_unexpected_list"][:5])
            )
        if r.get("observed_value") is not None:
            lines.append("    Observed: {}".format(r["observed_value"]))


def _report_cross_field_checks(lines, df):
    lines.append("DIMENSION: CONSISTENCY (Cross-Field Checks)")
    date_issues = check_date_consistency(df)
    reservation_issues = check_reservation_consistency(df)

    if not date_issues:
        lines.append("  PASS Date field cross validation")
        lines.append("    All arrival_date components match")
    else:
        for issue in date_issues:
            lines.append("  FAIL {}".format(issue))

    if not reservation_issues:
        lines.append("  PASS Business rule cross-validation")
        lines.append("    All cross field rules consistent")
    else:
        for issue in reservation_issues:
            lines.append("  FAIL {}".format(issue))

    return date_issues, reservation_issues


def _report_dimension_summary(lines, grouped, cross_field_issue_count):
    lines.append("DIMENSION SUMMARY")
    for dim in ALL_DIMENSIONS:
        if dim not in grouped:
            if dim in ["Outlier Detection", "Relationships Profile"]:
                lines.append("  {:25s} : computed below".format(dim))
            else:
                lines.append("  {:25s} : no checks".format(dim))
            continue
        dim_results = grouped[dim]
        passes = sum(1 for r in dim_results if r.success)
        dim_status = "PASSED" if passes == len(dim_results) else "FAILED"
        lines.append(
            "  {:25s} : {} ({}/{} passed)".format(
                dim, dim_status, passes, len(dim_results)
            )
        )

    if cross_field_issue_count == 0:
        lines.append(
            "  {:25s} : PASSED (all cross-field checks clean)".format(
                "Consistency (CrossField)"
            )
        )
    else:
        lines.append(
            "  {:25s} : FAILED ({} issues found)".format(
                "Consistency (CrossField)", cross_field_issue_count
            )
        )


def _report_holiday_comparison(lines, df, holiday_count):
    lines.append("--- Holiday vs Non-Holiday Comparisons")
    comparisons = [
        ("Cancellation Rate", "is_canceled", "{:.2%}"),
        ("Mean ADR", "adr", "{:.2f}"),
        ("Mean Lead Time", "lead_time", "{:.2f}"),
    ]
    for label, col, fmt in comparisons:
        if holiday_count > 0:
            val = df[df["is_holiday"] == 1][col].mean()
            lines.append("  Holiday {} : {}".format(label, fmt.format(val)))
        nh_val = df[df["is_holiday"] == 0][col].mean()
        lines.append("  Non-Holiday {} : {}".format(label, fmt.format(nh_val)))
        lines.append("")


def build_report(results, df):
    lines = []
    _report_header(lines, results)

    grouped = _group_by_dimension(results)
    _report_dimension_results(lines, grouped)

    date_issues, reservation_issues = _report_cross_field_checks(lines, df)

    cross_field_total = len(date_issues) + len(reservation_issues)
    _report_dimension_summary(lines, grouped, cross_field_total)

    lines.append("DATASET STATISTICS")
    lines.append("--General Metrics")
    general_met(lines, df)
    lines.append("--Completeness Metrics")
    completeness_met(lines, df)
    lines.append("---Holiday Feature Metrics")
    holiday_count, _ = holiday_met(lines, df)
    lines.append("---Uniqueness Metrics")
    uniqe_met(lines, df)
    lines.append("---Category Distributions")
    cat_met(lines, df)

    _report_holiday_comparison(lines, df, holiday_count)

    lines.append("OUTLIER DETECTION (IQR Method)")
    outlier_met(lines, df)
    lines.append("Correlation Analysis")
    corr_met(lines, df)
    lines.append("DISTRIBUTION PROFILE SUMMARY")
    dist_met(lines, df)

    return "\n".join(lines)
