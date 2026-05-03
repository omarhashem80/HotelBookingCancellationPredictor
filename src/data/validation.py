import pandas as pd


def _outlier_summary(df: pd.DataFrame) -> dict[str, int]:
    summary: dict[str, int] = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        summary[col] = int(outliers)
    return summary
