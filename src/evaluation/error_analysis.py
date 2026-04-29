from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_breakdown(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def segment_error_analysis(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, dict]:
    """Return misclassification rates by lead time and ADR segments."""
    local = df.copy().reset_index(drop=True)
    local["y_true"] = pd.Series(y_true).reset_index(drop=True)
    local["y_pred"] = pd.Series(y_pred).reset_index(drop=True)
    local["is_error"] = (local["y_true"] != local["y_pred"]).astype(int)

    analysis: dict[str, dict] = {}

    if "lead_time" in local.columns:
        local["lead_time_segment"] = pd.cut(
            local["lead_time"],
            bins=[-1, 30, 90, 180, float("inf")],
            labels=["short", "medium", "long", "very_long"],
        )
        analysis["lead_time"] = (
            local.groupby("lead_time_segment", observed=True)["is_error"].mean().to_dict()
        )

    if "adr" in local.columns:
        local["adr_segment"] = pd.qcut(local["adr"], q=4, duplicates="drop")
        analysis["adr"] = local.groupby("adr_segment", observed=True)["is_error"].mean().to_dict()

    return analysis
