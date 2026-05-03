import json

import pandas as pd
from sklearn.metrics import classification_report
from loguru import logger

from src.config.logging import configure_logging
from src.config.settings import get_settings
from src.evaluation.business_metrics import (
    cost_sensitive_score,
    false_negative_rate,
    revenue_loss_estimate,
)
from src.evaluation.error_analysis import (
    confusion_breakdown,
    segment_error_analysis,
)
from src.evaluation.metrics import calculate_classification_metrics
from src.utils.io import load_csv, load_model
from src.visualization.model_plots import plot_confusion, plot_roc


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, tuple):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> None:
    settings = get_settings()
    root = settings.project_root

    data_path = root / "data/processed/hotel_bookings_processed.csv"
    model_path = root / "reports/best_model.pkl"

    if not data_path.exists() or not model_path.exists():
        raise FileNotFoundError("Processed data or trained model not found. Run make train first.")

    df = load_csv(data_path)
    target_col = settings.target_column
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    logger.info("Loaded processed data: rows={}, cols={}", X_test.shape[0], X_test.shape[1])

    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    standard_metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
    business = {
        "false_negative_rate": false_negative_rate(y_test, pd.Series(y_pred)),
        "cost_sensitive_score": cost_sensitive_score(y_test, pd.Series(y_pred)),
    }

    if {"adr", "total_nights", "deposit_type"}.issubset(X_test.columns):
        business["revenue_loss_estimate"] = revenue_loss_estimate(
            y_test,
            pd.Series(y_pred),
            X_test["adr"],
            X_test["total_nights"],
            X_test["deposit_type"],
        )

    errors = {
        "confusion_breakdown": confusion_breakdown(y_test, pd.Series(y_pred)),
        "segment_error_analysis": segment_error_analysis(X_test, y_test, pd.Series(y_pred)),
    }

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion(y_test, y_pred, reports_dir / "figures")
    if y_prob is not None:
        plot_roc(y_test, y_prob, reports_dir / "figures")

    output = {
        "standard_metrics": standard_metrics,
        "business_metrics": business,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "error_analysis": errors,
    }
    safe_output = _json_safe(output)
    (reports_dir / "evaluation_report.json").write_text(
        json.dumps(safe_output, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved evaluation report to reports/evaluation_report.json")


if __name__ == "__main__":
    configure_logging()
    main()
