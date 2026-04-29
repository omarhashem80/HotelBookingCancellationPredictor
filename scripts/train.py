from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from scripts.preprocess import run_preprocess
from src.config.logging import configure_logging
from src.config.settings import get_settings
from src.evaluation.business_metrics import (
    cost_sensitive_score,
    false_negative_rate,
    revenue_loss_estimate,
)
from src.models.trainer import train_single_model
from src.tracking.logger import log_metrics, log_model, log_params
from src.tracking.mlflow_config import setup_mlflow
from src.utils.helpers import set_seed
from src.utils.io import load_csv, save_model
from src.visualization.model_plots import plot_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cancellation prediction models")
    parser.add_argument(
        "--models",
        type=str,
        default="baseline,logistic,random_forest,xgboost,catboost",
        help="Comma-separated model list",
    )
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a lightweight training configuration for smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    set_seed(settings.random_state)
    setup_mlflow(settings.mlflow_tracking_uri)

    root = settings.project_root
    processed_path = root / "data/processed/hotel_bookings_processed.csv"
    if not processed_path.exists():
        run_preprocess()

    df = load_csv(processed_path)
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    cv = 2 if args.fast else args.cv
    tune_profile = "fast" if args.fast else "full"
    if args.fast and "baseline" not in selected_models:
        selected_models = ["baseline", *selected_models[:1]]

    all_results: list[dict] = []
    best_f1 = -1.0
    best_artifact = root / "reports/best_model_metrics.json"
    best_model_path = root / "reports/best_model.pkl"

    for model_name in selected_models:
        with mlflow.start_run(run_name=model_name):
            result = train_single_model(
                df=df,
                model_name=model_name,
                target_col=settings.target_column,
                random_state=settings.random_state,
                cv=cv,
                tune_profile=tune_profile,
            )

            business_fnr = false_negative_rate(
                pd.Series(result.y_true), pd.Series(result.predictions)
            )
            business_cost = cost_sensitive_score(
                pd.Series(result.y_true), pd.Series(result.predictions)
            )
            metrics = {
                **result.metrics,
                "false_negative_rate": business_fnr,
                "cost_sensitive_score": business_cost,
            }

            if {"adr", "total_nights"}.issubset(result.X_test.columns):
                revenue_loss = revenue_loss_estimate(
                    pd.Series(result.y_true),
                    pd.Series(result.predictions),
                    result.X_test["adr"],
                    result.X_test["total_nights"],
                )
                metrics["revenue_loss_estimate"] = revenue_loss

            log_params({"model_name": model_name, **result.best_params})
            log_metrics(metrics)
            log_model(result.best_model, artifact_path=f"models/{model_name}")

            run_summary = {"model": model_name, **metrics}
            all_results.append(run_summary)

            if metrics.get("f1", 0.0) > best_f1:
                best_f1 = metrics["f1"]
                best_artifact.parent.mkdir(parents=True, exist_ok=True)
                best_artifact.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
                save_model(result.best_model, best_model_path)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(root / "reports/model_results.csv", index=False)
    plot_model_comparison(results_df, root / "reports/figures")
    print("Training complete. Best model stored in reports/best_model.pkl")


if __name__ == "__main__":
    configure_logging()
    main()
