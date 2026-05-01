import argparse
import json
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from loguru import logger

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

from src.features.sampling import get_smote_sampler
from src.features.selection import get_xgb_feature_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cancellation prediction models"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="baseline,logistic,ada_boost,xgboost,catboost",
        help="Comma-separated model list",
    )

    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE oversampling",
    )

    parser.add_argument(
        "--use-selector",
        action="store_true",
        help="Apply feature selection",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = get_settings()
    logger.info("Loaded settings with random_state={}", settings.random_state)
    set_seed(settings.random_state)
    setup_mlflow(settings.mlflow_tracking_uri)

    root = settings.project_root
    processed_path = root / "data/processed/hotel_bookings.csv"

    if not processed_path.exists():
        logger.info("Processed data missing; running preprocessing pipeline")
        run_preprocess()

    df = load_csv(processed_path)
    logger.info("Loaded processed data: rows={}, cols={}", df.shape[0], df.shape[1])

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    logger.info("Training models: {}", ", ".join(selected_models))

    sampler: Optional[object] = (
        get_smote_sampler(settings.random_state) if args.use_smote else None
    )
    if sampler is not None:
        logger.info("SMOTE oversampling enabled")

    selector: Optional[object] = (
        get_xgb_feature_selector(settings.random_state)
        if args.use_selector
        else None
    )
    if selector is not None:
        logger.info("Feature selection enabled")

    all_results: list[dict] = []
    best_metric = "f1"
    best_score = float("-inf")

    best_artifact = root / "reports/best_model_metrics.json"
    best_model_path = root / "reports/best_model.pkl"

    if best_artifact.exists():
        try:
            existing = json.loads(best_artifact.read_text(encoding="utf-8"))
            existing_score = existing.get(best_metric)
            if isinstance(existing_score, (int, float)):
                best_score = float(existing_score)
                logger.info(
                    "Loaded best score from disk: {}={:.4f}",
                    best_metric,
                    best_score,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read existing best metrics: {}", exc)

    for model_name in selected_models:
        with mlflow.start_run(run_name=model_name):
            logger.info("Starting training for model={}", model_name)

            result = train_single_model(
                df=df,
                model_name=model_name,
                target_col=settings.target_column,
                random_state=settings.random_state,
                selector=selector,
                sampler=sampler,
            )

            y_true = pd.Series(result.y_true)
            y_pred = pd.Series(result.predictions)

            business_fnr = false_negative_rate(y_true, y_pred)
            business_cost = cost_sensitive_score(y_true, y_pred)

            metrics = {
                **result.metrics,
                "false_negative_rate": business_fnr,
                "cost_sensitive_score": business_cost,
            }

            if {"adr", "total_nights"}.issubset(result.X_test.columns):
                metrics["revenue_loss_estimate"] = revenue_loss_estimate(
                    y_true,
                    y_pred,
                    result.X_test["adr"],
                    result.X_test["total_nights"],
                )

            log_params({"model_name": model_name, **result.best_params})
            log_metrics(metrics)
            log_model(result.best_model, artifact_path=f"models/{model_name}")

            logger.info(
                "Completed model={} with f1={:.4f}, precision={:.4f}, recall={:.4f}",
                model_name,
                metrics.get("f1", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
            )

            run_summary = {"model": model_name, **metrics}
            all_results.append(run_summary)

            current_score = metrics.get(best_metric)
            if current_score is None:
                logger.warning(
                    "Skipping best-model check for {} because metric '{}' is missing",
                    model_name,
                    best_metric,
                )
            elif current_score > best_score:
                best_score = current_score
                logger.info(
                    "New best model={} with {}={:.4f}",
                    model_name,
                    best_metric,
                    best_score,
                )

                best_artifact.parent.mkdir(parents=True, exist_ok=True)
                best_artifact.write_text(
                    json.dumps(run_summary, indent=2),
                    encoding="utf-8",
                )

                save_model(result.best_model, best_model_path)
            else:
                logger.info(
                    "Model {} did not improve {} (best={:.4f}, current={:.4f})",
                    model_name,
                    best_metric,
                    best_score,
                    current_score,
                )

    results_df = pd.DataFrame(all_results)

    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(reports_dir / "model_results.csv", index=False)

    plot_model_comparison(results_df, figures_dir)

    logger.info("Training complete. Best model exists in reports/best_model.pkl")


if __name__ == "__main__":
    configure_logging()
    main()