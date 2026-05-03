from pathlib import Path
from loguru import logger

from scripts.validate_data import run_hotel_validation
from src.config.logging import configure_logging
from src.data.cleaning import clean_data
from src.data.ingestion import load_and_merge
from src.features.build_features import build_features
from src.utils.io import read_config, save_csv


def run_preprocess() -> Path:
    cfg = read_config()

    raw_path = Path(cfg["data"]["merged_data_path"])
    processed_path = Path(cfg["data"]["processed_data_path"])
    report_path = cfg["reports"]["validation_path"]

    df = load_and_merge(raw_path)
    logger.info("Loaded raw data: rows={}, cols={}", df.shape[0], df.shape[1])
    run_hotel_validation(df, report_path)
    logger.info("Saved validation report to {}", report_path)
    cleaned = clean_data(df)

    logger.info(
        "Cleaned data: rows={}, cols={}", cleaned.shape[0], cleaned.shape[1]
    )
    featured = build_features(cleaned)
    logger.info(
        "Built features: rows={}, cols={}",
        featured.shape[0],
        featured.shape[1],
    )
    save_csv(featured, processed_path)
    logger.info("Saved processed dataset to {}", processed_path)
    return processed_path


if __name__ == "__main__":
    configure_logging()
    path = run_preprocess()
    logger.info("Saved processed dataset to: {}", path)
