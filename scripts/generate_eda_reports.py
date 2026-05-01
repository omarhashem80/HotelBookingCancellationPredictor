import sys
from pathlib import Path
from src.data.cleaning import clean_data
from src.utils.io import load_csv, read_config
from src.visualization.eda_plots import create_eda_plots
from loguru import logger
from src.config.logging import configure_logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_eda_reports():
    cfg = read_config()
    data_path = Path(cfg["data"]["merged_data_path"])
    logger.info("Loading merged data from {}", data_path)
    df = load_csv(data_path)
    logger.info("Loaded data: rows={}, cols={}", df.shape[0], df.shape[1])
    cleaned_df = clean_data(df)
    logger.info("Cleaned data: rows={}, cols={}", cleaned_df.shape[0], cleaned_df.shape[1])
    create_eda_plots(cleaned_df)
    logger.info("Generated EDA reports")


if __name__ == "__main__":
    configure_logging()
    generate_eda_reports()
