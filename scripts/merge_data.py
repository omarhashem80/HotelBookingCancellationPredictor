from src.data.collection import (
    fetch_all_holidays,
    fill_new_df,
    load_and_prepare,
    save_merged_dataset,
)
from src.utils.io import read_config
from src.config.logging import configure_logging
from loguru import logger


def merge_data():
    cfg = read_config()
    input_file = cfg['data']['raw_data_path']
    output_file = cfg['data']['merged_data_path']
    logger.info("Loading raw data from {}", input_file)
    df = load_and_prepare(input_file)
    logger.info("Loaded data: rows={}, cols={}", df.shape[0], df.shape[1])
    holiday_dict = fetch_all_holidays(df)
    logger.info("Fetched holidays for {} country-year pairs", len(holiday_dict))
    df = fill_new_df(df, holiday_dict)
    save_merged_dataset(df, output_file)
    logger.info("Saved merged dataset to {}", output_file)


if __name__ == "__main__":
    configure_logging()
    merge_data()
