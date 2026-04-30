from src.data.collection import (
    fetch_all_holidays,
    fill_new_df,
    load_and_prepare,
    save_merged_dataset,
)
from src.utils.io import read_config


def merge_data():
    cfg = read_config()
    input_file = cfg['data']['raw_data_path']
    output_file = cfg['data']['merged_data_path']
    df = load_and_prepare(input_file)
    holiday_dict = fetch_all_holidays(df)
    df = fill_new_df(df, holiday_dict)
    save_merged_dataset(df, output_file)


if __name__ == "__main__":
    merge_data()
