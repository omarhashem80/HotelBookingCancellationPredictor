import sys
from pathlib import Path
from src.data.cleaning import clean_data
from src.utils.io import load_csv, read_config
from src.visualization.eda_plots import create_eda_plots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_eda_reports():
    cfg = read_config()
    data_path = Path(cfg["data"]["data_path"])
    df = load_csv(data_path)
    cleaned_df = clean_data(df)
    create_eda_plots(cleaned_df)


if __name__ == "__main__":
    generate_eda_reports()
