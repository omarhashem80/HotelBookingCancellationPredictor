import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_path: str
    mlflow_tracking_uri: str
    random_state: int
    target_column: str = "is_canceled"
    project_root: Path = Path(__file__).resolve().parents[2]


def get_settings() -> Settings:
    """Load settings from environment variables with safe defaults."""
    return Settings(
        data_path=os.getenv("DATA_PATH", "data/raw/hotel_bookings_with_holidays.csv"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )
