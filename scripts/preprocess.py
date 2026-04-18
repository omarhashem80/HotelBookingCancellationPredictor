from __future__ import annotations

import json
from pathlib import Path

from src.config.logging import configure_logging
from src.config.settings import get_settings
from src.data.cleaning import clean_data
from src.data.ingestion import load_and_merge
from src.data.validation import validate_dataframe
from src.features.build_features import build_features
from src.utils.io import save_csv


def run_preprocess() -> Path:
	settings = get_settings()
	root = settings.project_root

	raw_path = root / settings.data_path
	processed_path = root / "data/processed/hotel_bookings_processed.csv"
	report_path = root / "reports/validation_report.json"

	df = load_and_merge(raw_path)
	validation_report = validate_dataframe(df, target_col=settings.target_column)

	report_path.parent.mkdir(parents=True, exist_ok=True)
	report_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

	cleaned = clean_data(df)
	featured = build_features(cleaned)
	save_csv(featured, processed_path)
	return processed_path


if __name__ == "__main__":
	configure_logging()
	path = run_preprocess()
	print(f"Saved processed dataset to: {path}")

