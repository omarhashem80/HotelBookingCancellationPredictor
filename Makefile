PYTHON := poetry run python

.PHONY: install preprocess merge_data validate train evaluate test lint format eda

install:
	poetry install

merge_data: data/raw/hotel_bookings.csv
	${PYTHON} scripts/merge_data.py
validate: data/raw/hotel_bookings_with_holidays.csv
	${PYTHON} scripts/validate_data.py
preprocess:
	$(PYTHON) -m scripts.preprocess

train:
	$(PYTHON) -m scripts.train

evaluate:
	$(PYTHON) -m scripts.evaluate

test:
	poetry run pytest

lint:
	poetry run flake8 src scripts tests

format:
	poetry run black src scripts tests

eda: data/raw/hotel_bookings_with_holidays.csv
	${PYTHON} scripts/generate_eda_reports.py
