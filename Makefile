PYTHON := poetry run python

.PHONY: install preprocess train evaluate test lint format eda

install:
	poetry install

preprocess:
	$(PYTHON) -m scripts.preprocess

train:
	$(PYTHON) -m scripts.train

evaluate:
	$(PYTHON) -m scripts.evaluate

test:
	poetry run pytest -q

lint:
	poetry run flake8 src scripts tests

format:
	poetry run black src scripts tests

eda:
	poetry run jupyter notebook notebooks/
