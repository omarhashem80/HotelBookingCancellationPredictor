# Hotel Booking Cancellation Prediction

Production-style supervised ML pipeline for predicting hotel booking cancellations, with modular architecture, experiment tracking, and CI-ready tests.

## Project Overview

Business problem: hotels lose revenue when likely cancellations are not identified early enough. This project predicts `is_canceled` from booking attributes and holiday-related context.

Primary dataset:

- `data/raw/hotel_bookings_with_holidays.csv`

Secondary source support:

- `src/data/ingestion.py` supports optional second dataset merge by key(s).

## Quick Start

### 1. Install

```bash
poetry install
```

### 2. Configure

`.env` is already included with defaults:

```env
DATA_PATH=data/raw/hotel_bookings_with_holidays.csv
MLFLOW_TRACKING_URI=file:./mlruns
RANDOM_STATE=42
```

### 3. Run Pipeline

```bash
make preprocess
make train
make evaluate
```

Fast smoke training:

```bash
poetry run python scripts/train.py --models baseline,logistic --fast
```

### 4. Run Tests

```bash
make test
```

### 5. MLflow UI

```bash
poetry run mlflow ui --backend-store-uri ./mlruns
```

Open http://127.0.0.1:5000 in your browser.

## Makefile Commands

- `make install`: install dependencies
- `make preprocess`: run ingestion + validation + cleaning + feature engineering
- `make train`: train and compare models, log to MLflow, save best model
- `make evaluate`: evaluate saved best model and generate report/plots
- `make test`: run pytest suite
- `make lint`: run flake8 checks
- `make format`: run black formatter
- `make eda`: open notebooks folder in Jupyter

## Folder Structure

```text
project/
├── data/
├── notebooks/
├── scripts/
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── visualization/
│   ├── tracking/
│   └── utils/
├── tests/
├── reports/
├── mlruns/
└── .github/workflows/
```

## Modeling Coverage

Implemented models:

- Baseline (`DummyClassifier`)
- Logistic Regression
- Random Forest
- XGBoost
- CatBoost (with native categorical support + early stopping)

## Expected Outputs

After `make train`:

- `reports/best_model.pkl`
- `reports/best_model_metrics.json`
- `reports/model_results.csv`
- MLflow runs under `mlruns/`

After `make evaluate`:

- `reports/evaluation_report.json`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png` (if probabilities available)

## CI/CD Pipeline

Continuous Integration and automated testing is handled natively via [GitHub Actions](.github/workflows/ci.yml).

## Team Contributions

Add your names and contribution breakdown here.
