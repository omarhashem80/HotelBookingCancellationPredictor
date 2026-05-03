# Hotel Booking Cancellation Prediction

Production-style supervised ML pipeline for predicting hotel booking cancellations, with modular architecture, experiment tracking, and CI-ready tests.

## Project Overview

Business problem: hotels lose revenue when likely cancellations are not identified early enough. This project predicts `is_canceled` from booking attributes and holiday-related context.

The pipeline covers ingestion, validation, cleaning, feature engineering, multi-model training with hyperparameter search, MLflow experiment tracking, business metric evaluation, and automated deployment to HuggingFace Spaces via GitHub Actions.

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

Train a specific model or a comma-separated list:

```bash
poetry run python -m scripts.train --models baseline,logistic,xgboost,catboost,histboost,adaboost
```

Optional flags:

```bash
--use-smote       # Apply SMOTE oversampling
--use-selector    # Apply XGBoost-based feature selection
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

### 6. Streamlit App

```bash
make streamlit
```

## Makefile Commands

- `make install`: install dependencies
- `make preprocess`: run ingestion + validation + cleaning + feature engineering
- `make train`: train model(s) and log to MLflow, save best model
- `make evaluate`: evaluate saved best model and generate report/plots
- `make test`: run pytest suite
- `make lint`: run flake8 checks
- `make format`: run black formatter
- `make eda`: generate EDA reports
- `make streamlit`: launch the Streamlit dashboard

## Folder Structure

```text
project/
├── data/
│   ├── raw/
│   └── processed/
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
│   └── figures/
├── mlruns/
├── Hotel-Booking/
└── .github/workflows/
```

## Modeling Coverage

Six models are implemented and tracked via MLflow:

- Baseline (`DummyClassifier`)
- Logistic Regression
- XGBoost
- CatBoost
- HistGradientBoosting
- AdaBoost

All models are trained inside a scikit-learn `Pipeline` with a shared preprocessing step (imputation, scaling, one-hot encoding). Hyperparameters are tuned via `GridSearchCV` with stratified 5-fold cross-validation optimising F1.

Current best model: **CatBoost** — F1: 0.9970, Accuracy: 99.83%

## Evaluation Metrics

Each trained run logs:

**Standard:** accuracy, precision, recall, F1

**Business:** false negative rate, cost-sensitive score (FN penalised 5× over FP), revenue loss estimate (ADR × nights × deposit type)

## Expected Outputs

After `make train`:

- Trained best model artifact
- Best model metrics (JSON)
- Model comparison results (CSV)
- MLflow runs under `mlruns/`
- Model comparison chart

After `make evaluate`:

- Evaluation report (JSON) with standard metrics, business metrics, classification report, and error analysis
- Confusion matrix plot
- ROC curve plot (when probabilities are available)

## CI/CD Pipeline

GitHub Actions triggers on every push to `main`/`master` and runs the full pipeline: install → test → preprocess → train → evaluate → quality gate (accuracy ≥ 98%) → export → deploy to HuggingFace Spaces.

The trained model is tracked with Git LFS and deployed automatically on passing all checks.

Live demo: https://huggingface.co/spaces/aelsayed1/Hotel-Booking
