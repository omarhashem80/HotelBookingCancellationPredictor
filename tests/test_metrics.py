from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate
from src.evaluation.business_metrics import (
    cost_sensitive_score,
    false_negative_rate,
    revenue_loss_estimate,
)
from src.evaluation.error_analysis import confusion_breakdown, segment_error_analysis
from src.evaluation.metrics import calculate_classification_metrics


def test_calculate_classification_metrics_without_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics == {
        "accuracy": 0.75,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": 0.8,
    }
    assert "roc_auc" not in metrics


def test_calculate_classification_metrics_with_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.7, 0.9])

    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_calculate_classification_metrics_zero_division_is_zero():
    metrics = calculate_classification_metrics(
        np.array([0, 1, 1]),
        np.array([0, 0, 0]),
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_confusion_breakdown_returns_named_counts():
    result = confusion_breakdown(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 0, 1]),
    )

    assert result == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_confusion_breakdown_handles_single_class_input():
    result = confusion_breakdown(pd.Series([0, 0]), pd.Series([0, 0]))

    assert result == {"tn": 2, "fp": 0, "fn": 0, "tp": 0}


def test_segment_error_analysis_groups_lead_time_and_adr():
    df = pd.DataFrame(
        {
            "lead_time": [10, 60, 120, 250],
            "adr": [50.0, 100.0, 150.0, 200.0],
        }
    )
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 0, 1, 1])

    result = segment_error_analysis(df, y_true, y_pred)

    assert result["lead_time"] == {
        "short": 0.0,
        "medium": 1.0,
        "long": 0.0,
        "very_long": 1.0,
    }
    assert len(result["adr"]) == 4
    assert set(result["adr"].values()) == {0.0, 1.0}


def test_segment_error_analysis_returns_empty_dict_without_supported_columns():
    result = segment_error_analysis(
        pd.DataFrame({"hotel": ["city", "resort"]}),
        pd.Series([0, 1]),
        pd.Series([0, 0]),
    )

    assert result == {}


def test_false_negative_rate_and_no_positive_edge_case():
    assert false_negative_rate(pd.Series([1, 1, 0]), pd.Series([0, 1, 0])) == 0.5
    assert false_negative_rate(pd.Series([0, 0]), pd.Series([0, 1])) == 0.0


def test_cost_sensitive_score_uses_custom_costs():
    result = cost_sensitive_score(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 0, 1]),
        fn_cost=10.0,
        fp_cost=2.0,
    )

    assert result == -12.0


def test_revenue_loss_estimate_applies_deposit_penalties():
    result = revenue_loss_estimate(
        y_true=pd.Series([1, 1, 1, 0]),
        y_pred=pd.Series([0, 0, 0, 0]),
        adr=pd.Series([100.0, 100.0, 100.0, 100.0]),
        total_nights=pd.Series([2, 2, 2, 2]),
        deposit_type=pd.Series(["No Deposit", "Refundable", "Non Refund", "Unknown"]),
    )

    assert result == 300.0


def test_json_safe_converts_nested_keys_and_sequences():
    interval = pd.Interval(0, 1)
    result = evaluate._json_safe({interval: {"items": [(1, interval)]}})

    assert result == {"(0, 1]": {"items": [[1, "(0, 1]"]]}}


@dataclass(frozen=True)
class DummySettings:
    project_root: object
    random_state: int = 42
    target_column: str = "is_canceled"


class DummyModel:
    def predict(self, X):
        return np.resize(np.array([0, 1]), len(X))

    def predict_proba(self, X):
        pred = self.predict(X)
        return np.column_stack([1 - pred, pred])


def test_evaluate_main_writes_report_with_business_and_error_sections(tmp_path, monkeypatch):
    data_path = tmp_path / "data" / "processed" / "hotel_bookings_processed.csv"
    model_path = tmp_path / "reports" / "best_model.pkl"
    data_path.parent.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder", encoding="utf-8")
    model_path.write_text("placeholder", encoding="utf-8")

    df = pd.DataFrame(
        {
            "is_canceled": np.resize(np.array([0, 1]), 20),
            "lead_time": np.arange(20) + 1,
            "adr": np.linspace(75.0, 125.0, 20),
            "total_nights": np.resize(np.array([1, 2]), 20),
            "deposit_type": np.resize(np.array(["No Deposit", "Refundable"]), 20),
            "feature": np.arange(20),
        }
    )

    monkeypatch.setattr(evaluate, "get_settings", lambda: DummySettings(project_root=tmp_path))
    monkeypatch.setattr(evaluate, "load_csv", lambda path: df)
    monkeypatch.setattr(evaluate, "load_model", lambda path: DummyModel())
    monkeypatch.setattr(evaluate, "plot_confusion", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "plot_roc", lambda *args, **kwargs: None)

    evaluate.main()

    report = json.loads((tmp_path / "reports" / "evaluation_report.json").read_text())
    assert {"standard_metrics", "business_metrics", "classification_report", "error_analysis"} <= set(
        report
    )
    assert "roc_auc" in report["standard_metrics"]
    assert "revenue_loss_estimate" in report["business_metrics"]
    assert "confusion_breakdown" in report["error_analysis"]


def test_evaluate_main_raises_when_required_artifacts_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "get_settings", lambda: DummySettings(project_root=tmp_path))

    with pytest.raises(FileNotFoundError, match="Processed data or trained model not found"):
        evaluate.main()
