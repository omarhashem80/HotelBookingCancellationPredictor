from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_confusion(y_true, y_pred, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    _ensure_dir(output)

    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count",
        },
        x=["Not Cancelled", "Cancelled"],
        y=["Not Cancelled", "Cancelled"],
        title="Confusion Matrix",
    )

    path = output / "confusion_matrix.png"
    fig.write_image(path)

    return path


def plot_roc(y_true, y_prob, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    _ensure_dir(output)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC={roc_auc:.4f})",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random Classifier",
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
    )

    path = output / "roc_curve.png"
    fig.write_image(path)

    return path


def plot_feature_importance(
    feature_names,
    importances,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    _ensure_dir(output)

    feature_names = np.array(feature_names)
    importances = np.array(importances)

    idx = np.argsort(importances)[-20:]

    df = pd.DataFrame({
        "feature": feature_names[idx],
        "importance": importances[idx],
    }).sort_values("importance", ascending=True)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 20 Feature Importances",
    )

    fig.update_layout(template="plotly_white")

    path = output / "feature_importance.png"
    fig.write_image(path)

    return path


def plot_model_comparison(
    results_df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    _ensure_dir(output)

    ordered = results_df.sort_values("f1", ascending=False)

    fig = px.bar(
        ordered,
        x="model",
        y="f1",
        text="f1",
        title="Model Comparison by F1 Score",
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="F1 Score",
        template="plotly_white",
    )

    path = output / "model_comparison_f1.png"
    fig.write_image(path)

    return path