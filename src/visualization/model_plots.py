from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def plot_confusion(y_true, y_pred, output_dir: str | Path) -> Path:
	output = Path(output_dir)
	_ensure_dir(output)
	cm = confusion_matrix(y_true, y_pred)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot(cmap="Blues")
	path = output / "confusion_matrix.png"
	plt.tight_layout()
	plt.savefig(path)
	plt.close()
	return path


def plot_roc(y_true, y_prob, output_dir: str | Path) -> Path:
	output = Path(output_dir)
	_ensure_dir(output)
	RocCurveDisplay.from_predictions(y_true, y_prob)
	path = output / "roc_curve.png"
	plt.tight_layout()
	plt.savefig(path)
	plt.close()
	return path


def plot_feature_importance(feature_names, importances, output_dir: str | Path) -> Path:
	output = Path(output_dir)
	_ensure_dir(output)
	feature_names = np.array(feature_names)
	importances = np.array(importances)
	idx = np.argsort(importances)[-20:]

	plt.figure(figsize=(10, 6))
	plt.barh(feature_names[idx], importances[idx])
	plt.title("Top Feature Importances")
	path = output / "feature_importance.png"
	plt.tight_layout()
	plt.savefig(path)
	plt.close()
	return path


def plot_model_comparison(results_df, output_dir: str | Path) -> Path:
	output = Path(output_dir)
	_ensure_dir(output)

	ordered = results_df.sort_values("f1", ascending=False)
	plt.figure(figsize=(10, 5))
	plt.bar(ordered["model"], ordered["f1"])
	plt.title("Model Comparison by F1")
	plt.ylabel("F1 Score")
	plt.xticks(rotation=30, ha="right")
	path = output / "model_comparison_f1.png"
	plt.tight_layout()
	plt.savefig(path)
	plt.close()
	return path

