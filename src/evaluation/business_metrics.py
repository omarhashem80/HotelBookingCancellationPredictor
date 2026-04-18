from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def false_negative_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	denom = fn + tp
	return float(fn / denom) if denom else 0.0


def revenue_loss_estimate(
	y_true: pd.Series,
	y_pred: pd.Series,
	adr: pd.Series | np.ndarray,
	total_nights: pd.Series | np.ndarray,
) -> float:
	"""Estimate loss from missed cancellations (false negatives)."""
	y_true = pd.Series(y_true).reset_index(drop=True)
	y_pred = pd.Series(y_pred).reset_index(drop=True)
	adr = pd.Series(adr).reset_index(drop=True)
	total_nights = pd.Series(total_nights).reset_index(drop=True)

	fn_mask = (y_true == 1) & (y_pred == 0)
	loss = (adr[fn_mask] * total_nights[fn_mask]).sum()
	return float(loss)


def cost_sensitive_score(
	y_true: pd.Series,
	y_pred: pd.Series,
	fn_cost: float = 5.0,
	fp_cost: float = 1.0,
) -> float:
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	total_cost = fn_cost * fn + fp_cost * fp
	return float(-total_cost)

