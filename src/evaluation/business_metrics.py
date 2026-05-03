import pandas as pd
from sklearn.metrics import confusion_matrix


def false_negative_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    _, _, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = fn + tp
    return float(fn / denom) if denom else 0.0


def revenue_loss_estimate(
        y_true: pd.Series,
        y_pred: pd.Series,
        adr: pd.Series,
        total_nights: pd.Series,
        deposit_type: pd.Series,
) -> float:
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "adr": adr,
        "nights": total_nights,
        "deposit": deposit_type
    })

    fn = (df["y_true"] == 1) & (df["y_pred"] == 0)

    deposit_penalty = df["deposit"].map({
        "No Deposit": 1.0,
        "Non Refund": 0.0,
        "Refundable": 0.5
    }).fillna(1.0)

    loss = (fn * df["adr"] * df["nights"] * deposit_penalty).sum()

    return float(loss)


def cost_sensitive_score(
    y_true: pd.Series,
    y_pred: pd.Series,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> float:
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = fn_cost * fn + fp_cost * fp
    return float(-total_cost)
