from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - fallback when xgboost is unavailable
    XGBClassifier = None
    HAS_XGBOOST = False


def get_xgboost_estimator(random_state: int = 42, scale_pos_weight: float = 1.0):
    if HAS_XGBOOST:
        return XGBClassifier(
            random_state=random_state,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
    return GradientBoostingClassifier(random_state=random_state)


def xgboost_param_grid() -> dict:
    if HAS_XGBOOST:
        return {
            "model__n_estimators": [300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [4, 6, 8],
            "model__subsample": [0.8, 1.0],
        }
    return {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    }
