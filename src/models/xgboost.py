from typing import Dict, Any
from xgboost import XGBClassifier

XGBOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    "eval_metric": "logloss",
    "n_jobs": -1,
}

XGBOOST_PARAM_GRID: Dict[str, list] = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}


def get_xgboost_estimator(
    random_state: int = 42
) -> XGBClassifier:
    return XGBClassifier(random_state=random_state, **XGBOOST_DEFAULT_PARAMS)


def get_xgboost_param_grid() -> Dict[str, list]:
    return XGBOOST_PARAM_GRID.copy()