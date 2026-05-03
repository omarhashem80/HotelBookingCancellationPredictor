from typing import Dict, Any
from sklearn.linear_model import LogisticRegression

LOGISTIC_DEFAULT_PARAMS: Dict[str, Any] = {
    "max_iter": 2000,
}

LOGISTIC_PARAM_GRID: Dict[str, list] = {
    "C": [0.01, 0.1, 1, 10],
}


def get_logistic_estimator(
    random_state: int = 42
) -> LogisticRegression:
    return LogisticRegression(random_state=random_state, **LOGISTIC_DEFAULT_PARAMS)


def get_logistic_param_grid() -> Dict[str, list]:
    return LOGISTIC_PARAM_GRID.copy()