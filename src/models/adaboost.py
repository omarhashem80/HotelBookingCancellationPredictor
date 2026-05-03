from typing import Dict, Any
from sklearn.ensemble import AdaBoostClassifier


ADABOOST_PARAM_GRID: Dict[str, list] = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.1, 0.5, 1.0],
}


def get_ada_boost_estimator(
    random_state: int = 42,
) -> AdaBoostClassifier:
    return AdaBoostClassifier(random_state=random_state)


def get_ada_boost_param_grid() -> Dict[str, list]:
    return ADABOOST_PARAM_GRID.copy()