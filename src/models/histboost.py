from typing import Dict, Any
from sklearn.ensemble import HistGradientBoostingClassifier



HISTBOOST_PARAM_GRID: Dict[str, list] = {
    "max_depth": [3, 6, 10],
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 200],
}


def get_histboost_estimator(
    random_state: int = 42
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=random_state)


def get_histboost_param_grid() -> Dict[str, list]:
    return HISTBOOST_PARAM_GRID.copy()