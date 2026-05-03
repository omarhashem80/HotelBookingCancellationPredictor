from typing import Dict, Any
from catboost import CatBoostClassifier

CATBOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    "verbose": 0,
}

CATBOOST_PARAM_GRID: Dict[str, list] = {
    "iterations": [200, 400],
    "depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}


def get_catboost_estimator(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(random_seed=random_state, **CATBOOST_DEFAULT_PARAMS)


def get_catboost_param_grid() -> Dict[str, list]:
    return CATBOOST_PARAM_GRID.copy()