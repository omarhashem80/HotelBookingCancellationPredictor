from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except Exception:  # pragma: no cover - fallback when catboost is unavailable
    CatBoostClassifier = None
    HAS_CATBOOST = False


def get_catboost_estimator(random_state: int = 42):
    if HAS_CATBOOST:
        return CatBoostClassifier(
            random_seed=random_state,
            depth=6,
            learning_rate=0.05,
            iterations=500,
            verbose=False,
        )
    return HistGradientBoostingClassifier(random_state=random_state)


def catboost_param_grid() -> dict:
    if HAS_CATBOOST:
        return {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [3.0, 5.0, 7.0],
            "iterations": [400, 600],
        }
    return {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [None, 8],
    }
