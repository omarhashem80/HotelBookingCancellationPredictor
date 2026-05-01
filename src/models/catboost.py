from catboost import CatBoostClassifier


def get_catboost_estimator(random_state: int = 42):
    return CatBoostClassifier(random_seed=random_state, verbose=0)


def catboost_param_grid() -> dict:
    return {
        'iterations': [200, 400],
        'depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1]
    }