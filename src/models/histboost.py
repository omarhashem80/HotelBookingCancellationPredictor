from sklearn.ensemble import HistGradientBoostingClassifier


def get_histboost_estimator(random_state: int = 42):
    return HistGradientBoostingClassifier(random_state=random_state)

def histboost_param_grid() -> dict:
    return {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.05, 0.1],
        'max_iter': [100, 200]
    }