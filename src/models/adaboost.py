from sklearn.ensemble import AdaBoostClassifier


def get_ada_boost_estimator(
    random_state: int = 42,
) -> AdaBoostClassifier:
    return AdaBoostClassifier(random_state=random_state,)


def ada_boost_param_grid() -> dict:
    return {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1.0],
    }
