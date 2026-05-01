from sklearn.linear_model import LogisticRegression


def get_logistic_estimator(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=random_state)


def logistic_param_grid() -> dict:
    return {"C": [0.01, 0.1, 1, 10]}
