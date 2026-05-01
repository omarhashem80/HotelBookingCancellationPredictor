from xgboost import XGBClassifier


def get_xgboost_estimator(random_state: int = 42):
    return XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)


def xgboost_param_grid() -> dict:
    return {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]}
