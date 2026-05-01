from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier


def get_xgb_feature_selector(
    random_state: int = 42,
    threshold: str = "median"
) -> SelectFromModel:
    
    estimator: BaseEstimator = XGBClassifier(
        n_estimators=100,
        random_state=random_state,
        eval_metric="logloss",
    )

    return SelectFromModel(
        estimator=estimator,
        threshold=threshold
    )