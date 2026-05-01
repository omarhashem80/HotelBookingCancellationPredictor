from sklearn.dummy import DummyClassifier


def get_baseline_estimator() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")
