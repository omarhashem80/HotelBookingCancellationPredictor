from imblearn.over_sampling import SMOTE


def get_smote_sampler(random_state: int = 42) -> SMOTE:
    return SMOTE(random_state=random_state)
