from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def get_sampler(sampling_policy: str, random_state: int = 42):
    if sampling_policy == "B0":
        return None
    if sampling_policy == "B3":
        return RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
    if sampling_policy.startswith("B4"):
        k = 5
        if "_" in sampling_policy:
            k = int(sampling_policy.split("_")[1])
        return SMOTE(random_state=random_state, k_neighbors=k)
    if sampling_policy == "B5":
        return BorderlineSMOTE(random_state=random_state, k_neighbors=5)
    if sampling_policy == "B6":
        return SMOTETomek(random_state=random_state)
    if sampling_policy == "B7":
        return ADASYN(random_state=random_state, n_neighbors=5)
    if sampling_policy in {"B1", "B2", "B8"}:
        return None
    raise ValueError(f"Unknown sampling policy: {sampling_policy}")
