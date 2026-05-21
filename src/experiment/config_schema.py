from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    experiment_id: str
    stage: int
    tier: int
    description: str
    missing_policy: str = "M1"
    outlier_policy: str = "O0"
    feature_policy: str = "F0"
    scale_policy: str = "S0"
    sampling_policy: str = "B0"
    model_name: str = "lightgbm"
    model_params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        parts = [
            self.missing_policy,
            self.outlier_policy,
            self.feature_policy,
            self.scale_policy,
            self.sampling_policy,
            self.model_name,
            str(sorted(self.model_params.items())),
        ]
        return "|".join(parts)
