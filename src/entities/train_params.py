from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    class_weight: str = field(default='balanced')
    random_state: int = field(default=999)
    n_estimators: int = field(default=459)
    max_depth: int = field(default=10)
    min_samples_split: int = field(default=2)
    min_samples_leaf: int = field(default=2)
    n_jobs: int = field(default=-1)
