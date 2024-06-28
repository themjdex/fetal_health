import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.25)
    random_state: int = field(default=999)
    shuffle: bool = field(default=True)
