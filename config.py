# project/config.py
from dataclasses import dataclass

@dataclass
class Config:
    data_path = "../data/train.csv"
    target_column: str = "salary"
    model_type: str = "logreg"
    random_state: int = 42
    test_size: float = 0.2