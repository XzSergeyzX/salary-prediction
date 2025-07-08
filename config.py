# project/config.py
from dataclasses import dataclass

@dataclass
class Config:
    data_path = "../salary_data.csv"
    target_column: str = "salary_in_usd"
    model_type: str = "logreg"
    random_state: int = 42
    test_size: float = 0.2
