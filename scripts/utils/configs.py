import os
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """実験全体の設定"""

    experiment_name: str
    """実験名を記載 ※'_'は使わず'-'を利用する事"""

    n_splits: int
    """Fold数"""


@dataclass(frozen=True)
class TransformerConfig:
    """Transformerに影響する設定"""

    model_name: str = "microsoft/deberta-v3-large"
    seed: int = 42
    max_length: int = 1024  # to avoid truncating majority of essays.
    lr: float = 1e-5
    train_batch_size: int = 4
    eval_batch_size: int = 8
    train_epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    num_labels: int = 6


def get_transformer_config():
    return TransformerConfig()


def get_experiment_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)

    config = {"experiment_name": yaml_config["model_name"], "n_splits": yaml_config["n_splits"]}

    return ExperimentConfig(**config)
