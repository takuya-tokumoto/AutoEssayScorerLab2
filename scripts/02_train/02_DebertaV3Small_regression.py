#!/usr/bin/env python
# coding: utf-8

import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]

sys.path.append(str(repo_dir / "scripts/"))
from utils.configs import get_experiment_config, get_transformer_config
from utils.logging import set_logger
from utils.path import PathManager
from utils.transformer import EssayScorer

warnings.simplefilter("ignore")

ex_config = get_experiment_config()
tr_config = get_transformer_config()

# パスの設定
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
mode = ex_config.experiment_name
path_to = PathManager(s3_dir, mode)


# ロギングの設定
logger = set_logger(__name__)


def seed_everything(seed):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# seed_everything(seed=CFG.seed)


def load_datasets(input_data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """所定のパスより学習/検証データを読み込む

    Args:
        input_data_dir (Path): ファイルの配置先パス

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: 学習データ、検証データ、検証データの目的変数
    """

    def convert_columns(data: pd.DataFrame) -> pd.DataFrame:
        """インプットデータの加工

        Args:
            data (pd.DataFrame): 入力データ

        Returns:
            pd.DataFrame: 加工済データ
        """

        # `label = score - 1`(labelは 0-5 の範囲に変更)
        data["label"] = data["score"].apply(lambda x: x - 1)

        # label を回帰では `float32` に変換、分類の場合は `int32` に変換
        data["label"] = data["label"].astype("float32")

        return data

    # データの読み込み
    # full_textが残っている02_train_data_splitter.py出力マートを持ってくる
    train_path: Path = input_data_dir / f"train_fold.csv"
    _train_data = pd.read_csv(train_path)
    valid_path: Path = input_data_dir / f"valid_fold.csv"
    _valid_data = pd.read_csv(valid_path)

    # データ加工
    train_data = convert_columns(_train_data)
    valid_data = convert_columns(_valid_data)

    # 評価用にvalidデータの目的変数を用意
    valid_y_int = _valid_data["score"].astype(int).values

    return train_data, valid_data, valid_y_int


def compute_metrics_for_regression(eval_pred):
    """QWK評価関数（回帰タスク）

    Args:
        eval_pred (_type_): _description_

    Returns:
        _type_: _description_
    """

    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.clip(0, 5).round(0), weights="quadratic")
    results = {"qwk": qwk}
    return results


def cross_validate(n_splits: int):
    """クロスバリデーションの実施

    Args:
        n_splits (int): バリデーション数
    """
    f1_scores = []
    kappa_scores = []
    predictions = []
    actual_labels = []

    model_name = tr_config.model_name
    max_length = tr_config.max_length
    essay_scorer = EssayScorer(model_name, max_length)

    for i in range(n_splits):

        # データの読み込み
        base_fold_dir: Path = path_to.skf_mart_dir / f"fold_{i}/"
        train_data, valid_data, valid_y_int = load_datasets(base_fold_dir)

        # 正解データを格納
        actual_labels.extend(valid_y_int)

        # ディレクトリの準備
        output_path: Path = path_to.deberta_v3_small_finetuned_dir / f"fold_{i}/"
        output_path.mkdir(parents=True, exist_ok=True)

        # 学習と評価の実施
        predictions_fold, f1_fold, kappa_fold = essay_scorer.train_and_evaluate(
            train_data, valid_data, valid_y_int, output_path
        )

        predictions.extend(predictions_fold)
        f1_scores.append(f1_fold)
        kappa_scores.append(kappa_fold)

        logger.info(f"F1 score for fold {i}: {f1_fold}")
        logger.info(f"Cohen kappa score for fold {i}: {kappa_fold}")

    # OOFでの評価結果を算出
    oof_f1_score = f1_score(actual_labels, predictions, average="weighted")
    oof_kappa_score = cohen_kappa_score(actual_labels, predictions, weights="quadratic")
    logger.info(f"Out-Of-Fold F1 score: {oof_f1_score}")
    logger.info(f"Out-Of-Fold Cohen kappa score: {oof_kappa_score}")


if __name__ == "__main__":

    logger.info(f"【条件： {mode}】Deverta-V3 実行開始")

    n_splits = ex_config.n_splits
    cross_validate(n_splits)

    ## s3にファイル移動

    logger.info(f"実行完了")
