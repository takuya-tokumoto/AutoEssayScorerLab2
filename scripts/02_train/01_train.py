#!/usr/bin/env python
# coding: utf-8

import os

## config.yamlの読み込み
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "..", "config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

import logging

## Import
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score

# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.model import Trainer
from utils.path import PathManager
from utils.qwk import quadratic_weighted_kappa, qwk_obj

## パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)

## ディレクトリの準備
path_to.models_weight_dir.mkdir(parents=True, exist_ok=True)

## ロギングの設定
# JSTタイムゾーンを定義
JST = timezone(timedelta(hours=+9), "JST")


# ロギングフォーマッタの拡張クラス
class JSTFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=JST)
        return dt.timetuple()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=JST)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


# loggerの取得
logger = logging.getLogger()
# 既存のハンドラをクリア
if logger.hasHandlers():
    logger.handlers.clear()
# 新しいハンドラを設定
handler = logging.FileHandler("training.log")
handler.setLevel(logging.INFO)
formatter = JSTFormatter("%(asctime)s:%(levelname)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# モデルパラメータ
model_params = {
    "lgbm": {
        "objective": qwk_obj,  # `utils/model.py`定義されている関数を指定
        "metrics": "None",
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 10,
        "colsample_bytree": 0.3,
        "reg_alpha": 2,
        "reg_lambda": 0.1,
        "n_estimators": 700,
        "random_state": 42,
        "extra_trees": True,
        "class_weight": "balanced",
        "device": "gpu" if torch.cuda.is_available() else "cpu",
        "verbosity": -1,
    },
    "xgb": {
        "objective": qwk_obj,  # `utils/model.py`定義されている関数を指定
        "metrics": "None",
        "learning_rate": 0.1,
        "max_depth": 5,
        "num_leaves": 10,
        "colsample_bytree": 0.5,
        "reg_alpha": 1.0,
        "reg_lambda": 0.1,
        "n_estimators": 1024,
        "random_state": 42,
        "extra_trees": True,
        "class_weight": "balanced",
        "tree_method": "hist",
        "device": "gpu" if torch.cuda.is_available() else "cpu",
    },
}


def load_data(path):
    """データ読み込み"""

    return pd.read_csv(path)


def prepare_data(input_data, feature_select):
    """データを指定の変数で絞りこんだうえで学習データ(X, y, y_int)を作成"""

    X = input_data[feature_select].astype(np.float32).values
    y = input_data[config["target"]].astype(np.float32).values - config["avg_train_score"]
    y_int = input_data[config["target"]].astype(int).values

    return X, y, y_int


def cross_validate(config):
    f1_scores = []
    kappa_scores = []
    predictions = []
    actual_labels = []

    for i in range(config["n_splits"]):

        ## データの読み込み
        train_path: Path = path_to.add_meta_mart_dir / f"fold_{i}/train_fold_add_meta.csv"
        train_data = load_data(train_path)
        valid_path: Path = path_to.add_meta_mart_dir / f"fold_{i}/valid_fold_add_meta.csv"
        valid_data = load_data(valid_path)

        # ディレクトリの準備
        model_fold_path: Path = path_to.models_weight_dir / f"fold_{i}/"
        model_fold_path.mkdir(parents=True, exist_ok=True)

        ## 特徴量の絞り込み計算 -> 変数重要度上位13,000件をピックアップ
        ## データ準備
        feature_all = list(filter(lambda x: x not in ["essay_id", "score"], train_data.columns))
        train_X, train_y, train_y_int = prepare_data(train_data, feature_all)
        valid_X, valid_y, valid_y_int = prepare_data(valid_data, feature_all)
        ## 全特徴量含めて学習
        trainer_all = Trainer(config, model_params)
        trainer_all.initialize_models()
        trainer_all.train(train_X, train_y)
        ## 変数重要度を取得
        fse = pd.Series(trainer_all.light.feature_importances_, feature_all)
        feature_select = fse.sort_values(ascending=False).index.tolist()[:13000]
        ## feature_select リストを pickle ファイルとして保存
        with open(model_fold_path / "feature_select.pickle", "wb") as f:
            pickle.dump(feature_select, f)

        ### 特徴量を絞り込んだうえでモデル学習
        ## データ準備
        train_X, train_y, train_y_int = prepare_data(train_data, feature_select)
        valid_X, valid_y, valid_y_int = prepare_data(valid_data, feature_select)
        # 正解データを格納
        actual_labels.extend(valid_y_int)

        ## 学習
        trainer = Trainer(config, model_params)
        trainer.initialize_models()
        # イテレーション回数の最適化
        trainer.train_with_early_stopping(train_X, train_y)
        logger.info(f"For fold {i}, the optimal number of iterations for LightGBM is {trainer.best_light_iteration}.")
        logger.info(f"For fold {i}, the optimal number of iterations for XGBoost is {trainer.best_xgb_iteration}.")
        # 本番
        trainer.train(train_X, train_y)

        ## 学習結果を保存
        trainer.save_weight(model_fold_path)

        ## 学習結果を評価
        predictions_fold = trainer.predict(valid_X)
        predictions_fold = predictions_fold + config["avg_train_score"]
        predictions_fold = np.clip(predictions_fold, 1, 6).round()
        # 予測結果を格納
        predictions.extend(predictions_fold)

        # F1スコア
        f1_fold = f1_score(valid_y_int, predictions_fold, average="weighted")
        f1_scores.append(f1_fold)
        # Cohen kappa score
        kappa_fold = cohen_kappa_score(valid_y_int, predictions_fold, weights="quadratic")
        kappa_scores.append(kappa_fold)

        logger.info(f"F1 score for fold {i}: {f1_fold}")
        logger.info(f"Cohen kappa score for fold {i}: {kappa_fold}")

    ## 評価結果
    # 各foldの平均評価を算出
    mean_f1_score = np.mean(f1_scores)
    mean_kappa_score = np.mean(kappa_scores)
    logger.info(f"Mean F1 score across {config['n_splits']} folds: {mean_f1_score}")
    logger.info(f"Mean Cohen kappa score across {config['n_splits']} folds: {mean_kappa_score}")
    # OOFでの評価結果を算出
    oof_f1_score = f1_score(actual_labels, predictions, average="weighted")
    oof_kappa_score = cohen_kappa_score(actual_labels, predictions, weights="quadratic")
    logger.info(f"Out-Of-Fold F1 score: {oof_f1_score}")
    logger.info(f"Out-Of-Fold Cohen kappa score: {oof_kappa_score}")


if __name__ == "__main__":

    logger.info(f"【条件： {mode}】実行開始")
    cross_validate(config)

    logger.info(f"実行完了")
