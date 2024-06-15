#!/usr/bin/env python
# coding: utf-8

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from tokenizers import AddedToken
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

warnings.simplefilter("ignore")
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.logging import set_logger
from utils.path import PathManager

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "..", "config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)

# パラメータ設定
# True USES REGRESSION, False USES CLASSIFICATION
USE_REGRESSION = True
# VERSION NUMBER FOR NAMING OF SAVED MODELS
VER = 1
# IF "LOAD_FROM" IS None, THEN WE TRAIN NEW MODELS
# LOAD_FROM = "/kaggle/input/deberta-v3-small-finetuned-v1/"
# WHEN TRAINING NEW MODELS SET COMPUTE_CV = True
# WHEN LOADING MODELS, WE CAN CHOOSE True or False
COMPUTE_CV = True
MODEL_NAME = "microsoft/deberta-v3-small"

# ロギングの設定
logger = set_logger(__name__)


class CFG:
    # n_splits = 5
    seed = 42
    max_length = 1024  # to avoid truncating majority of essays.
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6


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


def convert_columns(data: pd.DataFrame):
    """インプットデータ加工"""

    ## `label = score - 1`(labelは 0-5 の範囲に変更)
    data["label"] = data["score"].apply(lambda x: x - 1)

    ## label を回帰では `float32` に変換、分類の場合は `int32` に変換
    if USE_REGRESSION:
        data["label"] = data["label"].astype("float32")
    else:
        data["label"] = data["label"].astype("int32")

    return data


def load_model_dataset(input_data_dir: Path) -> pd.DataFrame:
    """インプットデータの用意"""

    ## データの読み込み
    # full_textが残っている02_train_data_splitter.py出力マートを持ってくる
    train_path: Path = input_data_dir / f"train_fold.csv"
    _train_data = pd.read_csv(train_path)
    valid_path: Path = input_data_dir / f"valid_fold.csv"
    _valid_data = pd.read_csv(valid_path)

    ## データ加工
    train_data = convert_columns(_train_data)
    valid_data = convert_columns(_valid_data)

    ## 評価用にvalidデータの目的変数を用意
    valid_y_int = _valid_data["score"].astype(int).values

    return train_data, valid_data, valid_y_int


class Tokenize(object):
    """train, valid, (test)データに対してトークン化処理"""

    def __init__(self, train, valid, tokenizer):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    def get_dataset(self, df):
        ds = Dataset.from_dict(
            {
                "essay_id": [e for e in df["essay_id"]],
                "full_text": [ft for ft in df["full_text"]],
                "label": [s for s in df["label"]],
            }
        )
        return ds

    def tokenize_function(self, example):
        tokenized_inputs = self.tokenizer(example["full_text"], truncation=True, max_length=CFG.max_length)
        return tokenized_inputs

    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)

        tokenized_train = train_ds.map(self.tokenize_function, batched=True)
        tokenized_valid = valid_ds.map(self.tokenize_function, batched=True)

        return tokenized_train, tokenized_valid, self.tokenizer


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


def compute_metrics_for_classification(eval_pred):
    """QWK評価関数（分類タスク）"""

    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    results = {"qwk": qwk}
    return results


def cross_validate(config):
    f1_scores = []
    kappa_scores = []
    predictions = []
    actual_labels = []

    for i in range(config["n_splits"]):

        ## データの読み込み
        base_fold_dir: Path = path_to.skf_mart_dir / f"fold_{i}/"
        train_data, valid_data, valid_y_int = load_model_dataset(base_fold_dir)
        # 正解データを格納
        actual_labels.extend(valid_y_int)

        ## ディレクトリの準備
        model_fold_path: Path = path_to.deberta_v3_small_finetuned_dir / f"fold_{i}/"
        model_fold_path.mkdir(parents=True, exist_ok=True)

        ## 保存先の中身確認
        # 保存先にファイルが存在 : False -> 学習を実施
        # 保存先のファイルが存在しない : True -> 学習を実施しない
        LOAD_FROM = False if len(os.listdir(model_fold_path)) == 0 else True

        ## training_args
        training_args = TrainingArguments(
            output_dir=model_fold_path,
            fp16=True,
            learning_rate=CFG.lr,
            per_device_train_batch_size=CFG.train_batch_size,
            per_device_eval_batch_size=CFG.eval_batch_size,
            num_train_epochs=CFG.train_epochs,
            weight_decay=CFG.weight_decay,
            evaluation_strategy="no",  # 評価を無効にする 'epoch',
            metric_for_best_model="qwk",
            save_strategy="no",  # 評価を無効にする 'epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
            warmup_ratio=CFG.warmup_ratio,
            lr_scheduler_type="linear",  # "cosine" or "linear" or "constant"
            optim="adamw_torch",
            logging_first_step=True,
        )

        ## Tokenizer
        # 新しいトークン追加
        # ("\n") : 新しい段落のため,  (" "*2) : ダブルスペースのため
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])
        # トークン化
        tokenize = Tokenize(train_data, valid_data, tokenizer)
        tokenized_train, tokenized_valid, _ = tokenize()

        ## Config
        config = AutoConfig.from_pretrained(MODEL_NAME)
        if USE_REGRESSION:
            ## 回帰タスクの場合ドロップアウトを除外
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
            config.num_labels = 1
        else:
            config.num_labels = CFG.num_labels

        if LOAD_FROM:
            model = AutoModelForSequenceClassification.from_pretrained(model_fold_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
            model.resize_token_embeddings(len(tokenizer))

        ## 学習
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if USE_REGRESSION:
            compute_metrics = compute_metrics_for_regression
        else:
            compute_metrics = compute_metrics_for_classification
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            # eval_dataset=tokenized_valid,
            eval_dataset=None,
            data_collator=data_collator,
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics
        )
        if LOAD_FROM is False:
            logger.info(f"学習を実行")
            trainer.train()

        ## 学習結果を保存
        if LOAD_FROM is False:
            logger.info(f"学習結果を保存")
            trainer.save_model(model_fold_path)
            tokenizer.save_pretrained(model_fold_path)

        ## 学習結果を評価
        _predictions_fold = trainer.predict(tokenized_valid).predictions
        if USE_REGRESSION:
            predictions_fold = _predictions_fold.round(0) + 1
        else:
            predictions_fold = _predictions_fold.argmax(axis=1) + 1
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
    # mean_f1_score = np.mean(f1_scores)
    # mean_kappa_score = np.mean(kappa_scores)
    # logger.info(f"Mean F1 score across {config['n_splits']} folds: {mean_f1_score}")
    # logger.info(f"Mean Cohen kappa score across {config['n_splits']} folds: {mean_kappa_score}")
    # OOFでの評価結果を算出
    oof_f1_score = f1_score(actual_labels, predictions, average="weighted")
    oof_kappa_score = cohen_kappa_score(actual_labels, predictions, weights="quadratic")
    logger.info(f"Out-Of-Fold F1 score: {oof_f1_score}")
    logger.info(f"Out-Of-Fold Cohen kappa score: {oof_kappa_score}")


if __name__ == "__main__":

    logger.info(f"【条件： {mode}】Deverta-V3 実行開始")
    cross_validate(config)

    ## s3にファイル移動

    logger.info(f"実行完了")
