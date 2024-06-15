#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score, f1_score
from tokenizers import AddedToken
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from utils.configs import get_transformer_config
from utils.logging import set_logger

logger = set_logger(__name__)

tr_config = get_transformer_config()


class EssayScorer:
    def __init__(self, model_name: str, max_length: int):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = self._setup_tokenizer()

    def _setup_tokenizer(self) -> PreTrainedTokenizer:
        """トークナイザーの初期化

        Returns:
            PreTrainedTokenizer: トークナイザー
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 改行やダブルホワイトスペースへの対応
        tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])
        return tokenizer

    def _tokenize_data(self, data: pd.DataFrame) -> Dataset:
        """入力データをトークナイズ化し、データセットとして返す

        Args:
            data (pd.DataFrame): 入力データ

        Returns:
            Dataset: トークナイズ化されたデータセット
        """

        def _get_dataset(df):
            ds = Dataset.from_dict(
                {
                    "essay_id": [e for e in df["essay_id"]],
                    "full_text": [ft for ft in df["full_text"]],
                    "label": [s for s in df["label"]],
                }
            )
            return ds

        def _tokenize_function(example, tokenizer, max_length):
            tokenized_inputs = tokenizer(example["full_text"], truncation=True, max_length=max_length)
            return tokenized_inputs

        # データの加工
        _dataset = _get_dataset(data)
        tokenized_data = _dataset.map(lambda x: _tokenize_function(x, self.tokenizer, self.max_length), batched=True)

        return tokenized_data

    def _setup_model(self, model_fold_path: Path, is_trained: bool):
        """モデルの設定

        Args:
            model_fold_path (Path): モデルの出力先
            is_trained (bool): 学習済か否かの判定

        Returns:
            _type_: モデル情報
        """
        if is_trained:
            # 学習済の場合は結果を取得する
            model = AutoModelForSequenceClassification.from_pretrained(model_fold_path)
        else:
            # 未学習の場合は設定を取得する
            config = AutoConfig.from_pretrained(self.model_name)

            # 回帰タスクの場合ドロップアウトを除外
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
            config.num_labels = 1

            # モデル情報を設定する
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
            model.resize_token_embeddings(len(self.tokenizer))
        return model

    def _setup_trainer(self, model, output_path: Path):
        """学習機の設定

        Args:
            model (_type_): 読み込み済モデル
            output_path (path): 出力先

        Returns:
            _type_: 学習機
        """
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_path,
            fp16=True,
            learning_rate=tr_config.lr,
            per_device_train_batch_size=tr_config.train_batch_size,
            per_device_eval_batch_size=tr_config.eval_batch_size,
            num_train_epochs=tr_config.train_epochs,
            weight_decay=tr_config.weight_decay,
            evaluation_strategy="no",  # 評価を無効にする 'epoch',
            metric_for_best_model="qwk",
            save_strategy="no",  # 評価を無効にする 'epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
            warmup_ratio=tr_config.warmup_ratio,
            lr_scheduler_type="linear",  # "cosine" or "linear" or "constant"
            optim="adamw_torch",
            logging_first_step=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        return trainer

    def train_and_evaluate(
        self, train_data: Dataset, valid_data: Dataset, valid_y_int: np.ndarray, output_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """学習と評価の実施

        Args:
            train_data (Dataset): 学習データ
            valid_data (Dataset): 検証データ
            valid_y_int (np.ndarray): 検証の正解データ
            output_path (Path): モデルの出力先

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 予測結果、F1スコア、カッパー係数
        """
        # 学習済かどうかを判定し、モデルや学習機の設定を取得
        is_trained = len(os.listdir(output_path)) > 0
        model = self._setup_model(output_path, is_trained)
        trainer = self._setup_trainer(model, output_path)

        # トークナイズ化
        tokenized_train = self._tokenize_data(train_data, self.tokenizer, self.max_length)
        tokenized_valid = self._tokenize_data(valid_data, self.tokenizer, self.max_length)

        if not is_trained:
            logger.info(f"学習を実行")
            trainer.train_dataset = tokenized_train
            trainer.train()
            logger.info(f"学習結果を保存")
            trainer.save_model(output_path)
            self.tokenizer.save_pretrained(output_path)

        predictions_fold = trainer.predict(tokenized_valid).predictions
        predictions_fold = predictions_fold.round(0) + 1
        predictions_fold = np.clip(predictions_fold, 1, 6).round()

        # 評価の実施
        f1_fold = f1_score(valid_y_int, predictions_fold, average="weighted")
        kappa_fold = cohen_kappa_score(valid_y_int, predictions_fold, weights="quadratic")

        return predictions_fold, f1_fold, kappa_fold
