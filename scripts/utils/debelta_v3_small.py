#!/usr/bin/env python
# coding: utf-8

## config.yamlの読み込み
import yaml
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config.yaml')
with open(config_path, "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

## Import
import os
import sys
import warnings
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score, cohen_kappa_score
from tokenizers import AddedToken
from datetime import datetime, timedelta, timezone
import logging
import dill
from pathlib import Path
warnings.simplefilter('ignore')
# 自作関数の読み込み

class CFG:
    MODEL_NAME = "microsoft/deberta-v3-small"
    # n_splits = 5
    seed = 42
    max_length = 1024 # to avoid truncating majority of essays.
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
#seed_everything(seed=CFG.seed)

## Tokenizer
class Tokenize(object):
    """train, valid, (test)データに対してトークン化処理"""

    def __init__(self, train, valid, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid
        self.max_length = max_length
        
    def get_dataset(self, df):
        ds = Dataset.from_dict({
                'essay_id': [e for e in df['essay_id']],
                'full_text': [ft for ft in df['full_text']],
                'label': [s for s in df['label']],
            })
        return ds
        
    def tokenize_function(self, example):
        tokenized_inputs = self.tokenizer(
            example['full_text'], truncation=True, max_length=self.max_length
        )
        return tokenized_inputs
    
    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)
        
        tokenized_train = train_ds.map(
            self.tokenize_function, batched=True
        )
        tokenized_valid = valid_ds.map(
            self.tokenize_function, batched=True
        )

        return tokenized_train, tokenized_valid, self.tokenizer

def compute_metrics_for_regression(eval_pred):
    """QWK評価関数（回帰タスク）"""
    
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results

def compute_metrics_for_classification(eval_pred):
    """QWK評価関数（分類タスク）"""
    
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results

## Tranier
class DebertaV3SmallTrainer:
    def __init__(self, config, model_params):
        self.config = config
        self.model_params = model_params

    def initialize_models(self):
        """モデルの初期化"""

        ## Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
        # 新しいトークン追加
        # ("\n") : 新しい段落のため,  (" "*2) : ダブルスペースのため
        self.tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        self.tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])

        ## Config
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        # 回帰タスクの場合ドロップアウトを除外
        self.config.attention_probs_dropout_prob = 0.0 
        self.config.hidden_dropout_prob = 0.0 
        self.config.num_labels = 1 

        ## training args
        training_args = TrainingArguments(
            output_dir=model_fold_path,
            fp16=True,
            learning_rate=CFG.lr,
            per_device_train_batch_size=CFG.train_batch_size,
            per_device_eval_batch_size=CFG.eval_batch_size,
            num_train_epochs=CFG.train_epochs,
            weight_decay=CFG.weight_decay,
            evaluation_strategy="no",  # 評価を無効にする 'epoch',
            metric_for_best_model='qwk',
            save_strategy="no",  # 評価を無効にする 'epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to='none',
            warmup_ratio=CFG.warmup_ratio,
            lr_scheduler_type='linear', # "cosine" or "linear" or "constant"
            optim='adamw_torch',
            logging_first_step=True,
        )

        ## model
        self.model = AutoModelForSequenceClassification.from_pretrained(CFG.MODEL_NAME, config=config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def process_tokenizer(self, input_data, tokenizer):
        """トークンナイズ化したデータセットを返す"""

        def get_dataset(self, df):
            ds = Dataset.from_dict({
                    'essay_id': [e for e in df['essay_id']],
                    'full_text': [ft for ft in df['full_text']],
                    'label': [s for s in df['label']],
                })
            return ds

        def tokenize_function(self, example):
            tokenized_inputs = self.tokenizer(
                example['full_text'], truncation=True, max_length=CFG.max_length
            )
            return tokenized_inputs

        ds = self.get_dataset(input_data)
        tokenized_ds = ds.map(
            tokenize_function, batched=True
        )

        return tokenized_ds, tokenizer

    def train_and_save(self, tokenized_train):
        """モデルの学習"""

        ## TRAIN WITH TRAINER
        trainer = Trainer( 
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train,
            # eval_dataset=tokenized_valid,
            eval_dataset=None,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
            # compute_metrics=compute_metrics
        )

        ## 学習
        self.trainer.train()

    def load_weight(self, save_path):
        """モデルの重みをロード"""

        ## Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(save_path)

        ## Config
        # 不要

        ## training args
        training_args = torch.load(save_path / 'training_args.bin')

        ## model
        model = AutoModelForSequenceClassification.from_pretrained(save_path)

    def predict(self, X):
        """予測結果を出力"""

        trainer = transformers.Trainer( 
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_test,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
        )

        # SAVE PREDICTIONS
        predictions = trainer.predict(self.tokenized_test).predictions
