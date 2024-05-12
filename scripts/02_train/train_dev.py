#!/usr/bin/env python
# coding: utf-8

## config.yamlの読み込み
import yaml
with open("config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

## Import
import os
import numpy as np
import pandas as pd
import polars as pl
import sys
import pickle
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import joblib
from pathlib import Path
import torch
# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.path import PathManager
from utils.model import *

## パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)

# モデルパラメータ
model_params = {
    'lgbm': {
        'objective': qwk_obj,  # qwk_objは事前に定義されている関数を指定
        'metrics': 'None',
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 10,
        'colsample_bytree': 0.3,
        'reg_alpha': 2.,
        'reg_lambda': 0.1,
        'n_estimators': 700,
        'random_state': 42,
        'extra_trees': True,
        'class_weight': 'balanced',
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
        'verbosity': - 1
    },
    'xgb': {
        'objective': qwk_obj,  # qwk_objは事前に定義されている関数を指定
        'metrics': 'None',
        'learning_rate': 0.1,
        'max_depth': 5,
        'num_leaves': 10,
        'colsample_bytree': 0.5,
        'reg_alpha': 1.0,
        'reg_lambda': 0.1,
        'n_estimators': 1024,
        'random_state': 42,
        'extra_trees': True,
        'class_weight': 'balanced',
        'tree_method': "hist",
        'device': "gpu" if torch.cuda.is_available() else "cpu"
    }
}

def load_data(path):
    return pd.read_csv(path)

def load_features(input_dir):
    with open(input_dir, "rb") as f:
        feature_select = pickle.load(f)

    return feature_select

def prepare_data(input_data, feature_select):
    feature_select = [feature for feature in feature_select if feature in input_data.columns]
    X = input_data[feature_select].astype(np.float32).values
    y = input_data[config['target']].astype(np.float32).values - config['a']
    y_int = input_data[config['target']].astype(int).values

    return X, y, y_int

def cross_validate(config):
    f1_scores = []
    kappa_scores = []
    predictions = []
    
    for i in range(config['n_splits']):

        ## データの読み込み
        train_path = os.path.join(path_to.middle_mart_dir, f'fold_{i}', f'train_fold.csv')
        train_data = load_data(train_path)
        valid_path = os.path.join(path_to.middle_mart_dir, f'fold_{i}', f'valid_fold.csv')
        valid_data = load_data(valid_path)

        ## 特徴量の選択
        load_path = path_to.aes2_cache_dir
        feature_select = load_features(load_path)
        train_X, train_y, train_y_int = prepare_data(train_data, feature_select)
        valid_X, valid_y, valid_y_int = prepare_data(valid_data, feature_select)

        ## 学習実施
        trainer = Trainer(config, model_params)
        trainer.initialize_models()
        trainer.train(train_X, train_y, valid_X, valid_y)

        ## 学習結果を保存
        # ディレクトリの準備
        model_path = path_to.models_weight_dir
        model_fold_path = os.path.join(path_to.models_weight_dir, f'fold_{i}')
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(model_fold_path):
            os.mkdir(model_fold_path)
        # モデルの保存
        trainer.save_weight(model_fold_path)

        ## 学習結果を評価
        predictions_fold = trainer.predict(valid_X)
        predictions_fold = predictions_fold + config['a']
        predictions_fold = np.clip(predictions_fold, 1, 6).round()
        predictions.append(predictions_fold)
        # F1スコア
        f1_fold = f1_score(valid_y_int, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)
        # Cohen kappa score
        kappa_fold = cohen_kappa_score(valid_y_int, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)

        print(f'F1 score for fold {i}: {f1_fold}')
        print(f'Cohen kappa score for fold {i}: {kappa_fold}')

    ## 評価結果
    mean_f1_score = np.mean(f1_scores)
    mean_kappa_score = np.mean(kappa_scores)
    print(f"Mean F1 score across {config['n_splits']} folds: {mean_f1_score}")
    print(f"Mean Cohen kappa score across {config['n_splits']} folds: {mean_kappa_score}")

if __name__ == '__main__':
    cross_validate(config)