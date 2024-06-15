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
import numpy as np
import pandas as pd
import polars as pl
import sys
import pickle
from sklearn.metrics import f1_score, cohen_kappa_score
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
import torch
# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.path import PathManager
from utils.model import Trainer
from utils.model import quadratic_weighted_kappa, qwk_obj

## パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)

# モデルパラメータ
model_params = {
    'lgbm': {
        'objective': qwk_obj,  # `utils.model.py`定義されている関数を指定
        'metrics': 'None',
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 10,
        'colsample_bytree': 0.3,
        'reg_alpha': 0.7,
        'reg_lambda': 0.1,
        'n_estimators': 700,
        'random_state': 412,
        'extra_trees': True,
        'class_weight': 'balanced',
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
        'verbosity': - 1
    },
}

if __name__ == '__main__':

    ## 学習データ用意
    load_path = path_to.train_all_mart_dir
    train_data = pd.read_csv(load_path)
    # 分割
    feature_names = list(filter(lambda x: x not in ['essay_id','score','full_text'], train_data.columns))
    features = feature_names
    # 目的変数と特徴量に分割
    X = train_data[feature_names].astype(np.float32).values
    y_split = train_data['score'].astype(int).values
    y = train_data['score'].astype(np.float32).values - config['avg_train_score']

    # ディレクトリの準備
    path_to.models_weight_dir.mkdir(parents=True, exist_ok=True)
    
    fse = pd.Series(0, index=features)
    f1_scores = []
    kappa_scores = []
    models = []
    predictions = []
    callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75,first_metric_only=True)]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in skf.split(X, y_split):

        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold, y_test_fold_int = y[train_index], y[test_index], y_split[test_index]

        model = lgb.LGBMRegressor(**model_params['lgbm'])
        
        predictor = model.fit(X_train_fold,
                        y_train_fold,
                        eval_names=['train', 'valid'],
                        eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                        eval_metric=quadratic_weighted_kappa,
                        callbacks=callbacks)

        models.append(predictor)
        predictions_fold = predictor.predict(X_test_fold)
        predictions_fold = predictions_fold + config['avg_train_score']
        predictions_fold = predictions_fold.clip(1, 6).round()
        predictions.append(predictions_fold)
        f1_fold = f1_score(y_test_fold_int, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        kappa_fold = cohen_kappa_score(y_test_fold_int, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)

        print(f'F1 score across fold: {f1_fold}')
        print(f'Cohen kappa score across fold: {kappa_fold}')

        fse += pd.Series(predictor.feature_importances_, features)

    feature_select = fse.sort_values(ascending=False).index.tolist()[:13000]
    ## feature_select リストを pickle ファイルとして保存
    with open(path_to.models_weight_dir / 'feature_select.pickle', 'wb') as f:
        pickle.dump(feature_select, f)