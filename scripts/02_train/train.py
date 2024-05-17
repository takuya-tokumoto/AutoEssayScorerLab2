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
import sys
import pickle
from sklearn.metrics import f1_score, cohen_kappa_score
from pathlib import Path
import torch
# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.path import PathManager
from utils.model import Trainer
from utils.qwk import quadratic_weighted_kappa, qwk_obj

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
        'reg_alpha': 2,
        'reg_lambda': 0.1,
        'n_estimators': 700,
        'random_state': 42,
        'extra_trees': True,
        'class_weight': 'balanced',
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
        'verbosity': - 1
    },
    'xgb': {
        'objective': qwk_obj,  # `utils.model.py`定義されている関数を指定
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
    """データ読み込み"""

    return pd.read_csv(path)

def prepare_data(input_data, feature_select):
    """データを指定の変数で絞りこんだうえで学習データ(X, y, y_int)を作成"""

    X = input_data[feature_select].astype(np.float32).values
    y = input_data[config['target']].astype(np.float32).values - config['avg_train_score']
    y_int = input_data[config['target']].astype(int).values

    return X, y, y_int

def cross_validate(config):
    f1_scores = []
    kappa_scores = []
    predictions = []
    
    for i in range(config['n_splits']):

        ## データの読み込み
        train_path: Path = path_to.middle_mart_dir / f'fold_{i}/train_fold.csv'
        train_data = load_data(train_path)
        valid_path: Path = path_to.middle_mart_dir / f'fold_{i}/valid_fold.csv'
        valid_data = load_data(valid_path)

        # ディレクトリの準備
        model_fold_path: Path = path_to.models_weight_dir / f'fold_{i}/'
        model_fold_path.mkdir(parents=True, exist_ok=True)
        
        ### 特徴量の絞り込み計算 -> 変数重要度上位13,000件をピックアップ
        # ## データ準備
        # feature_all = list(filter(lambda x: x not in ['essay_id','score'], train_data.columns))
        # train_X, train_y, train_y_int = prepare_data(train_data, feature_all)
        # valid_X, valid_y, valid_y_int = prepare_data(valid_data, feature_all)
        # ## 全特徴量含めて学習
        # trainer_all = Trainer(config, model_params)
        # trainer_all.initialize_models()
        # trainer_all.train(train_X, train_y, valid_X, valid_y)
        # ## 変数重要度を取得
        # fse = pd.Series(trainer_all.light.feature_importances_, feature_all)
        # feature_select = fse.sort_values(ascending=False).index.tolist()[:13000]
        # ## feature_select リストを pickle ファイルとして保存
        # with open(model_fold_path / 'feature_select.pickle', 'wb') as f:
        #     pickle.dump(feature_select, f)
        
        # pickle ファイルから feature_select リストを読み込む
        save_path = os.path.join(path_to.models_weight_dir / 'feature_select.pickle')
        with open(save_path, 'rb') as f:
            feature_select = pickle.load(f)

        ### 特徴量を絞り込んだうえでモデル学習
        ## データ準備
        train_X, train_y, train_y_int = prepare_data(train_data, feature_select)
        valid_X, valid_y, valid_y_int = prepare_data(valid_data, feature_select)
        ## 学習
        trainer = Trainer(config, model_params)
        trainer.initialize_models()
        trainer.train(train_X, train_y, valid_X, valid_y)
        ## 学習結果を保存
        trainer.save_weight(model_fold_path)

        ## 学習結果を評価
        predictions_fold = trainer.predict(valid_X)
        predictions_fold = predictions_fold + config['avg_train_score']
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