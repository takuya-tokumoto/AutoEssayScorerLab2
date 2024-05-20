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
import numpy as np
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
import joblib
# 自作関数の読み込み
from .qwk import quadratic_weighted_kappa, qwk_obj

## モデル用クラス
class Trainer:
    def __init__(self, config, model_params):
        self.config = config
        self.model_params = model_params
        self.light = None  # LightGBM モデル
        self.best_light_iteration = 150 # 仮で150
        self.xgb_regressor = None  # XGBoost モデル
        self.best_xgb_iteration = 150 # 仮で150

    def initialize_models(self):
        """モデルの初期化"""

        try:
            self.light = lgb.LGBMRegressor(**self.model_params['lgbm'])
            self.xgb_regressor = xgb.XGBRegressor(**self.model_params['xgb'])
        except KeyError as e:
            print(f"Error initializing models: {e}")
            raise

    def train_with_early_stopping(self, X_train, y_train):
        """早期停止を用いた学習と最適な学習回数の取得"""

        ## 検証用データを(X_train, y_train)から準備
        X_train_part, X_early_stopping, y_train_part, y_early_stopping = train_test_split(
            X_train, y_train, test_size=0.4, random_state=42)
        
        ## LightGBM
        # モデルの呼び出し
        _light = self.light
        # callback
        callbacks = [
            log_evaluation(period=25), 
            early_stopping(stopping_rounds=75,
            first_metric_only=True)
        ]
        # 学習
        _light.fit(
            X_train_part, y_train_part,
            eval_set=[(X_train_part, y_train_part), (X_early_stopping, y_early_stopping)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks
        )
        # 最適な学習回数の保存
        self.best_light_iteration = _light.best_iteration_

        ## XGB
        # モデルの呼び出し
        _xgb_regressor = self.xgb_regressor
        # callback  
        xgb_callbacks = [
            xgb.callback.EvaluationMonitor(period=25),
            xgb.callback.EarlyStopping(75, metric_name="QWK", maximize=True, save_best=True)
        ]
        # 学習
        _xgb_regressor.fit(
            X_train_part, y_train_part,
            eval_set=[(X_train_part, y_train_part), (X_early_stopping, y_early_stopping)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=xgb_callbacks
        )
        # 最適な学習回数の保存
        self.best_xgb_iteration = _xgb_regressor.best_iteration
    
    def train(self, X_train, y_train):
        """モデルの学習"""

        self.light.n_estimators = self.best_light_iteration
        # print(f"lgbmイテレーション回数：{self.light.n_estimators}")
        self.light.fit(X_train, y_train)

        self.xgb_regressor.n_estimators = self.best_xgb_iteration + 1  # XGBoost は0ベースのインデックスなので、1を加える
        # print(f"xgbイテレーション回数：{self.xgb_regressor.n_estimators}")
        self.xgb_regressor.fit(X_train, y_train)

    def save_weight(self, save_path):
        """モデルの学習結果を保存"""

        joblib.dump(self.light, os.path.join(save_path, 'lgbm_model.pkl'))
        self.xgb_regressor.save_model(os.path.join(save_path, 'xgb_model.json'))

    def load_weight(self, save_path):
        """モデルの重みをロード"""
        
        self.light = joblib.load(os.path.join(save_path, 'lgbm_model.pkl'))
        self.xgb_regressor.load_model(os.path.join(save_path, 'xgb_model.json'))
    
    def predict(self, X):
        """予測結果を出力"""

        predicted = None
        predicted = (
            0.76*self.light.predict(X)
            + 0.24*self.xgb_regressor.predict(X)
        )

        return predicted