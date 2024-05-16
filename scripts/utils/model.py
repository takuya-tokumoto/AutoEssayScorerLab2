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
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from utils import *
import joblib
from .qwk import quadratic_weighted_kappa, qwk_obj

## モデル用クラス
class Trainer:
    def __init__(self, config, model_params):
        self.config = config
        self.model_params = model_params
        self.light = None  # LightGBM モデルの初期化を遅延
        self.xgb_regressor = None  # XGBoost モデルの初期化を遅延

    def initialize_models(self):
        """モデルの初期化"""

        try:
            self.light = lgb.LGBMRegressor(**self.model_params['lgbm'])
            self.xgb_regressor = xgb.XGBRegressor(**self.model_params['xgb'])
        except KeyError as e:
            print(f"Error initializing models: {e}")
            raise
    
    def train(self, X_train, y_train, X_valid, y_valid):
        """モデルの学習"""

        callbacks = [
            log_evaluation(period=25), 
            early_stopping(stopping_rounds=75,
            first_metric_only=True)
        ]
        self.light.fit(
            X_train, y_train,
            eval_names=['train', 'valid'],
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=callbacks
        )

        xgb_callbacks = [
            xgb.callback.EvaluationMonitor(period=25),
            xgb.callback.EarlyStopping(75, metric_name="QWK", maximize=True, save_best=True)
        ]
        self.xgb_regressor.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=quadratic_weighted_kappa,
            callbacks=xgb_callbacks
        )

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