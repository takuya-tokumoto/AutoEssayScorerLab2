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
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
import catboost as cb
# from catboost import CatBoostRegressor, CatBoostClassifier
import joblib

## モデル用クラス
class CutOffTrainer:
    def __init__(self, config: dict, model_params: dict):
        self.config = config
        self.model_params = model_params
        self.light_classify = None  # LightGBM classifyモデル
        self.best_light_iteration = 150 # 仮で150
        self.cat_classify = None  # CatBoost classify モデル
        self.best_cat_iteration = 150 # 仮で150

    def initialize_models(self):
        """モデルの初期化"""
        try:
            self.light_classify = lgb.LGBMClassifier(**self.model_params['light_classify'])
            self.cat_classify = cb.CatBoostClassifier(**self.model_params['cat_classify'])
        except KeyError as e:
            print(f"Model parameter key missing: {e}")
            raise

    def train_with_early_stopping(self, X_train, y_train):
        """早期停止を用いた学習と最適な学習回数の取得"""

        ## 検証用データを(X_train, y_train)から準備
        X_train_part, X_early_stopping, y_train_part, y_early_stopping = train_test_split(
            X_train, y_train, test_size=0.4, random_state=42)
        
        ## LightGBM
        # モデルの呼び出し
        _light_classify = self.light_classify
        # callback
        callbacks = [
            lgb.callback.log_evaluation(period=25), 
            lgb.callback.early_stopping(stopping_rounds=75, first_metric_only=True)
        ]
        # 学習
        _light_classify.fit(
            X_train_part, y_train_part,
            eval_set=[(X_train_part, y_train_part), (X_early_stopping, y_early_stopping)],
            callbacks=callbacks
        )
        # 最適な学習回数の保存
        self.best_light_iteration = _light_classify.best_iteration_

        ## Catboost
        # モデルの呼び出し
        _cat_classify = self.cat_classify
        # 学習
        _cat_classify.fit(
            X_train_part, y_train_part,
            eval_set=[(X_train_part, y_train_part), (X_early_stopping, y_early_stopping)],
            use_best_model=True
        )
        # 最適な学習回数の保存
        self.best_cat_iteration = _cat_classify.get_best_iteration()
    
    def train(self, X_train, y_train):
        """モデルの学習"""

        self.light_classify.n_estimators = self.best_light_iteration
        self.light_classify.fit(X_train, y_train)

        self.cat_classify.n_estimators = self.best_cat_iteration
        self.cat_classify.fit(X_train, y_train)

    def save_weight(self, save_path):
        """モデルの学習結果を保存"""

        joblib.dump(self.light_classify, os.path.join(save_path, 'light_classify_model.pkl'))
        self.cat_classify.save_model(os.path.join(save_path, 'cat_classify_model.cbm'))

    def load_weight(self, save_path):
        """モデルの重みをロード"""
        
        self.light_classify = joblib.load(os.path.join(save_path, 'light_classify_model.pkl'))
        self.cat_classify.load_model(os.path.join(save_path, 'cat_classify_model.cbm'))
    
    def predict(self, X):
        """予測結果を出力"""

        predicted_light = self.light_classify.predict_proba(X)[:,1]
        predicted_cat = self.cat_classify.predict_proba(X)[:,1]

        return predicted_light, predicted_cat