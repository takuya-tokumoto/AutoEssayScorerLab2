#!/usr/bin/env python
# coding: utf-8

## config.yamlの読み込み
import yaml
with open("config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

## Import
import os
import numpy as np
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
import xgboost as xgb
from typing import Dict, Tuple
from sklearn.metrics import cohen_kappa_score
from utils import *
import torch
import joblib

## 関数
def quadratic_weighted_kappa(y_true, y_pred):
    """
    二次加重カッパ（QWK: Quadratic Weighted Kappa）を計算する関数です。
    これは二人の評価者が与えた離散的な数値スコア間の一致度を測る指標で、
    設定パラメータ 'a' を基にスコアを調整し、XGBoostおよびLightGBMモデルの予測値を異なる方法で処理します。

    Args:
        y_true (np.array または同様のデータ構造): 実際のラベル。
        y_pred (np.array, xgb.DMatrix, または同様のデータ構造): 予測スコアで、numpy配列またはXGBoostのDMatrixオブジェクトのいずれかです。

    Returns:
        tuple: 文字列 'QWK' と計算されたカッパスコアの浮動小数点数を含むタプル。XGBoostモデル以外では、
               追加の真偽値 'True' を含むタプルを返し、計算が成功したことを示します。

    Note:
        - この関数は `y_pred` が LightGBM モデルのための numpy 配列または XGBoost モデルの DMatrix であることを想定しています。
        - スコアは `y_true` および `y_pred` にグローバル設定辞書の 'a' キーの値を加えて調整され、
          さらに `y_pred` を1から6の範囲でクリップし、丸められます。
        - この関数を呼び出すスコープで 'config' 辞書とそのキー 'a' が定義されている必要があります。

    example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1, 2, 3, 3, 5])
        >>> config = {'a': 0.5}
        >>> quadratic_weighted_kappa(y_true, y_pred)
        ('QWK', 0.88, True)
    """


    if isinstance(y_pred, xgb.QuantileDMatrix):
        # XGB
        y_true, y_pred = y_pred, y_true

        y_true = (y_true.get_label() + config['a']).round()
        y_pred = (y_pred + config['a']).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk

    else:
        # For lgb
        y_true = y_true + config['a']
        y_pred = (y_pred + config['a']).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 'QWK', qwk, True

def qwk_obj(y_true, y_pred):
    """
    カスタム損失関数として動作し、勾配ブースティングモデル（特にXGBoostやLightGBM）で使用するための勾配とヘッセ行列を計算します。
    この関数は、モデルの予測値と実際の値に基づいて、予測誤差の勾配とヘッセ行列を求め、モデルの学習プロセスで使用します。
    具体的には、設定されたパラメータ 'a' と 'b' を使用して予測値とラベルを調整し、
    その後、損失関数に基づいて勾配とヘッセ行列を導出します。

    Args:
        y_true (np.array): 実際のラベルの配列。
        y_pred (np.array): モデルによって出力された予測値の配列。

    Returns:
        勾配の配列とヘッセ行列の配列を含むタプル。これにより、モデルの学習アルゴリズムが
        パラメータを効果的に更新できるようにします。

    Note:
        1. 実際のラベルと予測値に config から取得した 'a' の値を加算して調整します。
        2. 調整後の予測値を 1 から 6 の範囲にクリップし、整数に丸めます。
        3. 調整された予測値とラベルから二次の損失関数 'f' とその正則化項 'g' を計算します。
        4. 損失関数から勾配 'grad' とヘッセ行列 'hess' を計算し、これらをモデルの学習プロセスに返します。
    """

    labels = y_true + config['a']
    preds = y_pred + config['a']
    preds = preds.clip(1, 6)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-config['a'])**2 + config['b'])
    df = preds - labels
    dg = preds - config['a']
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))

    return grad, hess

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
        predicted = 0.76*self.light.predict(X)
        predicted += 0.24*self.xgb_regressor.predict(X)

        return predicted
