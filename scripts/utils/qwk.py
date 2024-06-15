#!/usr/bin/env python
# coding: utf-8

import os

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "..", "config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


import numpy as np
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score


## Quadratic Weighted Kappaに関する関数
def quadratic_weighted_kappa(y_true, y_pred):
    """
    二次加重カッパ（QWK: Quadratic Weighted Kappa）を計算する評価関数。
    これは二人の評価者が与えた離散的な数値スコア間の一致度を測る指標であり、
    設定パラメータ "config['avg_train_score']" を基にスコアを調整し、
    XGBoostおよびLightGBMモデルの予測値を異なる方法で処理します。

    Args:
        y_true (np.array または同様のデータ構造): 実際のラベル。
        y_pred (np.array, xgb.DMatrix, または同様のデータ構造): 予測スコアで、numpy配列またはXGBoostのDMatrixオブジェクトのいずれかです。

    Returns:
        tuple: 文字列 'QWK' と計算されたカッパスコアの浮動小数点数を含むタプル。XGBoostモデル以外では、
               追加の真偽値 'True' を含むタプルを返し、計算が成功したことを示します。

    Note:
        - この関数は `y_pred` が LightGBM モデルのための numpy 配列または XGBoost モデルの DMatrix であることを想定しています。
        - スコアは `y_true` および `y_pred` に "config['avg_train_score']" の値を加えて調整され、
          さらに `y_pred` を1から6の範囲でクリップし、丸められます。
        - この関数を呼び出すスコープで 'config' 辞書とそのキー 'avg_train_score' が定義されている必要があります。

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

        y_true = (y_true.get_label() + config["avg_train_score"]).round()
        y_pred = (y_pred + config["avg_train_score"]).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return "QWK", qwk

    else:
        # For lgb
        y_true = y_true + config["avg_train_score"]
        y_pred = (y_pred + config["avg_train_score"]).clip(1, 6).round()
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return "QWK", qwk, True


def qwk_obj(y_true, y_pred):
    """
    カスタム損失関数として動作し、勾配ブースティングモデル（特にXGBoostやLightGBM）で使用するための勾配とヘッセ行列を計算します。
    この関数は、モデルの予測値と実際の値に基づいて、予測誤差の勾配とヘッセ行列を求め、モデルの学習プロセスで使用します。
    具体的には、設定されたパラメータ "config['avg_train_score']" と "config['var_train_score']" を使用して予測値とラベルを調整し、
    その後、損失関数に基づいて勾配とヘッセ行列を導出します。

    Args:
        y_true (np.array): 実際のラベルの配列。
        y_pred (np.array): モデルによって出力された予測値の配列。

    Returns:
        勾配の配列とヘッセ行列の配列を含むタプル。これにより、モデルの学習アルゴリズムが
        パラメータを効果的に更新できるようにします。

    Note:
        1. 実際のラベルと予測値に config から取得した "config['avg_train_score']" の値を加算して調整します。
        2. 調整後の予測値を 1 から 6 の範囲にクリップし、整数に丸めます。
        3. 調整された予測値とラベルから二次の損失関数 'f' とその正則化項 'g' を計算します。
        4. 損失関数から勾配 'grad' とヘッセ行列 'hess' を計算し、これらをモデルの学習プロセスに返します。
    """

    labels = y_true + config["avg_train_score"]
    preds = y_pred + config["avg_train_score"]
    preds = preds.clip(1, 6)
    f = 1 / 2 * np.sum((preds - labels) ** 2)
    g = 1 / 2 * np.sum((preds - config["avg_train_score"]) ** 2 + config["var_train_score"])
    df = preds - labels
    dg = preds - config["avg_train_score"]
    grad = (df / g - f * dg / g**2) * len(labels)
    hess = np.ones(len(labels))

    return grad, hess
