#!/usr/bin/env python
# coding: utf-8

## Imports
import gc
import os
import numpy as np
import pandas as pd
import random
import pickle
import polars as pl
from sklearn.model_selection import StratifiedKFold


## Config
config = {
    'model_name': 'test',
    'input_dir': '../data/input/',
    'output_dir': '../data/input/middle/',
    'target': 'score',
    'SEED': 2018,
    'n_splits': 3,
    'a': 2.998,
    'b': 1.092,
}


## ディレクトリの準備
if not os.path.exists(config['output_dir']):
    os.mkdir(config['output_dir'])
if not os.path.exists(os.path.join(config['output_dir'], config['model_name'])):
    os.mkdir(os.path.join(config['output_dir'], config['model_name']))


## 読み込み
train_path = os.path.join(config['output_dir'], config['model_name'], 'train_all.csv')
train_data = pl.read_csv(train_path)


## 特徴量選択
# # 選択する変数を読み込み
# with open(os.path.join(config['input_dir'], "aes2-cache/feature_select.pickle"), "rb") as f:
#     feature_select = pickle.load(f)
#     feature_select = [feature for feature in feature_select if not feature.startswith('deberta_oof_')]
# # Converting the 'text' column to string type and assigning to X
# X = train_data[feature_select].astype(np.float32).values
# # Converting the 'score' column to integer type and assigning to y
# y_split = train_data['score'].astype(int).values
# y = train_data['score'].astype(np.float32).values - config['a']

X = train_data.drop(config['target'])
y = train_data[config['target']]

## 分割して保存
skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['SEED'])
for i, (train_index, valid_index) in enumerate(skf.split(X.to_pandas(), y.to_pandas())):
    print('fold', i)
    train_fold_df = train_data[train_index]
    valid_fold_df = train_data[valid_index]

    # CSVファイルとして保存
    train_fold_df.write_csv(os.path.join(config['output_dir'], config['model_name'], f'train_fold_{i}.csv'))
    valid_fold_df.write_csv(os.path.join(config['output_dir'], config['model_name'], f'valid_fold_{i}.csv'))