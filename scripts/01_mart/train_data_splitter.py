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
from utils import *

# 設定を読み込む
config_path = './00_param/config.yaml'
config = load_config(config_path)

## ディレクトリの準備
if not os.path.exists(config['output_dir']):
    os.mkdir(config['output_dir'])
if not os.path.exists(os.path.join(config['output_dir'], config['model_name'])):
    os.mkdir(os.path.join(config['output_dir'], config['model_name']))


## 読み込み
train_path = os.path.join(config['output_dir'], config['model_name'], 'train_all.csv')
train_data = pl.read_csv(train_path)
X = train_data.drop(config['target'])
y = train_data[config['target']]


## 分割して保存
skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['SEED'])
for i, (train_index, valid_index) in enumerate(skf.split(X.to_pandas(), y.to_pandas())):
    print('fold', i)
    train_fold_df = train_data[train_index]
    valid_fold_df = train_data[valid_index]

    # ディレクトリ作成    
    if not os.path.exists(os.path.join(config['output_dir'], config['model_name'], f'fold_{i}')):
        os.mkdir(os.path.join(config['output_dir'], config['model_name'], f'fold_{i}'))

    # CSVファイルとして保存
    train_fold_df.write_csv(os.path.join(config['output_dir'], config['model_name'], f'fold_{i}', f'train_fold.csv'))
    valid_fold_df.write_csv(os.path.join(config['output_dir'], config['model_name'], f'fold_{i}', f'valid_fold.csv'))