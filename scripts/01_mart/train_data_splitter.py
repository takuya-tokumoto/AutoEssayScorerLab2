#!/usr/bin/env python
# coding: utf-8

## config.yamlの読み込み
import yaml
with open("config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

## Import
import gc
import os
import numpy as np
import pandas as pd
import random
import pickle
import sys
import polars as pl
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.path import PathManager
from utils.data import *

## パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)

## ディレクトリの準備
if not os.path.exists(path_to.input_dir):
    path_to.input_dir.mkdir()
if not os.path.exists(path_to.middle_mart_dir):
    path_to.middle_mart_dir.mkdir()

## 読み込み
load_path = path_to.train_all_mart_dir
train_data = pl.read_csv(load_path)
# 目的変数を分割して準備
X = train_data.drop(config['target'])
y = train_data[config['target']]

## fold別に分割して保存
skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['SEED'])
for i, (train_index, valid_index) in enumerate(skf.split(X.to_pandas(), y.to_pandas())):
    print('fold', i)
    train_fold_df = train_data[train_index]
    valid_fold_df = train_data[valid_index]

    # ディレクトリ作成    
    if not os.path.exists(os.path.join(path_to.middle_mart_dir, f'fold_{i}')):
        os.mkdir(os.path.join(path_to.middle_mart_dir, f'fold_{i}'))

    # CSVファイルとして保存
    train_fold_df.write_csv(os.path.join(path_to.middle_mart_dir, f'fold_{i}', f'train_fold.csv'))
    valid_fold_df.write_csv(os.path.join(path_to.middle_mart_dir, f'fold_{i}', f'valid_fold.csv'))