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

## 学習データ用意
load_path = path_to.train_all_mart_dir
train_data = pl.read_csv(load_path)
# 目的変数と説明変数を分割
X = train_data.drop(config['target'])
y = train_data[config['target']]

## fold別に分割して保存
skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['SEED'])
for i, (train_index, valid_index) in enumerate(skf.split(X.to_pandas(), y.to_pandas())):
    print('fold', i)
    train_fold_df = train_data[train_index]
    valid_fold_df = train_data[valid_index]

    # ディレクトリ作成    
    base_fold_dir: Path = path_to.middle_mart_dir / f'fold_{i}/'
    base_fold_dir.mkdir(parents=True, exist_ok=True)

    # CSVファイルとして保存
    train_fold_save_dir: Path = base_fold_dir / 'train_fold.csv'
    train_fold_df.write_csv(train_fold_save_dir)
    valid_fold_save_dir: Path = base_fold_dir / 'valid_fold.csv'
    valid_fold_df.write_csv(valid_fold_save_dir)