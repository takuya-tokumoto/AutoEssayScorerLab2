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
from pathlib import Path
# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.path import PathManager
from utils.data import CreateDataset

## パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)

## ディレクトリ作成
path_to.middle_mart_dir.mkdir(parents=True, exist_ok=True)
path_to.train_logs_dir.mkdir(parents=True, exist_ok=True)
path_to.vectorizer_weight_dir.mkdir(parents=True, exist_ok=True)

## データ読み込み＆特徴量加工
create_dataset = CreateDataset(s3_dir, config)
train = create_dataset.preprocessing_train()
test = create_dataset.preprocessing_test()

## 保存
save_path = path_to.train_all_mart_dir
train.to_csv(save_path, index=False)
save_path = path_to.test_all_mart_dir
test.to_csv(save_path, index=False)