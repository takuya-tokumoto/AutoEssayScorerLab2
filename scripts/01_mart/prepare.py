# -*- encoding: utf-8 -*-

## config.yamlの読み込み
import yaml
with open("config.yaml", "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

## Import
import gc
import os
import sys
import numpy as np
import pandas as pd
import random
from pathlib import Path
import polars as pl
# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.path import PathManager
from utils.data import *

## パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)

## ディレクトリ作成
if not os.path.exists(path_to.mid_dir):
    path_to.mid_dir.mkdir()
if not os.path.exists(path_to.middle_files_dir):
    path_to.middle_files_dir.mkdir()

## データ読み込み＆特徴量加工
create_dataset = CreateDataset(s3_dir, config)
train = create_dataset.preprocessing_train()
test = create_dataset.preprocessing_test()

## 保存
save_path = path_to.train_all_mart_dir
train.to_csv(save_path, index=False)
save_path = path_to.test_all_mart_dir
test.to_csv(save_path, index=False)