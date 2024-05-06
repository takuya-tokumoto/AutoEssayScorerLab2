# -*- encoding: utf-8 -*-

## Imports
import gc
import os
import sys
import numpy as np
import pandas as pd
import random
import polars as pl
from data import *
import yaml
from utils import *

# 設定を読み込む
config_path = './00_param/config.yaml'
config = load_config(config_path)

## データ読み込み＆特徴量加工
create_dataset = CreateDataset(config)
train = create_dataset.preprocessing_train()
test = create_dataset.preprocessing_test()

# ディレクトリの準備
if not os.path.exists(config['output_dir']):
    os.mkdir(config['output_dir'])
if not os.path.exists(os.path.join(config['output_dir'], config['model_name'])):
    os.mkdir(os.path.join(config['output_dir'], config['model_name']))

## 一時保存
train_path = os.path.join(config['output_dir'], config['model_name'], 'train_all.csv')
train.to_csv(train_path, index=False)
test_path = os.path.join(config['output_dir'], config['model_name'], 'test_all.csv')
test.to_csv(test_path, index=False)