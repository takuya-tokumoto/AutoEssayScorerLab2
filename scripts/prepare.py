# -*- encoding: utf-8 -*-

## Imports
import gc
import os
import numpy as np
import pandas as pd
import random
import polars as pl
from data import *

## Config
config = {
    'sampling_mode': True,
    'model_name': 'test',
    'input_dir': '../data/input/',
    'output_dir': '../data/input/middle/',
    'target': 'score',
    'SEED': 2018,
    'n_splits': 15,
}

## データ読み込み＆特徴量加工
create_dataset = CreateDataset(config)
train, test = create_dataset.pipline()

# ディレクトリの準備
if not os.path.exists(config['output_dir']):
    os.mkdir(config['output_dir'])
if not os.path.exists(os.path.join(config['output_dir'], config['model_name'])):
    os.mkdir(os.path.join(config['output_dir'], config['model_name']))

## 一時保存
train_path = os.path.join(config['output_dir'], config['model_name'], 'train_all.csv')
pl.read_csv(train_path)
test_path = os.path.join(config['output_dir'], config['model_name'], 'test_all.csv')
test.to_csv(test_path, index=False)