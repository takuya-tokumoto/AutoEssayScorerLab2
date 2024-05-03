# -*- encoding: utf-8 -*-

## Imports
import gc
import os
import tqdm
import joblib
import numpy as np
import pandas as pd
import random
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from data import *

## Config
config = {
    'input_dir': '../data/input/',
    'output_dir': '../data/input/folds/',
    'target': 'score',
    'SEED': 2018,
    'n_splits': 15
}

## Load Data
def load_data(path, file_name):
    columns = [
        pl.col("full_text").str.split(by="\n\n").alias("paragraph")
    ]
    data_path = os.path.join(path, file_name)
    return pl.read_csv(data_path).with_columns(columns)

train = load_data(config['input_dir'], "learning-agency-lab-automated-essay-scoring-2/train.csv")
test = load_data(config['input_dir'], "learning-agency-lab-automated-essay-scoring-2/test.csv")

## 特徴量加工
create_dataset = CreateDataset(train, test, config)
train, test = create_dataset.pipline()

## Prepare features and targets
X_train = train.drop(columns=[config['target']])
y_train = train[config['target']]
X_test = test.drop(columns=[config['target']])
y_test = test[config['target']]

## Stratified K-Fold
skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['SEED'])

# Prepare directory to save models
if not os.path.exists(config['output_dir']):
    os.mkdir(config['output_dir'])

# Loop through each fold
for fold, (train_index, val_index) in tqdm(enumerate(skf.split(X_train, y_train))):
    # 学習データ
    train_fold = X_train.iloc[train_index]
    y_train_fold = y_train.iloc[train_index]
    # 特徴量データとターゲットを結合
    train_combined = pd.concat([train_fold, y_train_fold], axis=1)

    # 評価用データ
    valid_fold = X_train.iloc[val_index]
    y_valid_fold = y_train.iloc[val_index]
    # 特徴量データとターゲットを結合
    valid_combined = pd.concat([valid_fold, y_valid_fold], axis=1)
    
    # Save fold data
    train_fold_path = os.path.join(config['data_dir'], f'train_fold_{fold}.csv')
    valid_fold_path = os.path.join(config['data_dir'], f'valid_fold_{fold}.csv')
    
    train_combined.to_csv(train_fold_path, index=False)
    valid_combined.to_csv(train_fold_path, index=False)

    print(f"Fold {fold} data saved to {train_fold_path} and {valid_fold_path}")


