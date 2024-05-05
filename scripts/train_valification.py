#!/usr/bin/env python
# coding: utf-8

## Import
import os
import numpy as np
import pandas as pd
import polars as pl
import pickle
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import joblib

# Config
config = {
    'model_name': 'test',
    'input_dir': '../data/input/',
    'output_dir': '../data/input/middle/',
    'model_path': '../data/models/',
    'target': 'score',
    'SEED': 2018,
    'n_splits': 3,
    'a': 2.998,
    'b': 1.092,
}


def load_data(path):
    return pl.read_csv(path)


def load_features(input_dir):
    with open(os.path.join(input_dir, "aes2-cache/feature_select.pickle"), "rb") as f:
        feature_select = pickle.load(f)
    return feature_select


def prepare_data(input_data, feature_select):
    feature_select = [feature for feature in feature_select if feature in input_data.columns]
    X = input_data[feature_select].astype(np.float32).values
    y = input_data[config['target']].astype(np.float32).values - config['a']
    return X, y


def cross_validate(config):
    f1_scores = []
    kappa_scores = []
    
    for i in range(config['n_splits']):
        train_path = os.path.join(config['output_dir'], config['model_name'], f'train_fold_{i}.csv')
        train_data = load_data(train_path)
        valid_path = os.path.join(config['output_dir'], config['model_name'], f'valid_fold_{i}.csv')
        valid_data = load_data(valid_path)

        feature_select = load_features(config['input_dir'])
        train_X, train_y = prepare_data(train_data, feature_select)
        valid_X, valid_y = prepare_data(valid_data, feature_select)

        # Assuming Trainer is a previously defined and imported class
        trainer = Trainer()
        trainer.train(train_X, train_y, valid_X, valid_y)
        trainer.save_weight()

        predictions_fold = trainer.predict(valid_X)
        predictions_fold = predictions_fold + config['a']
        predictions_fold = np.clip(predictions_fold, 1, 6).round()
        predictions.append(predictions_fold)

        f1_fold = f1_score(y, predictions_fold, average='weighted')
        f1_scores.append(f1_fold)

        kappa_fold = cohen_kappa_score(y, predictions_fold, weights='quadratic')
        kappa_scores.append(kappa_fold)

        print(f'F1 score for fold {i}: {f1_fold}')
        print(f'Cohen kappa score for fold {i}: {kappa_fold}')

    mean_f1_score = np.mean(f1_scores)
    mean_kappa_score = np.mean(kappa_scores)
    print(f'Mean F1 score across {config['n_splits']} folds: {mean_f1_score}')
    print(f'Mean Cohen kappa score across {config['n_splits']} folds: {mean_kappa_score}')

if __name__ == '__main__':
    cross_validate(config)