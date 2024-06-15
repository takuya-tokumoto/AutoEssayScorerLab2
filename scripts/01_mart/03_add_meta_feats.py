#!/usr/bin/env python
# coding: utf-8

import os

## config.yamlの読み込み
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "..", "config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

## Import
import os
import sys
from pathlib import Path

import polars as pl

# 自作関数の読み込み
repo_dir = Path(__file__).parents[2]
root_dir = Path(__file__).parents[3]
s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/data/"
sys.path.append(str(repo_dir / "scripts/"))
from utils.metafeatures import GenerateMetaFeatures
from utils.path import PathManager

## パスの設定
mode = config["model_name"]
path_to = PathManager(s3_dir, mode)


if __name__ == "__main__":

    for i in range(config["n_splits"]):
        fold_num = str(i)
        print("fold", fold_num)

        ## データの読み込み
        train_path: Path = path_to.skf_mart_dir / f"fold_{fold_num}/train_fold.csv"
        train_data = pl.read_csv(train_path)
        valid_path: Path = path_to.skf_mart_dir / f"fold_{fold_num}/valid_fold.csv"
        valid_data = pl.read_csv(valid_path)

        ## fold別に特徴量を作成して追加
        generate_meta = GenerateMetaFeatures(s3_dir, config, fold_num, train_data, valid_data)
        train_data_add = generate_meta.preprocessing_train()
        valid_data_add = generate_meta.preprocessing_test()

        # 以降、full_text項目は不要なためドロップ
        train_data_add = train_data_add.drop(["full_text"], axis=1)
        valid_data_add = valid_data_add.drop(["full_text"], axis=1)

        # ディレクトリ準備
        base_fold_dir: Path = path_to.add_meta_mart_dir / f"fold_{fold_num}/"
        base_fold_dir.mkdir(parents=True, exist_ok=True)

        # CSVファイルとして保存
        train_fold_save_dir: Path = base_fold_dir / "train_fold_add_meta.csv"
        train_data_add.to_csv(train_fold_save_dir, index=False)
        valid_fold_save_dir: Path = base_fold_dir / "valid_fold_add_meta.csv"
        valid_data_add.to_csv(valid_fold_save_dir, index=False)
