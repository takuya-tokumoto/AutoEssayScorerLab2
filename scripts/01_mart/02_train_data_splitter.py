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

if __name__ == '__main__':
    
    ## パスの設定
    mode = config["model_name"]
    path_to = PathManager(s3_dir, mode)

    ## 追加データ用意
    suppliment_data = pl.read_csv(load_path).filter(pl.col("suppliment_flg")==1)
    # 目的変数と説明変数を分割
    X_sup = suppliment_data.drop(config['target'])
    y_sup = suppliment_data[config['target']]

    ## fold数にあわせて追加データも分割準備
    suppliment_data_list = []
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['SEED'])
    for i, (_, test_idx) in enumerate(skf.split(X_sup.to_pandas(), y_sup.to_pandas())):
        fold_num = str(i)

        # 各foldで抽出
        suppliment_frag = suppliment_data[test_idx]
        suppliment_frag = suppliment_frag.with_columns(pl.lit(fold_num).alias("fold"))

        # list(suppliment_data_list)に格納
        suppliment_data_list.append(suppliment_frag)

    # 縦結合
    suppliment_w_fold = pl.concat(suppliment_data_list, how="vertical")

    ## 学習データ用意
    load_path = path_to.train_all_mart_dir
    train_data = pl.read_csv(load_path).filter(pl.col("suppliment_flg")==0)
    # 目的変数と説明変数を分割
    X = train_data.drop(config['target'])
    y = train_data[config['target']]

    ## fold別に分割して保存
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['SEED'])
    for i, (train_index, valid_index) in enumerate(skf.split(X.to_pandas(), y.to_pandas())):
        fold_num = str(i)
        print('fold', fold_num)

        ## 学習データを分割
        train_fold_df = train_data[train_index]
        valid_fold_df = train_data[valid_index]

        ## 補足データを追加
        # 指定のfoldを抽出
        suppliment_data_frag = suppliment_w_fold.filter(pl.col("fold")==fold_num)
        suppliment_data_frag = suppliment_data_frag.drop("fold")
        # Union
        train_fold_df = pl.concat([train_fold_df, suppliment_data_frag], how="vertical")

        ## 以降、suppliment_flg項目は不要なためドロップ
        train_fold_df = train_fold_df.drop("suppliment_flg")
        valid_fold_df = valid_fold_df.drop("suppliment_flg")

        ## ディレクトリ作成    
        base_fold_dir: Path = path_to.skf_mart_dir / f'fold_{fold_num}/'
        base_fold_dir.mkdir(parents=True, exist_ok=True)

        ## CSVファイルとして保存
        train_fold_save_dir: Path = base_fold_dir / 'train_fold.csv'
        train_fold_df.write_csv(train_fold_save_dir)
        valid_fold_save_dir: Path = base_fold_dir / 'valid_fold.csv'
        valid_fold_df.write_csv(valid_fold_save_dir)