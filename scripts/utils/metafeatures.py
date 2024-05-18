#!/usr/bin/env python
# coding: utf-8

## Import
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer
import polars as pl
import joblib
from pathlib import Path
from scipy.special import softmax
import os
from glob import glob
# 自作関数の読み込み
from .path import PathManager

class GenerateMetaFeatures():

    def __init__(self, 
                 repo_dir: Path,
                 config: dict, 
                 fold_num: str,
                 train_data: pl.DataFrame = None, 
                 test_data: pl.DataFrame = None, ):

        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.path_to = PathManager(repo_dir, config["model_name"])
        self.fold_num = fold_num

    def identity_function(self, x):
        
        return x

    def fit_transform_TfidfVec(self, train_data, save_path):
        """
        この関数は、クラス内の学習データに対してTF-IDFベクトル化を行い、
        結果をDataFrame形式で返す。各DataFrameには、テキストデータがTF-IDF値に変換された特徴量列と、
        'essay_id'列が含まれる。

        Attributes:
            train_data (polars.DataFrame): 学習データを含むDataFrame。
                                        'full_text'と'essay_id'列が必要。
            save_path (pathlib.Path): ベクトル化の重みの保存先パス。

        Returns:
            tuple: TF-IDFにより変換された特徴を含むデータフレーム。

        Notes:
        - n-gramの範囲は3から6まで、最小文書頻度は0.05、最大文書頻度は0.95です。
        """
        
        # TfidfVectorizer parameter
        vectorizer = TfidfVectorizer(
                    tokenizer=self.identity_function,
                    preprocessor=self.identity_function,
                    token_pattern=None,
                    strip_accents='unicode',
                    analyzer = 'word',
                    ngram_range=(3,6),
                    min_df=0.05,
                    max_df=0.95,
                    sublinear_tf=True,
        )

        # 学習データ(train_data)の処理
        train_tfid = vectorizer.fit_transform([i for i in train_data['full_text']])
        joblib.dump(vectorizer, save_path)
        dense_matrix = train_tfid.toarray()
        tr_df = pd.DataFrame(dense_matrix)
        tfid_columns = [ f'tfid_{i}' for i in range(len(tr_df.columns))]
        tr_df.columns = tfid_columns
        tr_df['essay_id'] = train_data['essay_id']

        return tr_df

    def transform_TfidfVec(self, test_data, save_path): 
        """
        この関数は、学習データでの処理をテストデータに対してもTF-IDFベクトル化を行い、
        結果をDataFrame形式で返す。各DataFrameには、テキストデータがTF-IDF値に変換された特徴量列と、
        'essay_id'列が含まれる。

        Attributes:
            test_data (polars.DataFrame): 学習データを含むDataFrame。
                                        'full_text'と'essay_id'列が必要。
            save_path (pathlib.Path): ベクトル化の重みの保存先パス。

        Returns:
            tuple: TF-IDFにより変換された特徴を含むデータフレーム。
        """        
        # テストデータ(test_data)の処理
        vectorizer = joblib.load(save_path)
        test_tfid = vectorizer.transform([i for i in test_data['full_text']])
        dense_matrix = test_tfid.toarray()
        te_df = pd.DataFrame(dense_matrix)
        tfid_columns = [ f'tfid_{i}' for i in range(len(te_df.columns))]
        te_df.columns = tfid_columns
        te_df['essay_id'] = test_data['essay_id']

        return  te_df

    def fit_transform_CountVec(self, train_data, save_path):
        """
        与えられたデータセットからカウントベクトルを生成し、特徴データフレームとして返します。

        Attributes:
            train_data (polars.DataFrame): 学習データを含むDataFrame。
                                        'full_text'と'essay_id'列が必要。
            save_path (pathlib.Path): ベクトル化の重みの保存先パス。

        Returns:
            DataFrame: カウントベクトルにより変換された特徴を含むデータフレーム。

        注意:
        - n-gramの範囲は2から3まで、最小文書頻度は0.10、最大文書頻度は0.85です。
        """

        vectorizer_cnt = CountVectorizer(
                    tokenizer=self.identity_function,
                    preprocessor=self.identity_function,
                    token_pattern=None,
                    strip_accents='unicode',
                    analyzer = 'word',
                    ngram_range=(2,3),
                    min_df=0.10,
                    max_df=0.85,
        )

        ## 学習データ(train_data)の処理
        train_tfid = vectorizer_cnt.fit_transform([i for i in train_data['full_text']])
        # joblib.dump(vectorizer_cnt, save_path)
        joblib.dump(vectorizer_cnt, save_path)
        dense_matrix = train_tfid.toarray()
        tr_df = pd.DataFrame(dense_matrix)
        tfid_columns = [ f'tfid_cnt_{i}' for i in range(len(tr_df.columns))]
        tr_df.columns = tfid_columns
        tr_df['essay_id'] = train_data['essay_id']

        return tr_df

    def transform_CountVec(self, test_data, save_path):
        """
        学習データでの処理をテストデータに対してもカウントベクトルを生成し、特徴データフレームとして返します。

        Attributes:
            train_data (polars.DataFrame): 学習データを含むDataFrame。
                                        'full_text'と'essay_id'列が必要。
            save_path (pathlib.Path): ベクトル化の重みの保存先パス。

        Returns:
            DataFrame: カウントベクトルにより変換された特徴を含むデータフレーム。

        注意:
        - n-gramの範囲は2から3まで、最小文書頻度は0.10、最大文書頻度は0.85です。
        """
        ## テストデータ(test_data)の処理
        vectorizer_cnt = joblib.load(save_path)
        test_tfid = vectorizer_cnt.transform([i for i in test_data['full_text']])
        dense_matrix = test_tfid.toarray()
        te_df = pd.DataFrame(dense_matrix)
        tfid_columns = [ f'tfid_cnt_{i}' for i in range(len(te_df.columns))]
        te_df.columns = tfid_columns
        te_df['essay_id'] = test_data['essay_id']

        return te_df

    def preprocessing_train(self) -> pd.DataFrame :
        """学習データ(train_data)に対して一連の処理を実行"""

        # データの呼び出し
        train_feats = self.train_data.to_pandas() # polars -> pandasへ変更

        # 保存先ディレクトリ用意
        vectorizer_weight_fold_dir = self.path_to.vectorizer_weight_dir / f"fold_{self.fold_num}/"
        vectorizer_weight_fold_dir.mkdir(parents=True, exist_ok=True)

        # TfidfVectorizer
        save_path = vectorizer_weight_fold_dir / 'vectorizer.pkl'
        tmp = self.fit_transform_TfidfVec(self.train_data, save_path)
        train_feats = train_feats.merge(tmp, on='essay_id', how='left')
        print('---TfidfVectorizer 特徴量作成完了---')

        # CountVectorizer
        save_path = vectorizer_weight_fold_dir / 'vectorizer_cnt.pkl'
        tmp = self.fit_transform_CountVec(self.train_data, save_path)
        train_feats = train_feats.merge(tmp, on='essay_id', how='left')
        print('---CountVectorizer 特徴量作成完了---')

        print('■ trainデータ作成完了')

        return train_feats
        
    def preprocessing_test(self) -> pd.DataFrame :
        """Vlaidデータ(test_data) or テストデータ(test_data)に対して一連の処理を実行"""

        # データの呼び出し
        test_feats = self.test_data.to_pandas() # polars -> pandasへ変更

        # 保存先パス設定
        vectorizer_weight_fold_dir = self.path_to.vectorizer_weight_dir / f"fold_{self.fold_num}/"

        # TfidfVectorizer
        save_path: Path = vectorizer_weight_fold_dir / 'vectorizer.pkl'
        tmp = self.transform_TfidfVec(self.test_data, save_path)
        test_feats = test_feats.merge(tmp, on='essay_id', how='left')
        print('---TfidfVectorizer 特徴量作成完了---')

        # CountVectorizer
        save_path = vectorizer_weight_fold_dir / 'vectorizer_cnt.pkl'
        tmp = self.transform_CountVec(self.test_data, save_path)
        test_feats = test_feats.merge(tmp, on='essay_id', how='left')
        print('---CountVectorizer 特徴量作成完了---')

        print('■ testデータ作成完了')

        return test_feats


