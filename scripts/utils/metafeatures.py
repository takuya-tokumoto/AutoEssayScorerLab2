#!/usr/bin/env python
# coding: utf-8

## Import
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import roc_auc_score
import polars as pl
import joblib
from pathlib import Path
from scipy.special import softmax
import os
from glob import glob

# 自作関数の読み込み
from .path import PathManager
from .cutoff_model import CutOffTrainer

class GenerateMetaFeatures():

    def __init__(self, 
                 repo_dir: Path,
                 config: dict, 
                 fold_num: str,
                 train_data: pl.DataFrame = None, 
                 test_data: pl.DataFrame = None, ):

        self.config = config
        self.path_to = PathManager(repo_dir, self.config["model_name"])
        self.fold_num = fold_num
        self.train_data = train_data
        self.test_data = test_data
        self.cut_point_list: list = [1.5, 2.5, 3.5, 4.5, 5.5]
        self.exclude_features: list = ['essay_id', 'score', 'full_text']
        self.model_params: dict = {
            'light_classify': {
                'n_jobs': 4, 
                'random_state': 42,
            },
            'cat_classify': {
                'thread_count': 4, 
                'random_seed': 42,
                'verbose': False,
                "logging_level": "Silent"  # ログを出力しないように設定
            }
        }

    def identity_function(self, x):
        
        return x

    def fit_transform_TfidfVec(self, train_data: pl.DataFrame, save_path: Path) -> pd.DataFrame:
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

    def transform_TfidfVec(self, test_data: pl.DataFrame, save_path: Path) -> pd.DataFrame: 
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

    def fit_transform_CountVec(self, train_data: pl.DataFrame, save_path: Path) -> pd.DataFrame:
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

    def transform_CountVec(self, test_data, save_path) -> pd.DataFrame:
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
   
    def prepare_cutoff_data(self, 
                            cut_point: float, 
                            input_data: pd.DataFrame,
                            feature_select: list,
                            train_option: bool) -> np.ndarray:
        """データを指定の変数で絞りこんだうえで学習データ(X, y, y_int)を作成"""

        ## 不要項目を排除
        # self.exclude_features: score, full_text, essey_idなど
        feature_select = [
            feature for feature in feature_select 
            if feature not in self.exclude_features
        ]
        # cut-off由来のスコア由来の項目(e.g. *_cut) 
        feature_select = [
            feature for feature in feature_select
            if not feature.endswith('_cut')
        ]

        ## 指定された項目を説明変数として取得
        X = input_data[feature_select].astype(np.float32).values

        ## 目標変数のカラムが存在する場合
        if train_option:
            # 目標変数をcut_pointで2値化
            y_int = (input_data[self.config['target']] > cut_point).astype(int).values
            return X, y_int

        return X
    
    def fit_transform_cutoff_model(self, train_data: pd.DataFrame, save_path: Path) -> pd.DataFrame:
        """スコアを区切り分類モデルを作成し予測スコアを特徴量に加える"""

        ## cut位置を指定しながら各閾値で分類モデル作成     
        tr_df = pd.DataFrame()  
        prev_cutoff = None
        for cut_point in self.cut_point_list:

            ## cut-off用のデータを準備 ※不要な特徴量削除
            feature_all = train_data.columns
            train_X, train_y = self.prepare_cutoff_data(cut_point, train_data, feature_all, True)

            ##学習
            trainer = CutOffTrainer(self.config, self.model_params)
            trainer.initialize_models()
            # イテレーション回数の最適化
            trainer.train_with_early_stopping(train_X, train_y)
            # 本番学習
            trainer.train(train_X, train_y)

            ## 学習結果を保存
            # ディレクトリ準備
            save_fold_path = save_path / f'cutoff_{str(cut_point).replace(".", "_")}/'
            save_fold_path.mkdir(parents=True, exist_ok=True)
            # 保存
            trainer.save_weight(save_fold_path)

            ## 予測スコア取得
            predicted_light, predicted_cat = trainer.predict(train_X)
            """テスト的に学習データに対する精度結果出力を後で記載"""

            ## 分類モデルのスコアを特徴量に付与
            tr_df[f'LBGM_{str(cut_point)}_cut'] = predicted_light
            tr_df[f'CB_{str(cut_point)}_cut'] = predicted_cat
            # LBGM, Catboostの予測差分
            tr_df[f'LGBM_CB_diff_{cut_point}_cut'] = predicted_light - predicted_cat
            # ひとつ前のcut_pointスコアとの差分
            if prev_cutoff is not None:
                tr_df[f'diff_LGBM_{str(cut_point)}_cut'] = (
                    tr_df[f'LBGM_{str(cut_point)}_cut'] 
                    - tr_df[f'LBGM_{str(prev_cutoff)}_cut']
                )
                tr_df[f'diff_CB_{str(cut_point)}_cut'] = (
                    tr_df[f'CB_{str(cut_point)}_cut'] 
                    - tr_df[f'CB_{str(prev_cutoff)}_cut']
                )
            # prev_cutoffを更新
            prev_cutoff = cut_point

            print(f'AUC score for LGBM {cut_point} model: {roc_auc_score(train_y, predicted_light)}')
            print(f'AUC score for CatBoost {cut_point} model: {roc_auc_score(train_y, predicted_cat)}')

        tr_df['essay_id'] = train_data['essay_id']

        return tr_df
    
    def transform_cutoff_model(self, test_data: pd.DataFrame, save_path: Path) -> pd.DataFrame:
        """学習済みモデルをロードし分類モデルの予測スコアを特徴量に加える"""

        ## cut位置を指定しながら各閾値で分類モデル作成     
        te_df = pd.DataFrame()  
        prev_cutoff = None
        for cut_point in self.cut_point_list:

            ## cut-off用のデータを準備 ※不要な特徴量削除
            feature_all = test_data.columns
            train_X = self.prepare_cutoff_data(cut_point, test_data, feature_all, False)

            ##学習
            trainer = CutOffTrainer(self.config, self.model_params)
            trainer.initialize_models()

            ## 学習済みモデルをロード
            # ディレクトリ準備
            save_fold_path = save_path / f'cutoff_{str(cut_point).replace(".", "_")}/'
            save_fold_path.mkdir(parents=True, exist_ok=True)
            # ロード
            trainer.load_weight(save_fold_path)

            ## 予測スコア取得
            predicted_light, predicted_cat = trainer.predict(train_X)
            """テスト的に学習データに対する精度結果出力"""

            ## 分類モデルのスコアを特徴量に付与
            te_df[f'LBGM_{str(cut_point)}_cut'] = predicted_light
            te_df[f'CB_{str(cut_point)}_cut'] = predicted_cat
            # LBGM, Catboostの予測差分
            te_df[f'LGBM_CB_diff_{cut_point}_cut'] = predicted_light - predicted_cat
            # ひとつ前のcut_pointスコアとの差分
            if prev_cutoff is not None:
                te_df[f'diff_LGBM_{str(cut_point)}_cut'] = (
                    te_df[f'LBGM_{str(cut_point)}_cut'] 
                    - te_df[f'LBGM_{str(prev_cutoff)}_cut']
                )
                te_df[f'diff_CB_{str(cut_point)}_cut'] = (
                    te_df[f'CB_{str(cut_point)}_cut'] 
                    - te_df[f'CB_{str(prev_cutoff)}_cut']
                )
            # prev_cutoffを更新
            prev_cutoff = cut_point

        te_df['essay_id'] = test_data['essay_id']

        return te_df
    

    def preprocessing_train(self) -> pd.DataFrame :
        """学習データ(train_data)に対して一連の処理を実行"""

        ## データの呼び出し
        train_feats = self.train_data.to_pandas() # polars -> pandasへ変更

        ## tf-idf 保存先ディレクトリ
        vectorizer_weight_fold_dir = self.path_to.vectorizer_weight_dir / f"fold_{self.fold_num}/"
        vectorizer_weight_fold_dir.mkdir(parents=True, exist_ok=True)

        ## TfidfVectorizer
        save_path = vectorizer_weight_fold_dir / 'vectorizer.pkl'
        tmp = self.fit_transform_TfidfVec(self.train_data, save_path)
        train_feats = train_feats.merge(tmp, on='essay_id', how='left')
        print('---TfidfVectorizer 特徴量作成完了---')

        ## CountVectorizer
        save_path = vectorizer_weight_fold_dir / 'vectorizer_cnt.pkl'
        tmp = self.fit_transform_CountVec(self.train_data, save_path)
        train_feats = train_feats.merge(tmp, on='essay_id', how='left')
        print('---CountVectorizer 特徴量作成完了---')

        # Cut-off model 保存先ディレクトリ
        cutoff_models_weight_fold_dir = self.path_to.cutoff_models_weight_dir / f"fold_{self.fold_num}/"
        cutoff_models_weight_fold_dir.mkdir(parents=True, exist_ok=True)

        ## Cut-off model
        tmp = self.fit_transform_cutoff_model(train_feats, cutoff_models_weight_fold_dir)
        train_feats = train_feats.merge(tmp, on='essay_id', how='left')
        print('---Cut-off model Score 特徴量作成完了---')        

        print('■ trainデータ作成完了')

        return train_feats
        
    def preprocessing_test(self) -> pd.DataFrame :
        """Vlaidデータ(test_data) or テストデータ(test_data)に対して一連の処理を実行"""

        ## データの呼び出し
        test_feats = self.test_data.to_pandas() # polars -> pandasへ変更

        ## tf-idf 保存先ディレクトリ
        vectorizer_weight_fold_dir = self.path_to.vectorizer_weight_dir / f"fold_{self.fold_num}/"

        ## TfidfVectorizer
        save_path: Path = vectorizer_weight_fold_dir / 'vectorizer.pkl'
        tmp = self.transform_TfidfVec(self.test_data, save_path)
        test_feats = test_feats.merge(tmp, on='essay_id', how='left')
        print('---TfidfVectorizer 特徴量作成完了---')

        ## CountVectorizer
        save_path = vectorizer_weight_fold_dir / 'vectorizer_cnt.pkl'
        tmp = self.transform_CountVec(self.test_data, save_path)
        test_feats = test_feats.merge(tmp, on='essay_id', how='left')
        print('---CountVectorizer 特徴量作成完了---')

        # Cut-off model 保存先ディレクトリ
        cutoff_models_weight_fold_dir = self.path_to.cutoff_models_weight_dir / f"fold_{self.fold_num}/"        

        ## Cut-off model
        tmp = self.transform_cutoff_model(test_feats, cutoff_models_weight_fold_dir)
        test_feats = test_feats.merge(tmp, on='essay_id', how='left')
        print('---Cut-off model Score 特徴量作成完了---')        


        print('■ testデータ作成完了')

        return test_feats


