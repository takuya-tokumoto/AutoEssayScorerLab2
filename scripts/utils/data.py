#!/usr/bin/env python
# coding: utf-8

## Import
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import spacy
import string
import polars as pl
import torch
import joblib
from pathlib import Path
from scipy.special import softmax
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from datasets import Dataset
from glob import glob
# 自作関数の読み込み
from .path import PathManager

class CreateDataset():

    def __init__(self, repo_dir: Path, config: dict):
        self.train_data = pl.DataFrame()
        self.test_data = pl.DataFrame()
        self.config = config
        self.path_to = PathManager(repo_dir, config["model_name"])
        self.nlp = spacy.load("en_core_web_sm")
        self.english_vocab = self.load_english_vocab() 

    def load_data(self, path):
        """Read用関数"""

        columns = [
            pl.col("full_text").str.split(by="\n\n").alias("paragraph")
        ]
        return pl.read_csv(path).with_columns(columns)

    def load_dataset(self):
        """データ読み込み"""
        
        self.train_data = self.load_data(self.path_to.origin_train_dir)
        self.test_data = self.load_data(self.path_to.origin_test_dir)

        if self.config['sampling_mode']:
            self.train_data = self.train_data.sample(n=100, with_replacement=False)

    def load_english_vocab(self):
        """英語語彙セット(english-word-hx)を読み込み"""

        vocab_path = self.path_to.english_word_hx_dir
        with open(vocab_path, 'r') as file:
            return set(word.strip().lower() for word in file)
    
    def count_spelling_errors(self, text):
        """与えられたテキスト内(text)のスペルミスの数をカウント"""
            
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_.lower() for token in doc]
        spelling_errors = sum(1 for token in lemmatized_tokens if token not in self.english_vocab)
        return spelling_errors

    def removeHTML(self, x):
        """html記号を排除"""

        html=re.compile(r'<.*?>')
        return html.sub(r'',x)

    def dataPreprocessing(self, x):
        """
        与えられたテキストから不要な要素を除去し、フォーマットを整えることでデータの前処理を行う。

        Args:
            x (str): 前処理を行う生のテキストデータ。

        Returns:
            str: HTMLタグ、メンション、数値、URLが除去され、不要な空白や句読点が整理されたテキスト。

        処理内容:
        - テキストを全て小文字に変換。
        - HTMLタグを削除。
        - '@'で始まるメンションを削除。
        - 数値を削除。
        - URLを削除。
        - 連続する空白を一つの空白に置き換え。
        - 連続するコンマとピリオドをそれぞれ一つに置き換え。
        - 文字列の先頭と末尾の空白を削除。
        """

        # Convert words to lowercase
        x = x.lower()
        # Remove HTML
        x = self.removeHTML(x)
        # Delete strings starting with @
        x = re.sub("@\w+", '',x)
        # Delete Numbers
        x = re.sub("'\d+", '',x)
        x = re.sub("\d+", '',x)
        # Delete URL
        x = re.sub("http\w+", '',x)
        # Replace consecutive empty spaces with a single space character
        x = re.sub(r"\s+", " ", x)
        # Replace consecutive commas and periods with one comma and period character
        x = re.sub(r"\.+", ".", x)
        x = re.sub(r"\,+", ",", x)
        # Remove empty characters at the beginning and end
        x = x.strip()
        return x

    # paragraph features
    def remove_punctuation(self, text):
        """
        入力テキストから句読点をすべて取り除く。
        
        Args:
            text (str): 前処理を行う生のテキストデータ。
        
        Returns:
            str: 句読点を取り除いた文章。
        """

        # string.punctuation
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def Paragraph_Preprocess(self, tmp):
        """
        段落データを複数の前処理ステップに通し、段落ごとにさまざまな特徴を計算。

        Args:
            tmp (polars.DataFrame): 'paragraph' という列を含む Polars DataFrame。
            各エントリはテキストの段落を表します。

        Returns:
            polars.DataFrame: 前処理後のデータと各段落に関する追加情報を含む DataFrame。

        処理内容:
        - 段落リストを複数行のデータに展開。
        - 段落の前処理を実施。
        - 句読点の除去。
        - スペルミスの数をカウント。
        - 各段落の長さを計算。
        - 各段落の文と単語の数を計算。
        """

        # Expand the paragraph list into several lines of data
        tmp = tmp.explode('paragraph')
        # Paragraph preprocessing
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(self.dataPreprocessing))
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(self.remove_punctuation).alias('paragraph_no_pinctuation'))
        tmp = tmp.with_columns(pl.col('paragraph_no_pinctuation').map_elements(self.count_spelling_errors).alias("paragraph_error_num"))
        # Calculate the length of each paragraph
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x)).alias("paragraph_len"))
        # Calculate the number of sentences and words in each paragraph
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x.split('.'))).alias("paragraph_sentence_cnt"),
                        pl.col('paragraph').map_elements(lambda x: len(x.split(' '))).alias("paragraph_word_cnt"),)
        return tmp

    def Paragraph_Eng(self, train_tmp):
        """
        与えられたデータフレームに対して、段落の長さに基づく統計量を集計し、エッセイごとに結果をまとめる。

        Args:
            train_tmp (polars.DataFrame): 'paragraph' と 'paragraph_len' 列を含む Polars DataFrame。
            'essay_id' によって各エッセイが識別される。

        Returns:
            pandas.DataFrame: 各エッセイIDごとに集計された統計量を含む DataFrame。

        処理内容:
        - 段落の長さが特定の閾値リストに含まれるかどうかに基づいてカウントを集計。
        - さまざまな閾値での段落の長さの最大値、平均値、最小値、合計、最初と最後の値、尖度、第1四分位数、第3四分位数を計算。
        - 結果を 'essay_id' でグループ化し、順序を保持して集計。
        """

        paragraph_fea = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt']
        paragraph_fea2 = ['paragraph_error_num'] + paragraph_fea

        aggs = [
            # Count the number of paragraph lengths greater than and less than the i-value
            *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_>{i}_cnt") for i in [0, 50,75,100,125,150,175,200,250,300,350,400,500,600,700] ], 
            *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_<{i}_cnt") for i in [25,49]], 
            # other
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea2],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea2],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea2],
            *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in paragraph_fea2],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea2],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea2],
            *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in paragraph_fea2],
            *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in paragraph_fea2],  
            *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in paragraph_fea2],  
            ]
        
        df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()

        return df

    def Sentence_Preprocess(self, tmp):
        """
        テキストデータを文単位に前処理し、各文の長さと単語数を計算。

        Args:
            tmp (polars.DataFrame): 'full_text' 列を含む Polars DataFrame。
                                    この列には前処理を行うテキストデータが格納されています。

        Returns:
            polars.DataFrame: 各文の長さと単語数が計算された後の DataFrame。
                            文の長さが15文字以上のデータのみが含まれます。

        処理内容:
        - 'full_text' 列のデータを前処理し、ピリオドで文を分割します。
        - 分割された文を新しい行として展開します。
        - 各文の長さを計算し、15文字未満の文をフィルタリングします。
        - 各文の単語数を計算します。
        """

        # Preprocess full_text and use periods to segment sentences in the text
        tmp = tmp.with_columns(pl.col('full_text').map_elements(self.dataPreprocessing).str.split(by=".").alias("sentence"))
        tmp = tmp.explode('sentence')
        # Calculate the length of a sentence
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x)).alias("sentence_len"))
        # Filter out the portion of data with a sentence length greater than 15
        tmp = tmp.filter(pl.col('sentence_len')>=15)
        # Count the number of words in each sentence
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x.split(' '))).alias("sentence_word_cnt"))
        
        return tmp

    def Sentence_Eng(self, train_tmp):
        """
        データフレーム内の文に関する特定の特性に基づいて統計量を集計し、エッセイIDごとに結果をまとめる。

        Args:
            train_tmp (polars.DataFrame): 'sentence' と 'sentence_len' 列を含む Polars DataFrame。
                                        さらに 'essay_id' 列を用いてエッセイを識別します。

        Returns:
            pandas.DataFrame: エッセイIDごとに集計された各種統計量を含む DataFrame。

        処理内容:
        - 文の長さが特定の閾値以上、または以下である文の数をカウント。
        - 各文特性（sentence_fea）について最大値、平均値、最小値、合計値、最初の値、最後の値、尖度、第1四分位数、第3四分位数を計算。
        - 結果を 'essay_id' でグループ化し、順序を保持して集計し、最終的にPandasデータフレームに変換。
        """

        # feature_eng
        sentence_fea = ['sentence_len','sentence_word_cnt']

        aggs = [
            # Count the number of sentences with a length greater than i
            *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_>{i}_cnt") for i in [0,15,50,100,150,200,250,300] ], 
            *[pl.col('sentence').filter(pl.col('sentence_len') <= i).count().alias(f"sentence_<{i}_cnt") for i in [15,50] ], 
            # other
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
            *[pl.col(fea).sum().alias(f"{fea}_sum") for fea in sentence_fea],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
            *[pl.col(fea).kurtosis().alias(f"{fea}_kurtosis") for fea in sentence_fea],
            *[pl.col(fea).quantile(0.25).alias(f"{fea}_q1") for fea in sentence_fea], 
            *[pl.col(fea).quantile(0.75).alias(f"{fea}_q3") for fea in sentence_fea], 
            ]
        df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()

        return df

    def Word_Preprocess(self, tmp):
        """
        テキストデータを単語単位に前処理し、各単語の長さを計算した後、長さが0の単語を除外します。

        Args:
            tmp (polars.DataFrame): 'full_text' 列を含む Polars DataFrame。
                                この列には前処理を行うテキストデータが格納されています。

        Returns:
            polars.DataFrame: 単語とその長さを含む DataFrame。長さが0の単語は除外されます。

        処理内容:
        - 'full_text' 列のデータを前処理し、空白で単語を分割します。
        - 分割された単語を新しい行として展開します。
        - 各単語の長さを計算し、長さが0の単語をデータセットから除外します。
        """
        # Preprocess full_text and use spaces to separate words from the text
        tmp = tmp.with_columns(pl.col('full_text').map_elements(self.dataPreprocessing).str.split(by=" ").alias("word"))
        tmp = tmp.explode('word')
        # Calculate the length of each word
        tmp = tmp.with_columns(pl.col('word').map_elements(lambda x: len(x)).alias("word_len"))
        # Delete data with a word length of 0
        tmp = tmp.filter(pl.col('word_len')!=0)
        
        return tmp

    def Word_Eng(self, train_tmp):
        """
        テキストデータ内の単語の長さに関する統計量を集計し、エッセイIDごとに結果をまとめる。

        Args:
            train_tmp (polars.DataFrame): 'word' および 'word_len' 列を含む Polars DataFrame。 
                                        各単語は 'word' 列に、その長さは 'word_len' 列に格納されています。

        Returns:
            pandas.DataFrame: 各エッセイIDごとに集計された単語の長さに関する統計量を含む DataFrame。

        処理内容:
        - 単語の長さが特定の閾値以上である単語の数をカウント（1文字以上から15文字以上まで）。
        - 単語の長さに関する最大値、平均値、標準偏差、第1四分位数、中央値、第3四分位数を計算。
        - 結果を 'essay_id' でグループ化し、順序を保持して集計し、最終的にPandasデータフレームに変換。
        """

        aggs = [
            # Count the number of words with a length greater than i+1
            *[pl.col('word').filter(pl.col('word_len') >= i+1).count().alias(f"word_{i+1}_cnt") for i in range(15) ], 
            # other
            pl.col('word_len').max().alias(f"word_len_max"),
            pl.col('word_len').mean().alias(f"word_len_mean"),
            pl.col('word_len').std().alias(f"word_len_std"),
            pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
            pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
            pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
            ]
        df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()
        return df
    
    def load_deberta_preds_feats(self):
        """事前に作成済みのdebertaの予測値を読み込み"""

        deberta_oof = joblib.load(self.path_to.deberta_model_oof_dir)

        return deberta_oof
    
    def deberta_oof_scores(self, input_data):
        """学習済みモデルを用いて予測スコアを計算する"""

        def load_tokenizer_and_models():
            """トークナイザーとモデルを読み込む"""

            models = glob(str(self.path_to.pretrain_deberta_model_dir))
            tokenizer = AutoTokenizer.from_pretrained(models[0])
            return tokenizer, models
        
        def tokenize_data(tokenizer, input_data):
            """入力データをトークン化"""

            def tokenize(sample):
                return tokenizer(sample['full_text'], max_length=self.config["MAX_LENGTH"], truncation=True)

            ds = Dataset.from_pandas(input_data.to_pandas())
            ds = ds.map(tokenize).remove_columns(['essay_id', 'full_text'])

            return ds
        
        def predict_scores(models, tokenizer, ds):
            """予測スコアを計算"""

            args = TrainingArguments(
                ".", 
                per_device_eval_batch_size=self.config["EVAL_BATCH_SIZE"], 
                report_to="none"
            )
            
            predictions = []
            for model_path in models:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                trainer = Trainer(
                    model=model, 
                    args=args, 
                    data_collator=DataCollatorWithPadding(tokenizer), 
                    tokenizer=tokenizer
                )
                preds = trainer.predict(ds).predictions
                predictions.append(softmax(preds, axis=-1))
                del model, trainer
                torch.cuda.empty_cache()
                gc.collect()

            predicted_score = 0.
            for p in predictions:
                predicted_score += p    
            predicted_score /= len(predictions)

            return predicted_score
        
        tokenizer, models = load_tokenizer_and_models()
        ds = tokenize_data(tokenizer, input_data)
        predicted_score = predict_scores(models, tokenizer, ds)

        return predicted_score

    def preprocessing_train(self):
        """学習データ(train_data)に対して一連の処理を実行"""

        self.load_dataset()
        
        # Paragraph
        tmp = self.Paragraph_Preprocess(self.train_data)
        train_feats = self.Paragraph_Eng(tmp)
        print('---Paragraph 特徴量作成完了---')

        # Score
        train_feats['score'] = self.train_data['score']
        # full_text -> 後の処理で必要なので付与 ※決定木モデルには直接投入しないよう注意
        train_feats['full_text'] = self.train_data['full_text']

        # Sentence
        tmp = self.Sentence_Preprocess(self.train_data)
        train_feats = train_feats.merge(self.Sentence_Eng(tmp), on='essay_id', how='left')
        print('---Sentence 特徴量作成完了---')

        # Word
        tmp = self.Word_Preprocess(self.train_data)
        train_feats = train_feats.merge(self.Word_Eng(tmp), on='essay_id', how='left') 
        print('---Word 特徴量作成完了---')

        # Debertaモデルの予測値
        predicted_score = self.load_deberta_preds_feats()
        # predicted_score = self.deberta_oof_scores(self.train_data)
        for i in range(6):
            train_feats[f'deberta_oof_{i}'] = predicted_score[:, i]
        print('---Debertaモデル予測値 特徴量作成完了---')

        print('■ trainデータ作成完了')

        return train_feats

    def preprocessing_test(self):
        """テストデータ(train_data)に対して一連の処理を実行"""

        self.load_dataset()

        # Paragraph
        tmp = self.Paragraph_Preprocess(self.test_data)
        test_feats = self.Paragraph_Eng(tmp)
        print('---Paragraph 特徴量作成完了---')

        # full_text -> 後の処理で必要なので付与 ※決定木モデルには直接投入しないよう注意
        test_feats['full_text'] = self.test_data['full_text']

        # Sentence
        tmp = self.Sentence_Preprocess(self.test_data)
        test_feats = test_feats.merge(self.Sentence_Eng(tmp), on='essay_id', how='left')
        print('---Sentence 特徴量作成完了---')

        # Word
        tmp = self.Word_Preprocess(self.test_data)
        test_feats = test_feats.merge(self.Word_Eng(tmp), on='essay_id', how='left')
        print('---Word 特徴量作成完了---')

        # Debertaモデルの予測値
        predicted_score = self.deberta_oof_scores(self.test_data)
        for i in range(6):
            test_feats[f'deberta_oof_{i}'] = predicted_score[:, i]
        print('---Debertaモデル予測値 特徴量作成完了---')

        print('■ testデータ作成完了')

        return test_feats

        