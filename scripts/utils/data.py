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
import textstat
from spellchecker import SpellChecker
from collections import Counter
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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


    def add_textstat_features(self, df):

        def textstat_features(text):
            features = {}
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            features['smog_index'] = textstat.smog_index(text)
            features['coleman_liau_index'] = textstat.coleman_liau_index(text)
            features['automated_readability_index'] = textstat.automated_readability_index(text)
            features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
            features['difficult_words'] = textstat.difficult_words(text)
            features['linsear_write_formula'] = textstat.linsear_write_formula(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
            features['text_standard'] = textstat.text_standard(text, float_output=True)
            features['spache_readability'] = textstat.spache_readability(text)
            features['mcalpine_eflaw'] = textstat.mcalpine_eflaw(text)
            features['reading_time'] = textstat.reading_time(text)
            features['syllable_count'] = textstat.syllable_count(text)
            features['lexicon_count'] = textstat.lexicon_count(text)
            features['monosyllabcount'] = textstat.monosyllabcount(text)

            return features
        
        df['textstat_features'] = df['full_text'].apply(textstat_features)
        textstat_features_df = pd.DataFrame(df['textstat_features'].tolist())
        return textstat_features_df


    def add_extract_linguistic_features(self, df):
        def extract_linguistic_features(text):
            doc = self.nlp(text)
            features = {}

                # NER Features
            entity_counts = {"GPE": 0, "PERCENT": 0, "NORP": 0, "ORG": 0, "CARDINAL": 0, "MONEY": 0, "DATE": 0, 
                            "LOC": 0, "PERSON": 0, "QUANTITY": 0, "EVENT": 0, "ORDINAL": 0, "WORK_OF_ART": 0, 
                            "LAW": 0, "PRODUCT": 0, "TIME": 0, "FAC": 0, "LANGUAGE": 0}
            for entity in doc.ents:
                if entity.label_ in entity_counts:
                    entity_counts[entity.label_] += 1
            features['NER_Features'] = entity_counts

            # POS Features
            pos_counts = {"ADJ": 0, "NOUN": 0, "VERB": 0, "SCONJ": 0, "PRON": 0, "PUNCT": 0, "DET": 0, "AUX": 0, 
                        "PART": 0, "ADP": 0, "SPACE": 0, "CCONJ": 0, "PROPN": 0, "NUM": 0, "ADV": 0, 
                        "SYM": 0, "INTJ": 0, "X": 0}
            for token in doc:
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
            features['POS_Features'] = pos_counts

            # tag Features
            tags = {"RB": 0, "-RRB-": 0, "PRP$": 0, "JJ": 0, "TO": 0, "VBP": 0, "JJS": 0, "DT": 0, "''": 0, "UH": 0, "RBS": 0, "WRB": 0, ".": 0, 
                "HYPH": 0, "XX": 0, "``": 0, "SYM": 0, "VB": 0, "VBN": 0, "WP": 0, "CC": 0, "LS": 0, "POS": 0, "NN": 0, ",": 0, "NNPS": 0,
                "RP": 0, ":": 0, "$": 0, "PDT": 0, "VBZ": 0, "VBD": 0, "JJR": 0, "-LRB-": 0, "IN": 0, "RBR": 0, "WDT": 0, "EX": 0, "MD": 0,
                    "_SP": 0, "NNP": 0, "CD": 0, "VBG": 0, "NNS": 0, "PRP": 0}
            
            for token in doc:
                if token.tag_ in tags:
                    tags[token.tag_] += 1
            features['tag_Features'] = tags

            # tense features
            tenses = [i.morph.get("Tense") for i in doc]
            tenses = [i[0] for i in tenses if i]
            tense_counts = Counter(tenses)
            features['past_tense_ratio'] = tense_counts.get("Past", 0) / (tense_counts.get("Pres", 0) + tense_counts.get("Past", 0) + 1e-5)
            features['present_tense_ratio'] = tense_counts.get("Pres", 0) / (tense_counts.get("Pres", 0) + tense_counts.get("Past", 0) + 1e-5)
            
            
            # len features

            features['word_count'] = len(doc)
            features['sentence_count'] = len([sentence for sentence in doc.sents])
            features['words_per_sentence'] = features['word_count'] / features['sentence_count']
            features['std_words_per_sentence'] = np.std([len(sentence) for sentence in doc.sents])

            features['unique_words'] = len(set([token.text for token in doc]))
            features['lexical_diversity'] = features['unique_words'] / features['word_count']

            paragraph = text.split('\n\n')

            features['paragraph_count'] = len(paragraph)

            features['avg_chars_by_paragraph'] = np.mean([len(paragraph) for paragraph in paragraph])
            features['avg_words_by_paragraph'] = np.mean([len(nltk.word_tokenize(paragraph)) for paragraph in paragraph])
            features['avg_sentences_by_paragraph'] = np.mean([len(nltk.sent_tokenize(paragraph)) for paragraph in paragraph]) 

            # sentiment features
            analyzer = SentimentIntensityAnalyzer()
            sentences = nltk.sent_tokenize(text)

            compound_scores, negative_scores, positive_scores, neutral_scores = [], [], [], []
            for sentence in sentences:
                scores = analyzer.polarity_scores(sentence)
                compound_scores.append(scores['compound'])
                negative_scores.append(scores['neg'])
                positive_scores.append(scores['pos'])
                neutral_scores.append(scores['neu'])

            features["mean_compound"] = np.mean(compound_scores)
            features["mean_negative"] = np.mean(negative_scores)
            features["mean_positive"] = np.mean(positive_scores)
            features["mean_neutral"] = np.mean(neutral_scores)

            features["std_compound"] = np.std(compound_scores)
            features["std_negative"] = np.std(negative_scores)
            features["std_positive"] = np.std(positive_scores)
            features["std_neutral"] = np.std(neutral_scores)

            return features
        
        df['linguistic_features'] = df['full_text'].apply(extract_linguistic_features)
        df_linguistic = pd.json_normalize(df['linguistic_features'])
        
        return df_linguistic

    # ratio列を計算し、train_linguisticとtest_linguisticに追加する関数
    def add_ratio_columns(self, df):
        tag_cols = [col for col in df.columns if col.startswith('tag')]
        col_cols = [col for col in df.columns if col.startswith('col')]
        pos_cols = [col for col in df.columns if col.startswith('pos')]

        for col in tag_cols:
            df[f"{col}_ratio"] = df[col] / df['word_count']

        for col in col_cols:
            df[f"{col}_ratio"] = df[col] / df['word_count']

        for col in pos_cols:
            df[f"{col}_ratio"] = df[col] / df['word_count']
        return df

    def add_spell_check(self, df):
        def spell_check(text):
            spell = SpellChecker()
            words = nltk.word_tokenize(text)
            misspelled = spell.unknown(words)
            misspelled_count = len(misspelled)
            misspelled_ratio = misspelled_count / len(words)
            return misspelled_count, misspelled_ratio

        spell_check_results = df['full_text'].apply(spell_check)
        spell_check_df = pd.DataFrame(spell_check_results.to_list(), columns=['misspelled_count', 'misspelled_ratio'])
        return spell_check_df

    def readablity_features(self, df):
        df = df.to_pandas()
        textstat_features_df = self.add_textstat_features(df)
        linguistic_features_df = self.add_extract_linguistic_features(df)
        linguistic_features_df = self.add_ratio_columns(linguistic_features_df)
        spell_check_df = self.add_spell_check(df)

        return textstat_features_df, linguistic_features_df, spell_check_df
        
    
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

        # Readablity
        train_textstat_df, train_linguistic_df, train_spell_check_df = self.readablity_features(self.train_data)
        train_feats = (
            train_feats.merge(train_textstat_df, left_index=True, right_index=True)
            .merge(train_linguistic_df, left_index=True, right_index=True)
            .merge(train_spell_check_df, left_index=True, right_index=True)
        )
        print('---Readability 特徴量作成完了---')

        # # Debertaモデルの予測値
        # predicted_score = self.load_deberta_preds_feats()
        # # predicted_score = self.deberta_oof_scores(self.train_data)
        # for i in range(6):
        #     train_feats[f'deberta_oof_{i}'] = predicted_score[:, i]
        # print('---Debertaモデル予測値 特徴量作成完了---')

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

        # Readablity
        test_textstat_df, test_linguistic_df, test_spell_check_df = self.readablity_features(self.test_data)
        test_feats = (
            test_feats.merge(test_textstat_df, left_index=True, right_index=True)
            .merge(test_linguistic_df, left_index=True, right_index=True)
            .merge(test_spell_check_df, left_index=True, right_index=True)
        )
        print('---Readability 特徴量作成完了---')

        # # Debertaモデルの予測値
        # predicted_score = self.deberta_oof_scores(self.test_data)
        # for i in range(6):
        #     test_feats[f'deberta_oof_{i}'] = predicted_score[:, i]
        # print('---Debertaモデル予測値 特徴量作成完了---')

        print('■ testデータ作成完了')

        return test_feats

        