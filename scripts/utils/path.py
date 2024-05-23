#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

class PathManager:
    """ディレクトリパスおよびファイルパスを保持する。"""

    def __init__(self, repo_dir: Path, mode: str) -> None:

        #### input/
        self.input_dir: Path = repo_dir / "input/"
        # オリジナルのコンペデータ
        self.origin_train_dir: Path = (
            self.input_dir / "learning-agency-lab-automated-essay-scoring-2/train.csv"
        )
        self.origin_test_dir: Path = (
            self.input_dir / "learning-agency-lab-automated-essay-scoring-2/test.csv"
        )
        self.origin_sample_submit_dir: Path = (
            self.input_dir / "learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
        )
        # english-word-hx
        self.english_word_hx_dir: Path = (
            self.input_dir / "english-word-hx/words.txt"
        )       
        # aes2-preprocessing
        self.aes2_cache_dir: Path = (
            self.input_dir / "aes2-preprocessing/feature_select.pickle"
        )    
        # aes2-400-20240419134941/*/*
        self.pretrain_deberta_model_dir: Path = (
            self.input_dir / "aes2-400-20240419134941/*/*"
        )            
        # aes2-400-20240419134941/oof.pkl
        self.deberta_model_oof_dir: Path = (
            self.input_dir / "aes2-400-20240419134941/oof.pkl"
        )     
        ### input/train_logs_{mode}/
        self.train_logs_dir: Path = self.input_dir / f"train-logs-{mode}/"
        ## input/train_logs_{mode}/vectorize_weight/
        self.vectorizer_weight_dir: Path = self.train_logs_dir / f"vectorizer-weight/"
        ## カットオフモデルの重み：input/train_logs_{mode}/cutoff-model-weight/
        self.cutoff_models_weight_dir: Path = self.train_logs_dir / f"cutoff-model-weight/"
        ## アンサンブルモデルの重み：input/train_logs_{mode}/ens_model_weight/
        self.models_weight_dir: Path = self.train_logs_dir / "ens-model-weight/"
        ## ログファイル：input/train_logs_{mode}/ens_model_weight/training.log
        self.log_file_dir: Path = self.models_weight_dir / "training.log"

        ### input/middle_mart_{mode}/
        self.middle_mart_dir: Path = self.input_dir / f"middle-mart-{mode}/"
        ## input/middle_mart_{mode}/stratifiedkfold/
        self.skf_mart_dir: Path = self.middle_mart_dir / f"stratifiedkfold/"
        ## input/middle_mart_{mode}/add_metafeats/
        self.add_meta_mart_dir: Path = self.middle_mart_dir / f"add_metafeats/"
        # 特徴量付きのデータフレーム
        self.train_all_mart_dir: Path = (
            self.middle_mart_dir / "train_all.csv"
        )
        self.test_all_mart_dir: Path = (
            self.middle_mart_dir / "test_all.csv"
        )

        #### output/
        self.output_dir: Path = repo_dir / "output/"
        ### submitファイル
        self.submit_dir: Path = self.output_dir / "submit.csv"



        