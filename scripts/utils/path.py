from pathlib import Path
import yaml


class PathManager:
    """ディレクトリパスおよびファイルパスを保持する。"""

    def __init__(self, repo_dir: Path, mode: str) -> None:
        
        ## input/
        self.input_dir: Path = repo_dir / "data/input/"
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
        # aes2-cache
        self.aes2_cache_dir: Path = (
            self.input_dir / "aes2-cache/feature_select.pickle"
        )    

        # input/middle/
        self.mid_dir: Path = self.input_dir / "middle/"
        # input/middle/{mode}/
        self.middle_files_dir: Path = self.input_dir / "middle/" f"{mode}/"
        # TfidfVectorizerの重み
        self.vectorizer_fit_dir: Path = self.middle_files_dir / 'vectorizer.pkl'
        # CountVectorizerの重み
        self.cnt_vectorizer_fit_dir: Path = self.middle_files_dir / 'vectorizer_cnt.pkl'
        # 特徴量付きのデータフレーム
        self.train_all_mart_dir: Path = (
            self.middle_files_dir / "train_all.csv"
        )
        self.test_all_mart_dir: Path = (
            self.middle_files_dir / "test_all.csv"
        )

        ## models/{mode}/
        # モデルの重み
        self.models_weight: Path = repo_dir / "models/" f"{mode}/"

        ## data/output/
        self.output_dir: Path = repo_dir / "data/output/"
        # submitファイル
        self.submit_dir: Path = self.output_dir / "submit.csv"



        