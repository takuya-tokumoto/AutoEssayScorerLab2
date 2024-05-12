# -*- encoding: utf-8 -*-

import yaml
import zipfile
from pathlib import Path
import shutil

# print(shutil.make_archive('zip_test', 'zip', root_dir='C:/PycharmProjects/Sample01/test'))

# def load_config():
#     with open('config.yaml', 'r') as file:
#         return yaml.safe_load(file)

# def zip_directory(source_dir, output_path):
#     source_dir = Path(source_dir)
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)  # 出力ディレクトリがなければ作成

#     # デバッグ: 処理するディレクトリの確認
#     print(f"Zipping files in directory: {source_dir}")

#     with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for file in source_dir.rglob('*'):
#             if file.is_file():
#                 # デバッグ: ファイルをアーカイブに追加
#                 print(f"Adding {file} to ZIP")
#                 zipf.write(file, arcname=file.relative_to(source_dir))
#     print(f"ZIP file created at {output_path}")

# if __name__ == '__main__':
#     config = load_config()
#     model_name = config["model_name"]
#     print("Script started")
#     root_dir = Path(__file__).resolve().parents[3]
#     s3_dir = root_dir / "s3storage/01_public/auto_essay_scorer_lab2/"

#     source_directory = s3_dir / f'models/{model_name}/'  # ソースディレクトリのパス
#     output_zip_file = s3_dir / f'data/bring_out/model_weight_{model_name}.zip' # 出力するZIPファイルのパス

#     zip_directory(source_directory, output_zip_file)