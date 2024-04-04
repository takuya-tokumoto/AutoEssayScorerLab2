#!/bin/bash

# 解凍先のディレクトリを指定
unzip_destination="/kaggle/s3storage/01_public/2024_pii_data_detection/02_unzip_file"

# 解凍対象のディレクトリを指定
zip_destination="/kaggle/s3storage/01_public/2024_pii_data_detection/01_row"

# 解凍対象のファイルリスト
zipfiles=("pii-detection-removal-from-educational-data.zip" "pii-mixtral8x7b-generated-essays.zip")

# 各zipファイルに対して処理を行う
for zipfile in "${zipfiles[@]}"; do
    # ファイル名から拡張子を除いた名前を取得
    dirname="${zipfile%.zip}"

    # 解凍先のディレクトリ内に、ファイル名に基づいたディレクトリを作成
    mkdir -p "$unzip_destination/$dirname"

    # zipファイルを新しいディレクトリに解凍
    unzip -o "$zip_destination/$zipfile" -d "$unzip_destination/$dirname"
done