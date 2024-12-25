#!/usr/bin/env bash
# 參數設置
set -e
set -u
set -o pipefail

data_folder="/home/hsieh/espnet/egs2/nhk_radio/tts1/data/all_data_role"
output_folder="/home/hsieh/espnet/egs2/nhk_radio/tts1/data/all_data_role/cross_validation"
k=8  # 折數
random_seed=42

# 準備交叉驗證
python <<EOF
import os
from sklearn.model_selection import KFold

def split_dataset_by_article(data_folder, output_folder, k=5, random_seed=42):
    # 加載文件
    text_file = os.path.join(data_folder, "text")
    wav_scp_file = os.path.join(data_folder, "wav.scp")
    utt2spk_file = os.path.join(data_folder, "utt2spk")
    spk2utt_file = os.path.join(data_folder, "spk2utt")

    # 檢查文件是否存在
    if not all(os.path.exists(f) for f in [text_file, wav_scp_file, utt2spk_file, spk2utt_file]):
        raise FileNotFoundError("資料集缺少必要文件 (text, wav.scp, utt2spk, spk2utt)")

    # 加載內容
    with open(text_file, "r", encoding="utf-8") as f:
        text_lines = [line.strip() for line in f]
    with open(wav_scp_file, "r", encoding="utf-8") as f:
        wav_scp_lines = [line.strip() for line in f]
    with open(utt2spk_file, "r", encoding="utf-8") as f:
        utt2spk_lines = [line.strip() for line in f]
    with open(spk2utt_file, "r", encoding="utf-8") as f:
        spk2utt_lines = [line.strip() for line in f]

    # 提取所有 utt_id 和對應的 article_name
    def extract_article_name(utt_id):
        # 提取 <article_name>，即去掉最後兩部分
        return "_".join(utt_id.split("_")[1:-2])

    utt_ids = [line.split()[0] for line in text_lines]
    article_names = {extract_article_name(utt_id) for utt_id in utt_ids}  # 提取 article_name
    article_names = sorted(article_names)  # 保證順序一致

    # 檢查折數是否合法
    if len(article_names) < k:
        raise ValueError(f"折數 k={k} 大於文章數量 n={len(article_names)}。請降低折數或增加數據。")

    # 劃分 K 折
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(article_names)):
        fold_folder = os.path.join(output_folder, f"fold_{fold_idx+1}")
        os.makedirs(fold_folder, exist_ok=True)

        # 獲取訓練和驗證的 article_name
        train_articles = {article_names[i] for i in train_idx}
        test_articles = {article_names[i] for i in test_idx}

        # 根據 article_name 過濾 utt_id
        def filter_lines_by_article(lines, articles_set):
            return [line for line in lines if extract_article_name(line.split()[0]) in articles_set]

        # 根據 utt_id 過濾 spk2utt
        def filter_spk2utt(lines, utt_ids_set):
            filtered_lines = []
            for line in lines:
                parts = line.split()
                speaker = parts[0]
                utt_ids = set(parts[1:])
                filtered_utt_ids = utt_ids.intersection(utt_ids_set)
                if filtered_utt_ids:
                    filtered_lines.append(f"{speaker} {' '.join(filtered_utt_ids)}")
            return filtered_lines

        # 獲取訓練和測試的 utt_id
        train_utt_ids = set(line.split()[0] for line in filter_lines_by_article(text_lines, train_articles))
        test_utt_ids = set(line.split()[0] for line in filter_lines_by_article(text_lines, test_articles))

        # 保存到對應的文件夾
        for dataset, articles_set, utt_ids_set in [("train", train_articles, train_utt_ids), ("test", test_articles, test_utt_ids)]:
            dataset_folder = os.path.join(fold_folder, dataset)
            os.makedirs(dataset_folder, exist_ok=True)

            with open(os.path.join(dataset_folder, "text"), "w", encoding="utf-8") as f:
                f.write("\n".join(filter_lines_by_article(text_lines, articles_set)) + "\n")
            with open(os.path.join(dataset_folder, "wav.scp"), "w", encoding="utf-8") as f:
                f.write("\n".join(filter_lines_by_article(wav_scp_lines, articles_set)) + "\n")
            with open(os.path.join(dataset_folder, "utt2spk"), "w", encoding="utf-8") as f:
                f.write("\n".join(filter_lines_by_article(utt2spk_lines, articles_set)) + "\n")
            with open(os.path.join(dataset_folder, "spk2utt"), "w", encoding="utf-8") as f:
                f.write("\n".join(filter_spk2utt(spk2utt_lines, utt_ids_set)) + "\n")

        print(f"Fold {fold_idx+1} 已生成：{fold_folder}")

# 執行劃分
split_dataset_by_article("$data_folder", "$output_folder", $k, $random_seed)
EOF
