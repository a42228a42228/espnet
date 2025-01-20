#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# 通用變數定義
fs=44100
n_fft=2048
n_shift=512
win_length=null
# g2p=pyopenjtalk_prosody

sp_tokens=("<e>" "</e>")
g2p=pyopenjtalk_prosody_sp_token

opts=
if [ "${fs}" -eq 48000 ]; then
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_config_default=conf/tuning/train_vits_44k.yaml
train_config_role=conf/tuning/train_vits_44k_role.yaml
inference_config=conf/tuning/decode_vits.yaml
common_args=(
    --ngpu 4
    --lang jp
    --feats_type raw
    --min_wav_duration 0.38
    --fs "${fs}"
    --n_fft "${n_fft}"
    --n_shift "${n_shift}"
    --win_length "${win_length}"
    --token_type phn
    --cleaner jaconv
    --g2p ${g2p}
    --tts_task gan_tts
    --feats_extract linear_spectrogram
    --feats_normalize none
    --inference_config "${inference_config}"
    --inference_model "train.total_count.ave_10best.pth"
    ${opts}
)

# 定義訓練和測試函數
run_fold() {
    local fold=$1
    local train_set=$2
    local test_set=$3
    local tag_prefix=$4

    echo "正在處理 ${tag_prefix} 的 Fold ${fold}..."

    # 根據 tag_prefix 判斷 train_config 和額外選項
    local train_config=""
    local additional_opts=()
    if [ "${tag_prefix}" == "role" ]; then
        train_config="${train_config_role}"
        additional_opts+=(--use_sid true)
    else
        train_config="${train_config_default}"
    fi

    ./tts.sh \
        "${common_args[@]}" \
        --train_config "${train_config}" \
        --stage 2 \
        --stop_stage 5 \
        --train_set "${train_set}" \
        --valid_set "${test_set}" \
        --test_sets "${test_set}" \
        --srctexts "data/${train_set}/text" \
        --train_args "--init_param downloads/7e515fa2fd318b964d69b992a1bee634/exp/tts_train_full_band_vits_raw_phn_jaconv_pyopenjtalk_prosody/train.total_count.ave_10best.pth:tts:tts:tts.generator.text_encoder.emb --freeze_param tts.generator.text_encoder.encoder tts.generator.decoder tts.decoder" \
        --tag "train_text_emb_${tag_prefix}_fold_${fold}_finetune_vits_raw_phn_jaconv_${g2p}" \
        "${additional_opts[@]}"

    # 更新 token_list
    TOKEN_LIST_DIR="dump/token_list/phn_jaconv_${g2p}"
    pyscripts/utils/make_token_list_from_config.py downloads/7e515fa2fd318b964d69b992a1bee634/exp/tts_train_full_band_vits_raw_phn_jaconv_pyopenjtalk_prosody/config.yaml
    mv ${TOKEN_LIST_DIR}/tokens.{txt,txt.bak}
    ln -s $(pwd)/downloads/7e515fa2fd318b964d69b992a1bee634/exp/tts_train_full_band_vits_raw_phn_jaconv_pyopenjtalk_prosody/tokens.txt ${TOKEN_LIST_DIR}/tokens.txt
    TOKEN_LIST=${TOKEN_LIST_DIR}/tokens.txt
    for token in "${sp_tokens[@]}"; do
        echo "$token" >> "$TOKEN_LIST"
    done

    ./tts.sh \
        "${common_args[@]}" \
        --train_config "${train_config}" \
        --stage 7 \
        --stop_stage 8 \
        --train_set "${train_set}" \
        --valid_set "${test_set}" \
        --test_sets "${test_set}" \
        --srctexts "data/${train_set}/text" \
        --train_args "--init_param downloads/7e515fa2fd318b964d69b992a1bee634/exp/tts_train_full_band_vits_raw_phn_jaconv_pyopenjtalk_prosody/train.total_count.ave_10best.pth:tts:tts:tts.generator.text_encoder.emb --freeze_param tts.generator.text_encoder.encoder tts.generator.decoder tts.decoder" \
        --tag "train_text_emb_${tag_prefix}_fold_${fold}_finetune_vits_raw_phn_jaconv_${g2p}" \
        "${additional_opts[@]}"

    echo "${tag_prefix} 的 Fold ${fold} 完成"
}

# 逐個 Fold 同時執行 run_kfold 和 run_kfold_role
echo "========== 開始執行所有 Fold =========="
cv_data_folder="phrase_emphasis/cross_validation"
# cv_data_folder_role="all_data_role/cross_validation"

k=1

for fold in $(seq 1 $k); do
    echo "處理 Fold ${fold}..."

    # Run kfold_role
        # train_set_role="${cv_data_folder_role}/fold_${fold}/train"
        # test_set_role="${cv_data_folder_role}/fold_${fold}/test"
        # run_fold "${fold}" "${train_set_role}" "${test_set_role}" "role"

    # Run kfold
    train_set="${cv_data_folder}/fold_${fold}/train"
    test_set="${cv_data_folder}/fold_${fold}/test"
    run_fold "${fold}" "${train_set}" "${test_set}" "default"

    echo "Fold ${fold} 完成"
done

echo "所有 Fold 的 run_kfold 和 run_kfold_role 全部完成"
