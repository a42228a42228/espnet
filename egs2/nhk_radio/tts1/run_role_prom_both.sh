#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050
n_fft=1024
n_shift=256
win_length=null

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=prom_role/train
valid_set=prom_role/test
test_set=prom_role/test

train_config=conf/tuning/train_vits_22k.yaml
inference_config=conf/tuning/decode_vits.yaml

# Input example: こ、こんにちは

# 1. Phoneme + Pause
# (e.g. k o pau k o N n i ch i w a)
# g2p=pyopenjtalk

# 2. Kana + Symbol
# (e.g. コ 、 コ ン ニ チ ワ)
# g2p=pyopenjtalk_kana

# 3. Phoneme + Accent
# (e.g. k 1 0 o 1 0 k 5 -4 o 5 -4 N 5 -3 n 5 -2 i 5 -2 ch 5 -1 i 5 -1 w 5 0 a 5 0)
# g2p=pyopenjtalk_accent

# 4. Phoneme + Accent + Pause
# (e.g. k 1 0 o 1 0 pau k 5 -4 o 5 -4 N 5 -3 n 5 -2 i 5 -2 ch 5 -1 i 5 -1 w 5 0 a 5 0)
# g2p=pyopenjtalk_accent_with_pause

# 5. Phoneme + Prosody symbols
# (e.g. ^, k, #, o, _, k, o, [, N, n, i, ch, i, w, a, $)
# g2p=pyopenjtalk_prosody

sp_tokens=("<0>" "<1>" "<2>" "<e>" "</e>")
g2p=pyopenjtalk_prosody_sp_token_with_role

common_args=(
    --ngpu 2
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

tag_prefix=role_prom_both

##############################################
# pretrained text embedding
##############################################
# ./tts.sh \
#     --stage 2 \
#     --stop_stage 5 \
#     "${common_args[@]}" \
#     --train_config "${train_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_set}" \
#     --srctexts "data/${train_set}/text" \
#     --train_args "--init_param downloads/d3b0a3afbf3c5e54e676885ec949237b/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_prosody/train.total_count.ave_10best.pth:tts:tts:tts.generator.text_encoder.emb --freeze_param tts.generator.text_encoder.encoder tts.generator.decoder tts.decoder" \
#     --tag "${tag_prefix}_pretrained_emb_vits_raw_phn_jaconv_${g2p}" \
#     "${additional_opts[@]}"

# # 更新 token_list
# TOKEN_LIST_DIR=$(pwd)"/dump/token_list/phn_jaconv_${g2p}"
# pyscripts/utils/make_token_list_from_config.py downloads/d3b0a3afbf3c5e54e676885ec949237b/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_prosody/config.yaml
# mv ${TOKEN_LIST_DIR}/tokens.{txt,txt.bak}
# cp $(pwd)/downloads/d3b0a3afbf3c5e54e676885ec949237b/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_prosody/tokens.txt ${TOKEN_LIST_DIR}/tokens.txt
# TOKEN_LIST=${TOKEN_LIST_DIR}/tokens.txt
# for token in "${sp_tokens[@]}"; do
#     echo "$token" >> "$TOKEN_LIST"
# done

# ./tts.sh \
#     --stage 6 \
#     --stop_stage 8 \
#     "${common_args[@]}" \
#     --train_config "${train_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_set}" \
#     --srctexts "data/${train_set}/text" \
#     --train_args "--init_param downloads/d3b0a3afbf3c5e54e676885ec949237b/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_prosody/train.total_count.ave_10best.pth:tts:tts:tts.generator.text_encoder.emb --freeze_param tts.generator.text_encoder.encoder tts.generator.decoder tts.decoder" \
#     --tag "${tag_prefix}_pretrained_emb_vits_raw_phn_jaconv_${g2p}" \
#     "${additional_opts[@]}"


##############################################
# finetune all model
##############################################

# ./tts.sh \
#     --stage 2 \
#     --stop_stage 5 \
#     "${common_args[@]}" \
#     --train_config "${train_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_set}" \
#     --srctexts "data/${train_set}/text" \
#     --train_args "--init_param exp/tts_${tag_prefix}_pretrained_emb_vits_raw_phn_jaconv_${g2p}/train.total_count.ave_10best.pth" \
#     --tag "${tag_prefix}_finetune_vits_raw_phn_jaconv_${g2p}" \
#     "${additional_opts[@]}"

# # 更新 token_list
# TOKEN_LIST_DIR=$(pwd)"/dump/token_list/phn_jaconv_${g2p}"
# pyscripts/utils/make_token_list_from_config.py downloads/d3b0a3afbf3c5e54e676885ec949237b/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_prosody/config.yaml
# mv ${TOKEN_LIST_DIR}/tokens.{txt,txt.bak}
# cp $(pwd)/downloads/d3b0a3afbf3c5e54e676885ec949237b/exp/tts_train_vits_raw_phn_jaconv_pyopenjtalk_prosody/tokens.txt ${TOKEN_LIST_DIR}/tokens.txt
# TOKEN_LIST=${TOKEN_LIST_DIR}/tokens.txt
# for token in "${sp_tokens[@]}"; do
#     echo "$token" >> "$TOKEN_LIST"
# done

./tts.sh \
    --stage 9\
    --stop_stage  9\
    --skip_packing false\
    "${common_args[@]}" \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --srctexts "data/${train_set}/text" \
    --train_args "--init_param exp/tts_${tag_prefix}_pretrained_emb_vits_raw_phn_jaconv_${g2p}/train.total_count.ave_10best.pth" \
    --tag "${tag_prefix}_finetune_vits_raw_phn_jaconv_${g2p}" \
    "${additional_opts[@]}"