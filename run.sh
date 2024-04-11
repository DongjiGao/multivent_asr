#!/usr/bin/env bash

# 2024 Dongji Gao

set -euo pipefail

stage=0
stop_stage=100

finetuning_exp_dir=""
pretrained_model="pretrained_model/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/"
exp_dir="conformer_ctc/exp"
lang_dir="data/lang_bpe_500"
decoding_method="1best"

gpus=1

. ./cmd.sh
. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

events=(
    emergency_data
    political_data
    social_data
    technology_data
)

languages=(
    en
)

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Preparing transcribed and segmented multiVENT Lhotse dataset."
    ./prepare.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Fine-tuning Librispeech Transducer model on multiVent dataset."
    ./pruned_transducer_stateless7/finetune.py \
        --world-size ${gpus} \
        --num-epochs 10 \
        --start-epoch 1 \
        --exp-dir "${finetuning_exp_dir}" \
        --base-lr 0.005 \
        --lr-epochs 50 \
        --lr-batches 50000 \
        --bpe-model "${pretrained_model}/data/lang_bpe_500/bpe.model" \
        --do-finetune True \
        --finetune-ckpt "${pretrained_model}/exp/pretrained.pt" \
        --max-duration 500 \
        --language "en"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   log "Stage 1: Doing OTC flexible alingment." 
    for event in ${events[@]}; do
        for language in ${languages[@]}; do
            log "Aligning ${event} ${language}"
            ./conformer_ctc/otc_alignment.py \
                --event "${event}" \
                --language "${language}" \
                --method "${decoding_method}" \
                --exp-dir "${exp_dir}" \
                --lang-dir "${lang_dir}" 
       done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Post-processing OTC aligned text for better readability"
    for event in ${events[@]}; do
        for language in ${languages[@]}; do
            local/post_process_text.py \
                --text "${exp_dir}/otc-alignment-${event}_${language}.txt" \
                --whisper-text "data/${event}_${language}/text_raw" \
                --output-dir "data/${event}_${language}"
            sort "data/${event}_${language}/otc_aligned_text.txt" > "data/${event}_${language}/otc_aligned_text_sorted.txt"
        done
    done
fi
