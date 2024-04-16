#!/usr/bin/env bash

# 2024 Johns Hopkins University (author: Dongji Gao)

set -euo pipefail

stage=0
stop_stage=100

corpus_dir="/exp/mmohammadi/multiVENT/data_wav"

pretrained_model_dir="pretrained_model/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/"
#pretrained_model_dir="pretrained_model/icefall-asr-gigaspeech-zipformer-2023-10-17"
pretrained_lang_dir="${pretrained_model_dir}/data/lang_bpe_500"
finetune_out_exp_dir="pruned_transducer_stateless7/exp"
exp_dir="conformer_ctc/exp"

decoding_method="modified_beam_search"
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

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Preparing transcribed and segmented multiVENT Lhotse dataset."
    ./prepare.sh \
        --corpus-dir "${corpus_dir}" \
        --lang-dir "${pretrained_lang_dir}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Fine-tuning Zipformer Transducer model on MultiVent dataset."
    mkdir -p "${finetune_out_exp_dir}"
    ./pruned_transducer_stateless7/finetune.py \
        --world-size ${gpus} \
        --num-epochs 20 \
        --start-epoch 1 \
        --exp-dir "${finetune_out_exp_dir}" \
        --base-lr 0.005 \
        --lr-epochs 100 \
        --lr-batches 100000 \
        --bpe-model "${pretrained_model_dir}/data/lang_bpe_500/bpe.model" \
        --do-finetune True \
        --finetune-ckpt "${pretrained_model_dir}/exp/pretrained.pt" \
        --max-duration 500 \
        --language "en"
fi

echo "${pretrained_lang_dir}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Decoding with the fine-tuned model."
    ./pruned_transducer_stateless7/decode.py \
        --epoch 20 \
        --avg 5 \
        --exp-dir "${finetune_out_exp_dir}" \
        --max-duration 600 \
        --decoding-method "${decoding_method}" \
        --beam-size 8 \
        --lang-dir "${pretrained_lang_dir}" \
        --bpe-model "${pretrained_lang_dir}/bpe.model"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   log "Stage 3: Doing OTC flexible alingment." 
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

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Post-processing OTC aligned text for better readability"
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
