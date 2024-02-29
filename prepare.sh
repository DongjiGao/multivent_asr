#!/usr/bin/env bash

# 2024 Dongji Gao

set -euo pipefail

stage=0
stop_stage=100

corpus_dir="/exp/mmohammadi/multiVENT/data_wav"
data_dir="data"
manifest_dir="${data_dir}/manifests"

. ./cmd.sh
. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

events=(
    emergency_data
    police_data
    social_data
    technology_data
)
events=(
    police_data
    social_data
    technology_data
)


languages=(
    en
)

mkdir -p ${data_dir}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Preparing wav file for each event and language."
    wav_dir="wav_files"
    for event in ${events[@]}; do
        for langugage in ${languages[@]}; do
            log "Processing ${event} ${language}"

            output_wav_scp_dir="${data_dir}/${event}_${language}"
            mkdir -p "${output_wav_scp_dir}"
            local/make_wav.py \
                --corpus-dir "${corpus_dir}" \
                --event "${event}" \
                --language "${language}" \
                --output-wav-dir "${wav_dir}" \
                --output-wav-scp-dir "${output_wav_scp_dir}"
        done
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Generating segments using WHISPER."

    for event in ${events[@]}; do
        for language in ${languages[@]}; do
            log "Processing ${event} ${language}"
            wav_dir="${data_dir}/${event}_${language}"
            output_dir="${wav_dir}"

#            ${cuda_cmd} log/segmentation_${event}_${language}.log \
            local/whisper_segmentation.py \
                --wav-scp "${wav_dir}/wav.scp" \
                --output-dir "${output_dir}"
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Normalizing text"

    for event in ${events[@]}; do
        for language in ${languages[@]}; do
            log "Processing ${event} ${language}"
            text="${data_dir}/${event}_${language}/text_raw"
            segments="${data_dir}/${event}_${language}/segments_raw"
            output_dir="${data_dir}/${event}_${language}"

            local/normalize_text.py \
                --text "${text}" \
                --segments "${segments}" \
                --output-dir "${output_dir}"
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Making Lhoste manifest."

    mkdir -p "${manifest_dir}"
    for event in ${events[@]}; do
        for language in ${languages[@]}; do
            log "Processing ${event} ${language}"

            local/make_manifest.py \
                --data-dir "${data_dir}" \
                --event "${event}" \
                --language "${language}" \
                --manifest-dir "${manifest_dir}"
        done
    done
fi

exit 0;
