#!/usr/bin/env bash

# 2024 Johns Hopkins University (author: Dongji Gao)

set -euo pipefail

stage=0
stop_stage=100

corpus_dir=
lang_dir=

data_dir="data"
manifest_dir="${data_dir}/manifests"

feature_type="fbank"
feature_dir="${data_dir}/${feature_type}"

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

mkdir -p ${data_dir}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    wav_dir="wav_files"
    log "Stage 0: Preparing wav file for each event and language."

    for language in ${languages[@]}; do
        for event in ${events[@]}; do
            output_wav_scp_dir="${data_dir}/${event}_${language}"
            log "Processing ${event} ${language}"

            if [ -f "${wav_dir}/.${event}.${language}.done" ]; then
                log "Skip since ${wav_dir}/.${event}.${langauge}.done exists."
            else
                mkdir -p "${wav_dir}"
                mkdir -p "${output_wav_scp_dir}"
                local/make_wav.py \
                    --corpus-dir "${corpus_dir}" \
                    --event "${event}" \
                    --language "${language}" \
                    --output-wav-dir "${wav_dir}" \
                    --output-wav-scp-dir "${output_wav_scp_dir}" 

                log "wav files stored in ${wav_dir}"
                log "wav.scp file stored in ${output_wav_scp_dir}/"
                touch "${wav_dir}/.{event}.{language}.done"
            fi
        done
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Generating CTM file using WHISPER."

    for language in ${languages[@]}; do
        for event in ${events[@]}; do
            output_dir="${data_dir}/${event}_${language}"
            log "Processing ${event} ${language}"

            if [ -n "${OPENAI_API_KEY}" ]; then
                log "Please set OPENAI_API_KEY to use GPT for segmentation."
                exit 1
            else
                local/whisper_ctm.py \
                    --wav-scp "${output_dir}/wav.scp" \
                    --output-dir "${output_dir}"
                log "CTM info stored in ${output_dir}/ctm"
            fi
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generating raw segments from CTM file using GPT."
    for language in ${languages[@]}; do
        for event in ${events[@]}; do
            log "Processing ${event} ${language}"
            local/segment.py \
                --ctm "${data_dir}/${event}_${language}/ctm" \
                --output-dir "${data_dir}/${event}_${language}"
            log "raw segments stroed in ${data_dir}/${event}_${language}/segments_raw"
            log "corresponding raw texts stores in ${data_dir}/${event}_${language}/text_raw"
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Normalizing text"
    for language in ${languages[@]}; do
        for event in ${events[@]}; do
            log "Processing ${event} ${language}"

            text="${data_dir}/${event}_${language}/text_raw"
            segments="${data_dir}/${event}_${language}/segments_raw"
            output_dir="${data_dir}/${event}_${language}"

            local/normalize_text.py \
                --text "${text}" \
                --segments "${segments}" \
                --output-dir "${output_dir}"
            log "empty raw text and segments are removed"
            log "resulting text and segments file store in ${data_dir}/${event}_${language}"
        done
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Making Lhoste manifest."
    mkdir -p "${manifest_dir}"
    for language in ${languages[@]}; do
        for event in ${events[@]}; do
            log "Processing ${event} ${language}"

            local/make_manifest.py \
                --data-dir "${data_dir}" \
                --event "${event}" \
                --language "${language}" \
                --manifest-dir "${manifest_dir}"
            log "Lhotse manifests store in ${manifest_dir}"
        done
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Compute ${feature_type} feature for multiVENT"
    mkdir -p "${feature_dir}"

    if [ -e "${feature_dir}/.multivent.done" ]; then
        echo "Skip feature extraction since it has been done."
    else
        if [ "${feature_type}" = fbank ]; then
            for language in ${languages[@]}; do
                for event in ${events[@]}; do
                    log "Processing ${event} ${language} for ${feature_type} feature."

                    ./local/compute_fbank_multivent.py \
                        --manifest-dir "${manifest_dir}" \
                        --output-dir "${feature_dir}" \
                        --dataset "${event}_${language}" \
                        --perturb-speed false

                    lhotse cut trim-to-supervisions --discard-overlapping \
                        "${feature_dir}/multivent_cuts_${event}_${language}.jsonl.gz" - | \
                        gzip -c > "${feature_dir}/multivent_cuts_${event}_${language}_trimmed.jsonl.gz"

                    ./local/validate_manifest.py \
                        "${feature_dir}/multivent_cuts_${event}_${language}_trimmed.jsonl.gz"
                    log "Lhotse cuts with feature store in ${feature_dir}."
                done
            done
        else
            log "Error: not supported --feature-type ${feature_type}"
            exit 2
        fi
        touch "${feature_dir}/.multivent.done"
    fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Filtering cuts using BPE model"
    for language in ${languages[@]}; do
        for event in ${events[@]}; do
            ./local/filter_cuts.py \
                --bpe-model "${lang_dir}/bpe.model" \
                --in-cuts "${feature_dir}/multivent_cuts_${event}_${language}_trimmed.jsonl.gz" \
                --out-cuts "${feature_dir}/multivent_cuts_${event}_${language}_trimmed_filtered.jsonl.gz" 
        done
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Combine cuts for finetuning"
    for language in ${languages[@]}; do
        cat <(gunzip -c "${feature_dir}/multivent_cuts_emergency_data_${language}_trimmed_filtered.jsonl.gz") \
            <(gunzip -c "${feature_dir}/multivent_cuts_political_data_${language}_trimmed_filtered.jsonl.gz") \
            <(gunzip -c "${feature_dir}/multivent_cuts_social_data_${language}_trimmed_filtered.jsonl.gz") \
            <(gunzip -c "${feature_dir}/multivent_cuts_technology_data_${language}_trimmed_filtered.jsonl.gz") | shuf | \
            gzip -c > "${feature_dir}/multivent_cuts_combined_${language}_trimmed_filtered.jsonl.gz"
    done
fi

exit 0
