#!/usr/bin/env bash

# 2024 Dongji Gao

set -euo pipefail

exp_dir="conformer_ctc/exp"
lang_dir="data/lang_bpe_500"

decoding_method="1best"

. ./cmd.sh
. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

./conformer_ctc/decode.py \
    --method "${decoding_method}" \
    --exp-dir "${exp_dir}" \
    --lang-dir "${lang_dir}" 