#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo, Fangjun Kuang)
# Copyright 2022 Johns Hopkins University (Author: Guanbo Wang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import MultiVENTAsrDataModule
from conformer import Conformer
from gigaspeech_scoring import asr_text_post_processing

from icefall.otc_graph_compiler import OtcTrainingGraphCompiler
from icefall.checkpoint import load_checkpoint
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    write_error_stats,
    str2bool,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="1best",
        help="""Decoding method.
        Supported values are:
            - (1) 1best. Extract the best path from the decoding lattice as the
              decoding result.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=1000,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, attention-decoder, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, attention-decoder, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
        help="The lang dir",
    )

    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="""The LM dir.
        It should contain either G_4_gram.pt or G_4_gram.fst.txt
        """,
    )

    # OTC alignment related
    parser.add_argument(
        "--otc-token",
        type=str,
        default="▁<star>",
        help="OTC token",
    )

    parser.add_argument(
        "--allow-bypass-arc",
        type=str2bool,
        default=True,
        help="""Whether to add bypass arc to training graph for substitution
        and insertion errors (wrong or extra words in the transcript).""",
    )

    parser.add_argument(
        "--allow-self-loop-arc",
        type=str2bool,
        default=True,
        help="""Whether to self-loop bypass arc to training graph for deletion errors
        (missing words in the transcript).""",
    )

    parser.add_argument(
        "--bypass-weight",
        type=float,
        default=0.0,
        help="Weight associated with bypass arc",
    )

    parser.add_argument(
        "--self-loop-weight",
        type=float,
        default=0.0,
        help="Weight associated with self-loop arc",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "subsampling_factor": 4,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "feature_dim": 80,
            "nhead": 8,
            "attention_dim": 512,
            "num_decoder_layers": 6,
            # parameters for alignment
            "beam_size": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            "env_info": get_env_info(),
        }
    )
    return params


def align_one_batch(
    texts: List[str],   
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
    graph_compiler: OtcTrainingGraphCompiler,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best alignment without LM rescoring.

      model:
        The neural model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      grpah_compiler:
        OTC graph compiler for OTC alignment.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    device = next(model.parameters()).device

    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]

    nnet_output, memory, memory_key_padding_mask = model(feature, supervisions)
    # nnet_output is (N, T, C)

    # append OTC log-prob to the end of nnet_output
    _, _, V = nnet_output.shape

    otc_token_log_prob = torch.logsumexp(
        nnet_output[:, :, 1:], dim=-1, keepdim=True
    ) - torch.log(torch.tensor([V - 1])).to(device)

    nnet_output = torch.cat([nnet_output, otc_token_log_prob], dim=-1)

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            supervisions["start_frame"] // params.subsampling_factor,
            supervisions["num_frames"] // params.subsampling_factor,
        ),
        1,
    ).to(torch.int32)

    alignment_graph = graph_compiler.compile(
        texts=texts,
        allow_bypass_arc=params.allow_bypass_arc,
        allow_self_loop_arc=params.allow_self_loop_arc,
        bypass_weight=params.bypass_weight,
        self_loop_weight=params.self_loop_weight,
    )

    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=3,
    )

    lattice = k2.intersect_dense(
        alignment_graph,
        dense_fsa_vec,
        params.beam_size,
    )

    best_path = one_best_decoding(
        lattice=lattice,
        use_double_scores=params.use_double_scores,
    )

    hyp = get_texts(best_path)
    hyp_texts_list = [
        [graph_compiler.token_table[i] for i in hyp_ids] for hyp_ids in hyp
    ]
    hyp_texts = ["".join(text_list).replace("▁", " ") for text_list in hyp_texts_list]

    return {"otc-alignment": hyp_texts}


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    graph_compiler: OtcTrainingGraphCompiler,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      graph_compiler:
        The OTC graph compiler for OTC alignment
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = align_one_batch(
            texts=texts,
            params=params,
            model=model,
            batch=batch,
            graph_compiler=graph_compiler,
        )

        for key, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words in zip(cut_ids, hyps):
                this_batch.append((cut_id, hyp_words))

            results[key].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    for _, results in results_dict.items():
        align_path = params.exp_dir / f"otc-alignment-{test_set_name}.txt"
        with open(align_path, "w", encoding="utf-8") as ali_p:
            for cut_id, hyp_words in results:
                ali_p.write(f"{cut_id} {hyp_words}\n")


@torch.no_grad()
def main():
    parser = get_parser()
    MultiVENTAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.lm_dir = Path(args.lm_dir)

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log-{params.method}/log-alignment")
    logging.info("OTC alignment started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    graph_compiler = OtcTrainingGraphCompiler(
        params.lang_dir,
        otc_token=params.otc_token,
        device=device,
    )

    # remove OTC token as it is actually a fake token (the average of all non-blank tokens)
    max_token_id = graph_compiler.get_max_token_id() - 1
    # +1 for the blank
    num_classes = max_token_id + 1

    logging.info(f"device: {device}")

    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )

    # load pretrained model
    load_checkpoint(f"{params.exp_dir}/pretrained.pt", model)

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    multivent = MultiVENTAsrDataModule(args)

    multivent_cuts = multivent.multivent_cuts()

    multivent_dl = multivent.test_dataloaders(multivent_cuts)

    test_sets = [f"{params.event}_{params.language}"]
    test_dls = [multivent_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = align_dataset(
            dl=test_dl,
            params=params,
            model=model,
            graph_compiler=graph_compiler,
        )

        save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info("Done!")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
