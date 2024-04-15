#!/usr/bin/env python3

# 2024 John Hopkins University (author: Dongji Gao)

import os
import argparse
from pathlib import Path

from openai import OpenAI


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ctm",
        type=str,
        help="path to the CTM file",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="max duration of the segment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to the output directory",
    )
    return parser.parse_args()


def is_eos(word):
    eos_punctuations = (".", "!", "?")
    for punctuation in eos_punctuations:
        if word.endswith(punctuation):
            return True
    return False


def set_up_client():
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return client


def resegment(reco_id, text_list, ctm_list, client):
    assert len(text_list) == len(ctm_list)

    text = " ".join(text_list)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "assistant",
                "content": "Given the following text, identify and list each sentence separately, don't miss or add any word. return 1. plus identical text if there's only one sentence or one word:",
            },
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content

    num_sentences = len(content.split("\n"))
    num_tokens = len(content.split())
    num_words = num_tokens - num_sentences
    assert num_words == len(text_list), (
        content, text
    )

    segments = []
    start = None
    i = 0
    for sentence in content.split("\n"):
        text_list = []
        for word in sentence.split()[1:]:
            ctm_word, ctm_start, ctm_end = ctm_list[i]
            i += 1

            text_list.append(ctm_word)
            if start is None:
                start = ctm_start
        end = ctm_end
        text = " ".join(text_list)
        segments.append((reco_id, start, end, text))
        start = None

    return segments


def main():
    args = get_args()
    ctm_file = Path(args.ctm)
    output_dir = Path(args.output_dir)
    max_duration = args.max_duration

    segments = []
    text_list = []
    ctm_list = []

    client = set_up_client()

    prev_reco_id = None
    start = None
    with open(ctm_file, "r") as f:
        for line in f:
            word_id, _, word_st, word_dur, word = line.strip().split(" ", 4)

            reco_id = "_".join(word_id.split("_")[:-1])

            # new recordings
            if reco_id != prev_reco_id:
                # append the left of previous recordings
                if text_list:
                    end = word_end
                    cur_seg_duration = end - start

                    if cur_seg_duration > max_duration:
                        try:
                            result_segments = resegment(prev_reco_id, text_list, ctm_list, client)
                        except:
                            text = " ".join(text_list)
                            result_segments = [(prev_reco_id, start, end, text)]
                    else:
                        text = " ".join(text_list)
                        result_segments = [(prev_reco_id, start, end, text)]

                    for cur_segment in result_segments:
                        prev_reco_id, start, end, text = cur_segment
                        segments.append((prev_reco_id, start, end, text))

                    text_list = []
                    ctm_list = []

                start = None
                prev_reco_id = reco_id

            word_start = float(word_st)
            word_duration = float(word_dur)
            word_end = word_start + word_duration

            text_list.append(word)
            ctm_list.append((word, word_start, word_end))

            if start is None:
                start = word_start

            if is_eos(word):
                end = word_end
                cur_seg_duration = end - start

                if cur_seg_duration > max_duration:
                    try:
                        result_segments = resegment(prev_reco_id, text_list, ctm_list, client)
                    except:
                        text = " ".join(text_list)
                        result_segments = [(reco_id, start, end, text)]
                else:
                    text = " ".join(text_list)
                    result_segments = [(reco_id, start, end, text)]

                for cur_segment in result_segments:
                    prev_reco_id, start, end, text = cur_segment
                    segments.append((prev_reco_id, start, end, text))

                text_list = []
                ctm_list = []
                start = None

        # append the last one of last recording (it may not end with [., !, ?])
        if text_list:
            end = word_end

            if cur_seg_duration > max_duration:
                try:
                    result_segments = resegment(prev_reco_id, text_list, ctm_list, client)
                except:
                    pass
            else:
                text = " ".join(text_list)
                result_segments = [(reco_id, start, end, text)]

            for cur_segment in result_segments:
                prev_reco_id, start, end, text = cur_segment
                segments.append((prev_reco_id, start, end, text))

    with open(output_dir / "text_raw", "w") as ot, open(
        output_dir / "segments_raw", "w"
    ) as seg:
        for reco_id, start, end, text in segments:
            start_str = format(int(format(start, "0.3f").replace(".", "")), "08d")
            end_str = format(int(format(end, "0.3f").replace(".", "")), "08d")
            segment_id = f"{reco_id}-{start_str}-{end_str}"
            ot.write(f"{segment_id} {text}\n")
            seg.write(f"{segment_id} {reco_id} {start:.3f} {end:.3f}\n")


if __name__ == "__main__":
    main()
