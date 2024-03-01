#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path


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


def main():
    args = get_args()
    ctm_file = Path(args.ctm)
    output_dir = Path(args.output_dir)
    max_duration = args.max_duration

    segments = []
    text_list = []
    prev_reco_id = None
    start = None
    with open(ctm_file, "r") as f:
        for line in f:
            word_id, _, word_start_str, word_duration_str, word = line.strip().split()
            reco_id = "_".join(word_id.split("_")[:-1])

            # new recordings
            if reco_id != prev_reco_id:
                # append the left of previous recordings
                if text_list:
                    end = word_end
                    text = " ".join(text_list)
                    segments.append((prev_reco_id, start, end, text))
                    text_list = []

                start = None
                prev_reco_id = reco_id

            # check duration:
            cur_seg_duration = word_end - start if start is not None else 0
            if cur_seg_duration >= max_duration:
                # split by (possible) BOS
                if word[0].isupper():
                    end = word_end
                    text = " ".join(text_list)
                    segments.append((reco_id, start, end, text))
                    text_list = []
                    start = None


            word_start = float(word_start_str)
            word_duration = float(word_duration_str)
            word_end = word_start + word_duration

            text_list.append(word)

            if start is None:
                start = word_start

            if is_eos(word):
                end = word_end
                text = " ".join(text_list)
                segments.append((reco_id, start, end, text))
                text_list = []
                start = None

        # append the last one (it may not end with [., !, ?])
        if text_list:
            end = word_end
            text = " ".join(text_list)
            segments.append((reco_id, start, end, text))

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
