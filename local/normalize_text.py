#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer
from num2words import num2words


def get_args():
    parser = argparse.ArgumentParser(description="This script normalized text file")
    parser.add_argument(
        "--text",
        type=str,
        help="raw text file",
    )
    parser.add_argument(
        "--segments",
        type=str,
        help="segments file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to the output directory",
    )
    return parser.parse_args()


def normalize_text(text_list, normalizer):
    punctuations = (".", ",", "?")

    text = normalizer(" ".join(text_list))
    normalized_text_list = []

    # normalize numbers
    for word in text.split():
        if not word.isalpha():
            try:
                word = num2words(word).replace("-", " ")
            except:
                pass
        normalized_text_list.append(word)

    # caplitize and remove punctuations
    normalized_text = " ".join(normalized_text_list).upper()

    for punctuation in punctuations:
        normalized_text = normalized_text.replace(punctuation, "")
    normalized_text = normalized_text.strip()

    return normalized_text


def main():
    args = get_args()
    text_file = Path(args.text)
    segments_file = Path(args.segments)
    output_dir = Path(args.output_dir)

    normalizer = EnglishTextNormalizer()

    seg_ids = []

    with open(text_file, "r") as tf:
        with open(output_dir / "text", "w") as ot:
            for line in tf:
                line_list = line.strip().split()

                if len(line_list) > 1:
                    raw_text_list = line_list[1:]
                    normalized_text = normalize_text(raw_text_list, normalizer)

                    if normalized_text:
                        seg_id = line_list[0]
                        ot.write(f"{seg_id} {normalized_text}\n")
                        seg_ids.append(seg_id)

    if segments_file:
        seg_dict = {}
        with open(segments_file, "r") as sf:
            with open(output_dir / "segments", "w") as seg:
                for line in sf:
                    seg_id, reco_id, start_time, end_time = line.strip().split()
                    assert seg_id not in seg_dict
                    seg_dict[seg_id] = (reco_id, start_time, end_time)

                for seg_id in seg_ids:
                    reco_id, start_time, end_time = seg_dict[seg_id]
                    seg.write(f"{seg_id} {reco_id} {start_time} {end_time}\n")


if __name__ == "__main__":
    main()