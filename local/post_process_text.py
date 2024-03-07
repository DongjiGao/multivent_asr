#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path

from kaldialign import align


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        help="path to the text file",
    )
    parser.add_argument(
        "--whisper-text",
        type=str,
        help="path to the whisper raw text file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to the output directory",
    )
    return parser.parse_args()


def align_text(text_list, whisper_text_list, error_token="ERR"):

    for i, word in enumerate(text_list):
        if word == "<star>":
            text_list[i] = "*"

    result = []
    ali = align(text_list, whisper_text_list, error_token)
    for word, whisper_word in ali:
        if word == "*":
            aligned_word = word
        elif word == error_token:
            assert whisper_word != error_token
            aligned_word = whisper_word
        elif whisper_word == error_token:
            assert word != error_token
            aligned_word = word
        else:
            aligned_word = whisper_word

        result.append(aligned_word)
    return result

def main():
    args = arg_parser()
    text_file = Path(args.text)
    whisper_text_file = Path(args.whisper_text)
    output_dir = Path(args.output_dir)

    # read ans store whisper text
    whisper_text = {}
    with open(whisper_text_file, "r") as wt:
        for line in wt:
            utt_id, text = line.strip().split(maxsplit=1)
            assert utt_id not in whisper_text
            whisper_text[utt_id] = text

    with open(text_file, "r") as tf:
        with open(output_dir / "otc_aligned_text.txt", "w") as oat:
            for line in tf:
                utt_id, text = line.strip().split(maxsplit=1)
                assert utt_id in whisper_text

                text_list = text.lower().split()
                whisper_text_list = whisper_text[utt_id].split()

                aligned_text_list = align_text(text_list, whisper_text_list)
                aligned_text_list[0] = aligned_text_list[0].capitalize()

                oat.write(f"{utt_id} {' '.join(aligned_text_list)}\n")



if __name__ == "__main__":
    main()
