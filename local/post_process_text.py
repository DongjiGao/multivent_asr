#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        help="path to the text file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to the output directory",
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    text_file = Path(args.text)
    output_dir = Path(args.output_dir)

    result = []
    with open(text_file, "r") as tf:
        with open(output_dir / "otc_aligned_text.txt", "w") as oat:
            for line in tf:
                line = line.strip().split()
                utt_id = line[0]
                cur_text = [utt_id]

                for word in line[1:]:
                    if word == "<star>":
                        cur_text.append("*")
                    elif word == "I":
                        cur_text.append(word)
                    else:
                        cur_text.append(word.lower())

                result.append(" ".join(cur_text))

            result = sorted(result)
            for text in result:
                oat.write(f"{text}\n")


if __name__ == "__main__":
    main()
