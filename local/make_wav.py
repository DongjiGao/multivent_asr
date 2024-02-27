#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path

import ffmpeg


def get_args():
    parser = argparse.ArgumentParser(description="This script creates wav.scp file")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        help="path to the wav file",
    )
    parser.add_argument(
        "--event",
        type=str,
        help="event",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="language",
    )
    parser.add_argument(
        "--output-wav-dir",
        type=str,
        help="path to the output wav directory",
    )
    parser.add_argument(
        "--output-wav-scp-dir",
        type=str,
        help="path to the output wav.scp directory",
    )
    return parser.parse_args()


def main():
    args = get_args()
    corpus_dir = Path(args.corpus_dir)
    event = args.event
    langauge = args.language
    output_wav_dir = Path(args.output_wav_dir)
    output_wav_scp_dir = Path(args.output_wav_scp_dir)

    with open(output_wav_scp_dir / "wav.scp", "w") as ws:
        wav_dir = corpus_dir / event / langauge
        for wav in wav_dir.glob("*.wav"):
            wav_id = wav.stem
            in_wav_path = wav.resolve()
            
            (
                ffmpeg.input(in_wav_path)
                .output(str(output_wav_dir / f"{wav_id}.wav"), ac=1, ar=16000)
                .run()
            )

            ws.write(f"{wav_id} {output_wav_dir}/{wav_id}.wav\n")


if __name__ == "__main__":
    main()
