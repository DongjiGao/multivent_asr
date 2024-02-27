#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path

import torch
import whisper
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="This script creates wav.scp file")
    parser.add_argument(
        "--wav-scp",
        type=str,
        help="path to the wav file",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="large",
        help="model size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to the output directory",
    )
    return parser.parse_args()


def post_process_segments(segments):
    processed_segments = []
    start_time = None
    text = ""

    for segment in segments:
        cur_start_time = segment["start"]
        cur_end_time = segment["end"]
        cur_text = segment["text"]

        if start_time is None or not text:
            start_time = cur_start_time

        text += cur_text
        # add segment if it's a full sentence (end with period)
        if text.endswith("."):
            end_time = cur_end_time
            processed_segments.append((start_time, end_time, text))
            text = ""
    # check if the last segment is add (it may not end with period)
    if text:
        text += "."
        end_time = cur_end_time
        processed_segments.append((start_time, end_time, text))

    return processed_segments


def main():
    args = get_args()
    wav_scp = Path(args.wav_scp)
    output_dir = Path(args.output_dir)
    model_size = args.model_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size)
    model = model.to(device)

    with open(output_dir / "segments", "w") as seg, open(
        output_dir / "text", "w"
    ) as ot:
        with open(wav_scp, "r") as ws:
            for line in tqdm(ws):
                wav_id, wav_path = line.strip().split()
                result = model.transcribe(wav_path)

                segments = post_process_segments(result["segments"])
                for segment in segments:
                    start_time, end_time, text = segment

                    st_str = format(
                        int(format(start_time, "0.3f").replace(".", "")), "08d"
                    )
                    et_str = format(
                        int(format(end_time, "0.3f").replace(".", "")), "08d"
                    )
                    segment_id = f"{wav_id}-{st_str}-{et_str}"

                    seg.write(f"{segment_id} {wav_id} {start_time:.3f} {end_time:.3f}\n")
                    ot.write(f"{segment_id} {text}\n")


if __name__ == "__main__":
    main()
