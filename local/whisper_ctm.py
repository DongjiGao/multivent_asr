#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path

import torch
import whisper_timestamped as whisper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav-scp",
        type=str,
        help="path to the wav file",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "base", "small", "medium"],
        default="medium",
        help="model size",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="language",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to the output directory",
    )
    return parser.parse_args()


def main():
    args = get_args()
    wav_scp = Path(args.wav_scp)
    output_dir = Path(args.output_dir)
    model_size = args.model_size
    language = args.language
    max_duration = args.max_duration

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(f"{model_size}.{language}", device=device)

    with open(output_dir / "ctm", "w") as ctm:
        with open(wav_scp, "r") as ws:
            for line in ws:
                wav_id, wav_path = line.strip().split()
                audio = whisper.load_audio(wav_path)

                # write reco2dur file
                audio_duration = len(audio) / 16000

                result = whisper.transcribe(
                    model,
                    audio,
                    beam_size=5,
                    best_of=5,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                )

                segments = result["segments"]
                word_index = 0
                exceed_audio = False

                for segment in segments:
                    if exceed_audio:
                        break

                    word_infos = segment["words"]
                    for word_info in word_infos:
                        word = word_info["text"]
                        w_start = word_info["start"]
                        w_end = word_info["end"]
                        w_duration = float(w_end) - float(w_start)

                        if w_end >= audio_duration:
                            exceed_audio = True
                            break

                        ctm.write(
                            f"{wav_id}_{word_index} 0 {w_start} {w_duration:.3f} {word}\n"
                        )
                        word_index += 1


if __name__ == "__main__":
    main()
