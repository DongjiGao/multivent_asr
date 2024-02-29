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


def post_process_segments(segments, max_duration):
    processed_segments = []
    start = None
    sentence_list = []

    for segment in segments:
        words = segment["words"]
        for word_info in words:
            word = word_info["text"]
            cur_start, cur_end = word_info["start"], word_info["end"]

            if start is None:
                start = cur_start

            sentence_list.append(word)

            # EOS
            if word.endswith(".") or word.endswith("?"):
                sentence = " ".join(sentence_list)
                processed_segments.append((start, cur_end, sentence))

                sentence_list = []
                start = None
        # Check if current segment length >= max duration allowed for segment
        cur_duration = cur_end - start
        if cur_duration >= max_duration:
            sentence = " ".join(sentence_list)
            processed_segments.append((start, cur_end, sentence))
            sentence_list = []
            start = None

    # check if the last segment is add (it may not end with period)
    if sentence_list:
        sentence = " ".join(sentence_list)

        end = cur_end
        processed_segments.append((start, end, sentence))

    return processed_segments


def main():
    args = get_args()
    wav_scp = Path(args.wav_scp)
    output_dir = Path(args.output_dir)
    model_size = args.model_size
    language = args.language
    max_duration = args.max_duration

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(f"{model_size}.{language}", device=device)

    with open(output_dir / "segments_raw", "w") as seg, open(
        output_dir / "text_raw", "w"
    ) as ot:
        with open(wav_scp, "r") as ws:
            for line in ws:
                wav_id, wav_path = line.strip().split()
                audio = whisper.load_audio(wav_path)

                print(f"id: {wav_id}")
                result = whisper.transcribe(
                    model,
                    audio,
                    beam_size=5,
                    best_of=5,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                )

                segments = result["segments"]

                # re-segment based on punctuation ("." or "?")
                post_segments = post_process_segments(segments, max_duration)

                for segment in post_segments:
                    start_time, end_time, text = segment

                    if text:
                        st_str = format(
                            int(format(start_time, "0.3f").replace(".", "")), "08d"
                        )
                        et_str = format(
                            int(format(end_time, "0.3f").replace(".", "")), "08d"
                        )
                        segment_id = f"{wav_id}-{st_str}-{et_str}"

                        seg.write(
                            f"{segment_id} {wav_id} {start_time:.3f} {end_time:.3f}\n"
                        )
                        ot.write(f"{segment_id} {text}\n")


if __name__ == "__main__":
    main()
