#!/usr/bin/env python3

# 2024 Dongji Gao

import argparse
from pathlib import Path

import torch
import whisper_timestamped as whisper
from whisper.normalizers import EnglishTextNormalizer
from num2words import num2words
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


def post_process_segments(segments):
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

    # check if the last segment is add (it may not end with period)
    if sentence_list:
        sentence = " ".join(sentence_list)
        end = cur_end
        processed_segments.append((start, end, sentence))

    return processed_segments


def normalize_text(text, normalizer):
    text = normalizer(text)
    normalized_text_list = []

    for word in text.split():
        if word.isnumeric():
            word = num2words(word).replace("-", " ")
        normalized_text_list.append(word)
    normalized_text = " ".join(normalized_text_list).upper()

    return normalized_text


def main():
    args = get_args()
    wav_scp = Path(args.wav_scp)
    output_dir = Path(args.output_dir)
    model_size = args.model_size
    language = args.language

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(f"{model_size}.{language}", device=device)

    normalizer = EnglishTextNormalizer()

    with open(output_dir / "segments", "w") as seg, open(
        output_dir / "text", "w"
    ) as ot:
        with open(wav_scp, "r") as ws:
            for line in tqdm(ws):
                wav_id, wav_path = line.strip().split()
                audio = whisper.load_audio(wav_path)
                result = whisper.transcribe(model, audio)

                segments = result["segments"]
                post_segments = post_process_segments(segments)

                for segment in post_segments:
                    start_time, end_time, text = segment
                    normalized_text = normalize_text(text, normalizer)

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
                    ot.write(f"{segment_id} {normalized_text}\n")


if __name__ == "__main__":
    main()
