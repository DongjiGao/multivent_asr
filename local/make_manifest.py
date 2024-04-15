#!/usr/bin/env python3

# 2024 Johns Hopkins University (author: Dongji Gao)

import argparse
from pathlib import Path

import lhotse


def get_args():
    parser = argparse.ArgumentParser(
        description="This script creates manifest file for multiVENT dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="path to the data directory",
    )
    parser.add_argument(
        "--event",
        type=str,
        help="event name",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="language",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        help="output manifest dir",
    )
    return parser.parse_args()


def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    event = args.event
    language = args.language
    manifest_dir = Path(args.manifest_dir)

    recording_set, supervision_set, _ = lhotse.kaldi.load_kaldi_data_dir(
        data_dir / f"{event}_{language}", sampling_rate=16000
    )

    recording_set.to_file(
        str(manifest_dir / f"multivent_recordings_{event}_{language}.jsonl.gz")
    )
    supervision_set.to_file(
        str(manifest_dir / f"multivent_supervisions_{event}_{language}.jsonl.gz")
    )


if __name__ == "__main__":
    main()
