#!/usr/bin/env python3

text = "data/dev/text"
seg = "data/dev/segments"

with open(text, "r") as f:
    with open(seg, "w") as g:
        for line in f.readlines():
            utt_id = line.strip().split()[0]
            wav_id, start, end = utt_id.split("-")
            start = float(start) / 1000
            end = float(end) / 1000
            g.write(f"{utt_id} {wav_id} {start:.3f} {end:.3f}\n")
