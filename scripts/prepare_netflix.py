"""Convert the LIVE-Netflix-II per-video metadata into the TSKAN data splits.

Reads the 420 pickle files from the ``Dataset_Information/Pkl_Files`` archive
of the LIVE-Netflix-II database (Bampis et al., 2018, see paper ref. [17])
and writes, with a fixed 70/15/15 split (``random.seed(10)``):

* ``netflixpluspadded_{TRAIN,VALID,TEST}.ts``
  Multivariate time series (7 variables x 16 chunks) in TS format. The
  variables are computed per scene-cut-delimited segment:
  rebuffering_duration, video_bitrate, chunk_duration, qp, framerate,
  width, height. Sessions with more than 16 chunks are skipped and shorter
  sessions are padded with the first value up to 16 chunks.
  Target: ``retrospective_zscored_mos``.

* ``compnetflixpluspadded_{TRAIN,VALID,TEST}.ts``
  CSV of the expert-crafted features used by the interpretable baselines
  (paper, Section V-B): avg_bitrate, initial_delay, stallings,
  stalling_duration, switches, plus the target mos_zscore.

Usage
-----
::

    python scripts/prepare_netflix.py \
        --pkl-dir <path to Dataset_Information/Pkl_Files> \
        --out-dir data/netflix2
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd

# NOTE: the TEST/VALID headers keep the original ("Pus") spelling found in
# the shipped .ts files.
TS_HEADERS = {
    split: (
        f"@problemname Netflix {'Plus' if split == 'TRAIN' else 'Pus'} Dataset Padded\n"
        "@timestamps false\n"
        "@missing false\n"
        "@univariate false\n"
        "@dimension 7\n"
        "@equallength true\n"
        "@serieslength 16\n"
        "@targetlabel true\n"
        "@data\n"
    )
    for split in ("TRAIN", "VALID", "TEST")
}

CSV_HEADER = "avg_bitrate, initial_delay, stallings, stalling_duration, switches, mos_zscore\n"

MAX_CHUNKS = 16
# sessions with exactly 18 segments are dropped (paper: samples with more
# than 16 chunks are removed)
SKIP_SEGMENTS = 18


def write_dimension(f, dim, padding):
    for _ in range(padding):
        f.write(f"{dim[0]},")
    for k, val in enumerate(dim):
        if k > 0:
            f.write(",")
        f.write(f"{val}")
    f.write(":")


def session_to_features(data: pd.DataFrame):
    """Per-segment variables of one session (list of 7 parallel arrays)."""
    fps = data["frame_rate"]
    QP = data["per_segment_encoding_QP"]
    bitrate = data["playout_bitrate"]
    width = data["per_segment_encoding_width"]
    height = data["per_segment_encoding_height"]
    is_rebuffered_bool = data["is_rebuffered_bool"]
    scene_cuts = data["scene_cuts"]
    scene_cuts.insert(len(QP), data["N_playback_frames"])

    rebuffering_duration = []
    video_bitrate = []
    chunk_duration = []
    framerate_array = []

    start_image = 0
    buffered_total = 0
    overall_buffered_total = 0
    for i in range(len(scene_cuts)):
        j = start_image + overall_buffered_total
        while j < scene_cuts[i] + buffered_total + overall_buffered_total:
            buffered_total = buffered_total + is_rebuffered_bool[j]
            j = j + 1

        overall_buffered_total = overall_buffered_total + buffered_total
        rebuffering_duration.append(buffered_total / fps)
        video_bitrate.append(bitrate[start_image + overall_buffered_total])
        chunk_duration.append((scene_cuts[i] - start_image) / fps)
        framerate_array.append(fps)

        start_image = scene_cuts[i]
        buffered_total = 0

    return {
        "rebuffering_duration": rebuffering_duration,
        "video_bitrate": video_bitrate,
        "chunk_duration": chunk_duration,
        "QP": QP,
        "framerate_array": framerate_array,
        "width": width,
        "height": height,
    }


def session_to_expert_features(data: pd.DataFrame, feats: dict):
    """Expert-crafted features (paper ref. [22] style) of one session."""
    bitrate = data["playout_bitrate"]
    width = data["per_segment_encoding_width"]
    is_reb_pd = pd.Series(data["is_rebuffered_bool"])
    stallings = int(((is_reb_pd == 1) & (is_reb_pd.shift(fill_value=0) == 0)).sum())
    width_pd = pd.Series(width)
    switches = int((width_pd != width_pd.shift()).sum()) - 1
    return {
        "avg_bitrate": float(np.mean(bitrate)),
        "initial_delay": 0,
        "stallings": stallings,
        "stalling_duration": float(np.sum(feats["rebuffering_duration"])),
        "switches": switches,
    }


def prepare(pkl_dir, out_dir, seed=10):
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    print(f"Total number of files: {len(files)}")
    if not files:
        raise SystemExit(f"no .pkl files found in {pkl_dir}")

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    paths = {
        split: os.path.join(out_dir, f"netflixpluspadded_{split}.ts") for split in ("TRAIN", "TEST", "VALID")
    }
    comp_paths = {
        split: os.path.join(out_dir, f"compnetflixpluspadded_{split}.ts") for split in ("TRAIN", "TEST", "VALID")
    }
    ts_f = {s: open(p, "w") for s, p in paths.items()}
    comp_f = {s: open(p, "w") for s, p in comp_paths.items()}
    for s, f in ts_f.items():
        f.write(TS_HEADERS[s])
    for f in comp_f.values():
        f.write(CSV_HEADER)

    counts = {s: 0 for s in ("TRAIN", "TEST", "VALID")}
    n_skipped = 0
    for file_path_i in files:
        r = random.random()
        if r > 0.3:
            split = "TRAIN"
        elif r > 0.15:
            split = "TEST"
        else:
            split = "VALID"

        data = pd.read_pickle(file_path_i)
        segments = len(data["per_segment_encoding_QP"])
        if segments == SKIP_SEGMENTS:
            n_skipped += 1
            continue

        feats = session_to_features(data)
        mos_zscore = data["retrospective_zscored_mos"]
        padding = MAX_CHUNKS - segments

        f = ts_f[split]
        write_dimension(f, feats["rebuffering_duration"], padding)
        write_dimension(f, feats["video_bitrate"], padding)
        write_dimension(f, feats["chunk_duration"], padding)
        write_dimension(f, feats["QP"], padding)
        write_dimension(f, feats["framerate_array"], padding)
        write_dimension(f, feats["width"], padding)
        write_dimension(f, feats["height"], padding)
        f.write(f"{mos_zscore}\n")

        ef = session_to_expert_features(data, feats)
        comp_f[split].write(
            f"{ef['avg_bitrate']}, {ef['initial_delay']}, {ef['stallings']}, "
            f"{ef['stalling_duration']}, {ef['switches']}, {mos_zscore}\n"
        )
        counts[split] += 1

    for f in list(ts_f.values()) + list(comp_f.values()):
        f.close()

    print(f"split sizes: {counts} (skipped {n_skipped} sessions with {SKIP_SEGMENTS} segments)")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl-dir", required=True,
                    help="path to the LIVE-Netflix-II Dataset_Information/Pkl_Files directory")
    ap.add_argument("--out-dir", default="data/netflix2")
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    prepare(args.pkl_dir, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
