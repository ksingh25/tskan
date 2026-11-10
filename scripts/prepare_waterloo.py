"""Convert the Waterloo QoE Database 3 raw session logs into the TSKAN splits.

The raw data of this repository is in ``data/waterloo3/raw/``:
one CSV per session in ``streaming_logs/`` (5 time steps x 17 columns)
and ``data.csv`` mapping each session file to its raw MOS rating
(Waterloo QoE Database 3, Duanmu et al., 2018, see paper ref. [27]).

The MOS ratings are z-scored across sessions, and with a fixed 70/15/15
split (``random.seed(10)``) the script writes:

* ``waterloo3_zscore_{TRAIN,VALID,TEST}.ts``
  Multivariate time series (7 variables x 5 time steps) in TS format:
  rebuffering_duration, video_bitrate, chunk_duration, qp, framerate,
  width, height. Target: z-scored MOS.

* ``compwaterloo_{TRAIN,VALID,TEST}.ts``
  CSV of the expert-crafted features used by the interpretable baselines
  (paper, Section V-B): avg_bitrate, initial_delay, stallings,
  stalling_duration, switches, plus the target mos_zscore.

Usage
-----
::

    python scripts/prepare_waterloo.py \
        --logs-dir data/waterloo3/raw/streaming_logs \
        --mos-csv data/waterloo3/raw/data.csv \
        --out-dir data/waterloo3
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
from scipy import stats

TS_HEADER = (
    "@problemname Waterloo QoE dataset 3\n"
    "@timestamps false\n"
    "@missing false\n"
    "@univariate false\n"
    "@dimension 7\n"
    "@equallength true\n"
    "@serieslength 5\n"
    "@targetlabel true\n"
    "@data\n"
)

CSV_HEADER = "avg_bitrate, initial_delay, stallings, stalling_duration, switches, mos_zscore\n"

# columns of each session CSV (column 0 is the representation index)
TS_VARS = ["rebuffering_duration", "video_bitrate", "chunk_duration",
           "qp", "framerate", "width", "height"]
N_TIME_STEPS = 5


def prepare(logs_dir, mos_csv, out_dir, seed=10):
    files = sorted(glob.glob(os.path.join(logs_dir, "*.csv")))
    print(f"Total number of session files: {len(files)}")
    if not files:
        raise SystemExit(f"no session CSVs found in {logs_dir}")

    dt = pd.read_csv(mos_csv)
    zscore = stats.zscore(dt["mos"].values)  # (N_sessions,)

    # map z-scores to session files (robust to ordering)
    name_to_z = dict(zip(dt["streaming_log"], zscore))
    missing = [os.path.basename(f) for f in files if os.path.basename(f) not in name_to_z]
    if missing:
        raise SystemExit(f"session files missing from {mos_csv}: {missing[:5]} ...")

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    paths = {
        split: os.path.join(out_dir, f"waterloo3_zscore_{split}.ts") for split in ("TRAIN", "TEST", "VALID")
    }
    comp_paths = {
        split: os.path.join(out_dir, f"compwaterloo_{split}.ts") for split in ("TRAIN", "TEST", "VALID")
    }
    ts_f = {s: open(p, "w") for s, p in paths.items()}
    comp_f = {s: open(p, "w") for s, p in comp_paths.items()}
    for f in ts_f.values():
        f.write(TS_HEADER)
    for f in comp_f.values():
        f.write(CSV_HEADER)

    counts = {s: 0 for s in ("TRAIN", "TEST", "VALID")}
    for file_path_i in files:
        r = random.random()
        if r > 0.3:
            split = "TRAIN"
        elif r > 0.15:
            split = "TEST"
        else:
            split = "VALID"

        df = pd.read_csv(file_path_i)
        mos_z = name_to_z[os.path.basename(file_path_i)]

        # --- multivariate time series (7 vars x 5 steps) ---
        parts = []
        for i in range(len(TS_VARS)):
            for j in range(N_TIME_STEPS):
                parts.append(repr(df.loc[j][TS_VARS[i]]))
                parts.append("," if j < N_TIME_STEPS - 1 else ":")
        ts_f[split].write("".join(parts) + f"{mos_z}\n")

        # --- expert-crafted features ---
        rebuffering_duration = df["rebuffering_duration"]
        width_pd = pd.Series(df["width"])
        row = {
            "avg_bitrate": float(np.mean(df["video_bitrate"])),
            "initial_delay": float(rebuffering_duration.iloc[0]),
            "stallings": int((rebuffering_duration > 0).sum()),
            "stalling_duration": float(np.sum(rebuffering_duration)),
            "switches": int((width_pd != width_pd.shift()).sum()) - 1,
        }
        comp_f[split].write(
            f"{row['avg_bitrate']}, {row['initial_delay']}, {row['stallings']}, "
            f"{row['stalling_duration']}, {row['switches']}, {mos_z}\n"
        )
        counts[split] += 1

    for f in list(ts_f.values()) + list(comp_f.values()):
        f.close()

    print(f"split sizes: {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs-dir", default="data/waterloo3/raw/streaming_logs")
    ap.add_argument("--mos-csv", default="data/waterloo3/raw/data.csv")
    ap.add_argument("--out-dir", default="data/waterloo3")
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    prepare(args.logs_dir, args.mos_csv, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
