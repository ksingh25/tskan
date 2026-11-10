"""Loaders for the TSKAN time series datasets (TS regression format).

The processed datasets shipped in ``data/`` are stored in the TS text
format with a regression target::

    @problemname <name>
    @timestamps false
    @missing false
    @univariate false
    @dimension <D>
    @equallength true
    @serieslength <T>
    @targetlabel true
    @data
    v1_0,v1_1,...,v1_T-1:v2_0,...,:...,vD_0,...,vD_T-1:<target>

Each line is one sample: D variables (chunks), T time steps per variable,
and one regression target (z-scored MOS).
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["load_ts", "load_dataset", "DATASET_FILES"]

# processed split file name per dataset
DATASET_FILES = {
    "netflix2": "netflixpluspadded_{split}.ts",
    "waterloo3": "waterloo3_zscore_{split}.ts",
}

SPLITS = ("TRAIN", "VALID", "TEST")


def load_ts(path):
    """Load a multivariate time series regression file in TS format.

    Parameters
    ----------
    path : str or path-like
        Path to the ``.ts`` file.

    Returns
    -------
    X : np.ndarray, shape (N, D, T), dtype float64
        N samples, D variables, T time steps.
    y : np.ndarray, shape (N,), dtype float64
        Regression target per sample.
    """
    n_dims = None
    series_len = None
    has_target = False
    in_data = False
    samples_x = []
    samples_y = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not in_data:
                if line.startswith("@dimension"):
                    n_dims = int(line.split()[1])
                elif line.startswith("@serieslength"):
                    series_len = int(line.split()[1])
                elif line.startswith("@targetlabel"):
                    has_target = line.split()[1].lower() == "true"
                elif line.startswith("@data"):
                    in_data = True
                continue

            parts = line.split(":")
            if has_target:
                x_parts, y_val = parts[:-1], float(parts[-1])
            else:
                x_parts, y_val = parts, None
            x = np.array([np.array(p.split(","), dtype=np.float64) for p in x_parts])
            if x.shape != (n_dims, series_len):
                raise ValueError(
                    f"{path}: line has shape {x.shape}, expected ({n_dims}, {series_len})"
                )
            samples_x.append(x)
            if y_val is not None:
                samples_y.append(y_val)

    if n_dims is None:
        raise ValueError(f"{path}: missing @dimension tag")
    X = np.stack(samples_x).astype(np.float64)
    y = np.array(samples_y, dtype=np.float64)
    return X, y


def load_dataset(dataset, data_dir="."):
    """Load the train/valid/test splits of a processed dataset.

    Parameters
    ----------
    dataset : 'netflix2' or 'waterloo3'
    data_dir : str, optional
        Path to the ``data/`` directory (default: ``.``).

    Returns
    -------
    dict with keys X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    if dataset not in DATASET_FILES:
        raise ValueError(f"unknown dataset '{dataset}', expected one of {list(DATASET_FILES)}")
    pattern = DATASET_FILES[dataset]
    out = {}
    for split in SPLITS:
        path = os.path.join(str(data_dir), dataset, pattern.format(split=split))
        X, y = load_ts(path)
        out[f"X_{split.lower()}"] = X
        out[f"y_{split.lower()}"] = y
    return out
