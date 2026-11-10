"""Frequency-domain (DFT) features used by TSKAN (paper, Section III-B).

For each variable (channel) ``v`` of a sample, the discrete Fourier
transform over the temporal dimension is computed::

    X_v(f) = sum_t x_v(t) e^{-j 2 pi f t / T},   f = 0, ..., T-1

From it TSKAN retains the first ``F + 1`` frequencies (``f = 0..F``):

* the DC component ``M_v(0) = |X_v(0)|`` (average level),
* the magnitude ``M_v(f) = |X_v(f)|`` for ``f = 1..F``,
* the phase ``phi_v(f) = arg(X_v(f))`` for ``f = 1..F``
  (and for ``f = 0`` as well when ``include_dc_phase`` is True).

In the paper, ``F = 1`` is used, followed by a top-k selection of the
most important components (Section III-C).
"""

from __future__ import annotations

import numpy as np

__all__ = ["FEATURE_NAMES", "dft_features", "dft_feature_names"]

# The time series contain 7 variables; the 7th (video height) is constant
# within a session and is dropped, leaving the 6 variables below.
FEATURE_NAMES = [
    "rebuffering_duration",
    "video_bitrate",
    "chunk_duration",
    "qp",
    "framerate",
    "width",
]


def dft_features(X, F=1, include_dc_phase=False):
    """Compute DFT-based features for every channel of every sample.

    Parameters
    ----------
    X : np.ndarray, shape (N, V, T)
        N samples, V variables (channels), T time steps.
    F : int, default 1
        Number of non-DC frequencies to retain.
    include_dc_phase : bool, default False
        If True, also retain the phase of the DC component.

    Returns
    -------
    np.ndarray, shape (N, n_features) with n_features = V * ((F+1) + F + include_dc_phase)
    """
    n_per_var = (F + 1) + F + (1 if include_dc_phase else 0)
    out = np.empty((X.shape[0], X.shape[1] * n_per_var), dtype=np.float64)
    for n, x in enumerate(X):
        i = 0
        for ch in x:
            c = np.fft.fft(ch)
            out[n, i : i + F + 1] = np.abs(c[: F + 1])
            i += F + 1
            if include_dc_phase:
                out[n, i] = np.angle(c[0])
                i += 1
            out[n, i : i + F] = np.angle(c[1 : F + 1])
            i += F
    return out


def dft_feature_names(V=6, F=1, include_dc_phase=False, var_names=FEATURE_NAMES):
    """Human-readable names of the features produced by :func:`dft_features`."""
    names = []
    for v in var_names[:V]:
        names.append(f"M_{v}(0)")
        for f in range(1, F + 1):
            names.append(f"M_{v}({f})")
        if include_dc_phase:
            names.append(f"phase_{v}(0)")
        for f in range(1, F + 1):
            names.append(f"phase_{v}({f})")
    return names
