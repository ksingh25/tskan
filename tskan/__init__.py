"""TSKAN: Interpretable Machine Learning for QoE modeling over Time Series Data.

Combines compact frequency-domain (DFT) features with a single-layer
Kolmogorov-Arnold Network (KAN) readout for QoE (MOS) prediction from
raw streaming time series.
"""

from .data import load_ts, load_dataset
from .features import FEATURE_NAMES, dft_features, dft_feature_names
from .model import TSKAN

__version__ = "0.1.0"
__all__ = [
    "load_ts",
    "load_dataset",
    "FEATURE_NAMES",
    "dft_features",
    "dft_feature_names",
    "TSKAN",
]
