"""TSKAN model: a single-layer KAN readout on compact features.

The TSKAN model (paper, Figures 1 and 4) is a one-layer Kolmogorov-Arnold
Network ``width = [n_in, 1]``: the output is the sum of the trained
univariate activation functions of the input features, each of which is a
visualizable spline. The model is trained with LBFGS, as done in the
paper. It wraps ``pykan.kan.KAN`` (see https://github.com/KindXiaoming/pykan).
"""

from __future__ import annotations

import numpy as np
import torch

try:  # pykan cloned alongside (original setup: `from pykan.kan import KAN`)
    from pykan.kan import KAN
except ImportError:  # pykan installed with pip (`pykan==0.2.8`, top-level `kan`)
    from kan import KAN

__all__ = ["TSKAN"]


class TSKAN:
    """Single-layer KAN regressing MOS from (scaled) features.

    Parameters
    ----------
    n_features : int
        Number of input features.
    grid : int, default 3
        Number of spline intervals per edge.
    k : int, default 3
        Spline order.
    seed : int, default 9
        Random seed for the spline initialization.
    device : str or None
        'cpu' / 'cuda'; defaults to CUDA if available.
    """

    def __init__(self, n_features, grid=3, k=3, seed=9, device=None):
        self.n_features = n_features
        self.grid = grid
        self.k = k
        self.seed = seed
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = KAN(
            width=[n_features, 1], grid=grid, k=k, seed=seed,
            device=self.device, auto_save=False,
        )
        self.train_history = None

    @staticmethod
    def _as_tensor(arr, device):
        dtype = torch.get_default_dtype()
        return (
            torch.from_numpy(np.asarray(arr, dtype=np.float32))
            .type(dtype)
            .to(device)
        )

    def _dataset(self, X_train, y_train, X_valid, y_valid):
        """pykan dataset dict; 'test_*' here plays the validation role."""
        return {
            "train_input": self._as_tensor(X_train, self.device),
            "test_input": self._as_tensor(X_valid, self.device),
            "train_label": self._as_tensor(np.asarray(y_train, dtype=np.float32)[:, None], self.device),
            "test_label": self._as_tensor(np.asarray(y_valid, dtype=np.float32)[:, None], self.device),
        }

    def fit(self, X_train, y_train, X_valid, y_valid, steps=60, lr=1e-3):
        """Train with LBFGS (pykan ``KAN.fit``), tracking train/validation loss."""
        results = self.model.fit(
            self._dataset(X_train, y_train, X_valid, y_valid),
            opt="LBFGS",
            steps=steps,
            lr=lr,
        )
        self.train_history = results
        return self

    @property
    def importance(self):
        """Input importance scores alpha_j (paper, Section III-C)."""
        return self.model.attribute(1, 0, plot=False).detach()

    def top_k_indices(self, k=10):
        """Indices of the k most important inputs, sorted ascending."""
        idx = torch.topk(self.model.attribute(1, 0, plot=False), k).indices.numpy()
        return np.sort(idx)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model.forward(self._as_tensor(X, self.device))
        return out.cpu().numpy().ravel()

    def n_params(self):
        """Number of parameters of the learned activation functions.

        Counts, per input edge, the spline grid points and the base/spline
        scales (12 with grid=3), as reported in the paper (120 for top-k=10).
        """
        n = 0
        for name, p in self.model.named_parameters():
            if any(t in name for t in ("act_fun.0.grid", "scale_base", "scale_sp")):
                n += p.numel()
        return n

    def plot(self, *args, **kwargs):
        """Plot the learned univariate activation functions (pykan KAN.plot)."""
        return self.model.plot(*args, **kwargs)

    def get_fun(self, *args, **kwargs):
        return self.model.get_fun(*args, **kwargs)
