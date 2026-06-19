from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IdentityProjection:
    """Pickle-friendly projection shim for direct HVG mode."""

    n_features: int

    def __post_init__(self) -> None:
        n = int(self.n_features)
        self.n_components = n
        self.n_components_ = n
        self.n_features_in_ = n
        self.mean_ = np.zeros((n,), dtype=np.float32)
        self.components_ = np.eye(n, dtype=np.float32)

    def transform(self, x):
        return np.asarray(x, dtype=np.float32)
