"""Pickle-compatibility shim.

The published GAUGE checkpoints were produced with a benchmarking patch
(`hvg2000_patch.py` in the training repo) that replaces PCA projection with a
direct top-variance highly-variable-gene (HVG) selection. That patch records an
`IdentityProjection` instance (module path `hvg2000_projection`) inside the
pickled `FeatureArtifacts.pca` field. This file must stay import-compatible
(same module name, same class name, same public attributes) so that artifact
bundles exported from those checkpoints can be unpickled without depending on
the original training repository's sys.path tricks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IdentityProjection:
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
