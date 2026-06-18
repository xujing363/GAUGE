"""GAUGE core inference engine.

Thin, friendly wrapper around the original GAUGE model implementation
(internal codename in source: ``drugwm``). This package does not reimplement
any model math: it loads exported model bundles and orchestrates calls into
the original ``drugwm.model`` / ``drugwm.features`` / ``drugwm.planner``
modules so that predictions are bit-for-bit identical to the published
benchmark results.
"""
from __future__ import annotations

from . import _drugwm_path  # noqa: F401  must run before numpy/pandas/torch import anywhere
from .bundle import ModelBundle, load_bundle
from .predict import (
    PredictionResult,
    predict_one,
    rank_drugs,
    resolve_drug,
    resolve_sample,
    score_combination,
)

__all__ = [
    "ModelBundle",
    "load_bundle",
    "PredictionResult",
    "predict_one",
    "rank_drugs",
    "resolve_drug",
    "resolve_sample",
    "score_combination",
]

__version__ = "1.3.0"
