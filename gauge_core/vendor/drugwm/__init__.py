"""Vendored minimal subset of the original GAUGE model implementation
(internal codename: drugwm) needed for pure inference: model architecture,
feature/projection utilities, and the candidate-ranking planner.

Vendored verbatim (unmodified) from the training repository so that
`pickle.load` of exported model bundles resolves `drugwm.features.FeatureArtifacts`
and `drugwm.model.TerminalWorldModel` to the same module path used when the
bundles were created. This package intentionally excludes training-only
modules (data ingestion, knowledge-graph construction, benchmarking
harnesses) so the deployed software has no dependency on the original
repository's absolute paths or its sqlite3-based ChEMBL parsing.
"""
__all__ = ["__version__"]
__version__ = "0.1.0"
