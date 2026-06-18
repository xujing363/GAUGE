"""A minimal, self-contained stand-in for `drugwm.kg_prior.MultiKGGraphArtifacts`.

`drugwm.model.MultiKGActionEncoder.__init__` only ever reads 6 attributes off
the kg_artifacts object it is given: `branch_names`, `drug_ids`,
`drug_to_local`, `node_table`, `edge_table`, `coverage`. The original class
lives in `drugwm.kg_prior`, a training-only module that (transitively)
imports `sqlite3` for ChEMBL parsing and is not vendored into this software.
Exported model bundles instead carry an instance of *this* class, built once
at export time from the original object's same 6 fields, so that loading a
bundle never needs to import `drugwm.kg_prior`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class KGGraphArtifacts:
    branch_names: list[str]
    drug_ids: list[int]
    drug_to_local: dict[int, int]
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    coverage: pd.DataFrame
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_original(cls, original: Any) -> "KGGraphArtifacts":
        return cls(
            branch_names=list(getattr(original, "branch_names", [])),
            drug_ids=[int(x) for x in getattr(original, "drug_ids", [])],
            drug_to_local={int(k): int(v) for k, v in getattr(original, "drug_to_local", {}).items()},
            node_table=getattr(original, "node_table", pd.DataFrame()).copy(),
            edge_table=getattr(original, "edge_table", pd.DataFrame()).copy(),
            coverage=getattr(original, "coverage", pd.DataFrame()).copy(),
        )
