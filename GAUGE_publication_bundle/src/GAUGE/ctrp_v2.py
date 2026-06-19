from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path


CTRPDIR = Path("/mnt/raid5/xujing/KG/KG_GAUGE_PublicData/waibu/CTRP/v2")
_SIBLING_HELPERS = (
    Path("/mnt/raid5/xujing/KG/benchmarking/02_drug_split_CTRP_v2/scripts/prepare_ctrp_v2.py"),
    Path("/mnt/raid5/xujing/KG/benchmarking/01_random_cell_split_CTRP_v2/scripts/prepare_ctrp_v2.py"),
)


def _load_sibling_helper():
    for helper_path in _SIBLING_HELPERS:
        if not helper_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("_GAUGE_ctrp_v2_prepare", helper_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helper = getattr(module, "ensure_ctrp_v2_materialized", None)
        if callable(helper):
            return helper
    return None


def ensure_ctrp_v2_materialized(force: bool = False) -> Path:
    os.environ.setdefault("GAUGE_CTRP_DIR", str(CTRPDIR))
    helper = _load_sibling_helper()
    if helper is None:
        expected = (
            CTRPDIR / "CTRPv1.csv",
            CTRPDIR / "gene_expression.csv",
            CTRPDIR / "drug_names.csv",
            CTRPDIR / "drug_smiles.csv",
        )
        if force or not all(path.exists() for path in expected):
            raise FileNotFoundError(
                "No CTRP v2 materialization helper was found and the expected CTRP tables are missing."
            )
    else:
        helper(force=force)
    return CTRPDIR


@lru_cache(maxsize=1)
def ensure_ctrp_v2_runtime() -> Path:
    return ensure_ctrp_v2_materialized()
