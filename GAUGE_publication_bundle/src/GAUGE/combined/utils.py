from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import ensure_dir, normalize_name, write_json


@dataclass(frozen=True)
class SourceRunArtifacts:
    run_dir: Path
    model: Any
    artifacts: Any
    manifest: dict[str, Any]
    contract: dict[str, Any]


def ensure_source_run_dir(source_run_dir: Path) -> Path:
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    required = ["model.pt", "artifacts.pkl", "benchmark_contract.json", "manifest.json"]
    missing = [name for name in required if not (source_run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Combined source_run_dir is missing required files: {', '.join(missing)} at {source_run_dir}"
        )
    return source_run_dir


def make_output_dir(root: Path, name: str = "Combined") -> Path:
    return ensure_dir(Path(root) / name)


def read_excel_safely(path: Path, sheet_name: str | int | None = 0, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    except ValueError:
        return pd.DataFrame()


def canonical_pair_key(drug_a: str, drug_b: str) -> str:
    a = normalize_name(drug_a)
    b = normalize_name(drug_b)
    if a <= b:
        return f"{a}||{b}"
    return f"{b}||{a}"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
