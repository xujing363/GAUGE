from __future__ import annotations

import json
import math
import shutil
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def normalize_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\b(hydrochloride|mesylate|malate|sodium|potassium|phosphate|acetate|sulfate)\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def copy_config_snapshot(source: Path, target_dirs: Iterable[Path]) -> list[Path]:
    source = Path(source)
    copied: list[Path] = []
    for target_dir in target_dirs:
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def safe_corr(a: Iterable[float], b: Iterable[float], method: str = "pearson") -> float:
    s = pd.DataFrame({"a": list(a), "b": list(b)}).dropna()
    if len(s) < 2 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return float("nan")
    return float(s["a"].corr(s["b"], method=method))


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x
