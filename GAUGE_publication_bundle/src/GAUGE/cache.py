from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


CACHE_SCHEMA_VERSION = 1


def file_signature(path: Path) -> dict[str, Any]:
    path = Path(path)
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def files_signature(paths: list[Path]) -> list[dict[str, Any]]:
    return [file_signature(Path(p)) for p in paths if Path(p).exists()]


def cache_key(payload: dict[str, Any]) -> str:
    body = {"cache_schema_version": CACHE_SCHEMA_VERSION, **payload}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


@dataclass
class CacheManager:
    cache_dir: Path
    use_cache: bool = True
    rebuild_cache: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path(self, namespace: str, key: str, name: str) -> Path:
        return self.cache_dir / namespace / key / name

    def load_pickle(self, namespace: str, key: str, name: str) -> Any | None:
        path = self.path(namespace, key, name)
        hit = bool(self.use_cache and not self.rebuild_cache and path.exists())
        self.events.append(
            {
                "namespace": namespace,
                "cache_key": key,
                "name": name,
                "path": str(path),
                "event": "hit" if hit else "miss",
            }
        )
        if not hit:
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def save_pickle(self, namespace: str, key: str, name: str, value: Any) -> Path:
        path = self.path(namespace, key, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(value, f)
        self.events.append(
            {
                "namespace": namespace,
                "cache_key": key,
                "name": name,
                "path": str(path),
                "event": "saved",
            }
        )
        return path

    def write_reports(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        current = pd.DataFrame(
            self.events,
            columns=["namespace", "cache_key", "name", "path", "event"],
        )
        hits_path = out_dir / "cache_hits.csv"
        if hits_path.exists():
            try:
                current = pd.concat([pd.read_csv(hits_path), current], ignore_index=True)
            except pd.errors.EmptyDataError:
                pass
        current.to_csv(hits_path, index=False)
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_dir": str(self.cache_dir),
            "use_cache": bool(self.use_cache),
            "rebuild_cache": bool(self.rebuild_cache),
            "events": current.to_dict(orient="records"),
        }
        with (out_dir / "cache_manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
