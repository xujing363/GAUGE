"""Minimal `.env` loader (no extra dependency). Reads KEY=VALUE lines from
`<software_root>/.env` into `os.environ`, without overwriting variables
already set in the real environment (so a real env var always wins)."""
from __future__ import annotations

import os
from pathlib import Path

_SOFTWARE_ROOT = Path(__file__).resolve().parent.parent
_LOADED = False


def load_dotenv() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    env_path = _SOFTWARE_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
