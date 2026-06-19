from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GAUGE.benchmark_cli import main

BENCH_DIR = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = "configs/explainability_smoke_2000.yaml"
SMOKE_RUN_NAME = "explainability_smoke_2000_hvg2000"
SMOKE_MAX_ROWS = "2000"


def build_argv(device: str, extra_args: list[str] | None = None) -> list[str]:
    return [
        "--benchmark-dir",
        str(BENCH_DIR),
        "--config",
        SMOKE_CONFIG,
        "--device",
        device,
        *(extra_args or []),
        "--max-rows",
        SMOKE_MAX_ROWS,
        "--run-name",
        SMOKE_RUN_NAME,
        "--gdsc-source-mode",
        "v2",
        "run",
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_explainability_smoke_2000.py cuda:N|cpu [benchmark args]")
    main(build_argv(sys.argv[1], sys.argv[2:]))
