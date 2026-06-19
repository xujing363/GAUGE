from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from hvg2000_patch import apply_patches

apply_patches()

from GAUGE.benchmark_cli import main


if __name__ == "__main__":
    main(["--benchmark-dir", str(ROOT), *sys.argv[1:], "evaluate"])
