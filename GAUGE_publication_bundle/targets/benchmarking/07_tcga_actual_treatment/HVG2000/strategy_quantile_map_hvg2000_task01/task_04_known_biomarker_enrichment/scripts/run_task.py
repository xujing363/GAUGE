from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Delegate to the shared implementation in scripts/task_04_known_biomarker_enrichment.py.
from task_04_known_biomarker_enrichment import main


if __name__ == "__main__":
    main(["--task-dir", str(Path(__file__).resolve().parents[1]), *sys.argv[1:]])
