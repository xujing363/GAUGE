from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from task_02_clinical_response_concordance import main


if __name__ == "__main__":
    main(["--task-dir", str(Path(__file__).resolve().parents[1]), *sys.argv[1:]])
