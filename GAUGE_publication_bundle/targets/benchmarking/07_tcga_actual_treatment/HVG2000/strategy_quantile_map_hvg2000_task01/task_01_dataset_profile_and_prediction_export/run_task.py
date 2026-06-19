from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SCRIPTS = TASK_ROOT / "scripts"
if str(LOCAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from export_task01 import main


if __name__ == "__main__":
    main(["--task-dir", str(Path(__file__).resolve().parent), *sys.argv[1:]])
