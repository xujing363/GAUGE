"""Make the vendored GAUGE model code (internal package name ``drugwm``)
importable, plus this software's top-level ``kg_types`` module used by the
pickle-compatibility layer. Importing this module has the side effect of
mutating ``sys.path``; every other gauge_core module imports it first.

By default this points at the *vendored* minimal ``drugwm`` subset shipped
inside this software (model.py/features.py/planner.py/explainability.py
only -- no training code, no sqlite3-based ChEMBL parsing, no absolute
paths), which is what makes the exported model bundles fully portable.
Set ``GAUGE_DRUGWM_REPO`` to point at a full drugwm checkout instead (e.g.
for re-exporting bundles from a newly trained checkpoint via
``scripts/export_model_bundle.py``, which manages its own sys.path and does
not import this module).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SOFTWARE_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_DRUGWM_REPO = Path(os.environ["GAUGE_DRUGWM_REPO"]) if os.environ.get("GAUGE_DRUGWM_REPO") else _VENDOR_DIR

for _p in (str(_SOFTWARE_ROOT), str(_DRUGWM_REPO), str(_VENDOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Defensive only: the vendored `drugwm` subset has no sqlite3 dependency, so
# this should no longer be needed in normal operation. Kept in case a
# developer points GAUGE_DRUGWM_REPO at the full original repository (whose
# drugwm.kg_prior imports sqlite3, which can collide with this conda env's
# newer libstdc++ -- see scripts/export_model_bundle.py for the same fix).
_conda_lib = Path(sys.prefix) / "lib"
if _conda_lib.is_dir():
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    if str(_conda_lib) not in _existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{_conda_lib}:{_existing}" if _existing else str(_conda_lib)
    _libstdcxx = _conda_lib / "libstdc++.so.6"
    if _libstdcxx.exists():
        try:
            import ctypes

            ctypes.CDLL(str(_libstdcxx), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
