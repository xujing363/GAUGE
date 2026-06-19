#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$THIS_DIR/.." && pwd)"

export KGPUB_ROOT="$BUNDLE_ROOT"
export KGPUB_TARGETS="$BUNDLE_ROOT/targets"
export KGPUB_SRC="$BUNDLE_ROOT/src"

# Python import root for `import GAUGE`
export KGPUB_PY_ROOT="${KGPUB_PY_ROOT:-$KGPUB_SRC}"
# KG-like root for scripts that reference /benchmarking, /Combined, /Peru, /DrugDesign_Sec
export KGPUB_KG_ROOT="${KGPUB_KG_ROOT:-/mnt/raid5/xujing/KG}"
# External dataset root (TCGA/CTRdb etc.)
export KGPUB_DATA_ROOT="${KGPUB_DATA_ROOT:-/mnt/raid5/xujing/Agent/Datasets}"
# Optional PRISM patch roots
export KGPUB_PRISM_PATCH_ROOT="${KGPUB_PRISM_PATCH_ROOT:-/mnt/raid5/xujing/KG/PRISM/Secondary}"

export PYTHONPATH="$KGPUB_PY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
