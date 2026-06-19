#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$KGPUB_TARGETS/Peru/A" && bash run_all.sh "$@"
cd "$KGPUB_TARGETS/Peru/B" && bash run_all.sh "$@"
cd "$KGPUB_TARGETS/Peru/C" && bash run_all.sh "$@"
