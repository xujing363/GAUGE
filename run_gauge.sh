#!/usr/bin/env bash
# Launch the GAUGE desktop/web application.
#
# Usage:
#   ./run_gauge.sh                 # launch on http://localhost:8501
#   GAUGE_PORT=8888 ./run_gauge.sh # launch on a custom port
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${GAUGE_PORT:-8501}"

# The bundled GAUGE conda environment ships a newer libstdc++ than some host
# systems. Exporting this *before* Python starts is the robust fix (see
# gauge_core/_drugwm_path.py for the in-process fallback used by tests).
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$SCRIPT_DIR"
exec python3 -m streamlit run app/Home.py --server.port "$PORT" --server.headless true
