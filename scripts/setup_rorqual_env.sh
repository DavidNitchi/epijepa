#!/bin/bash

# One-time environment bootstrap for EpiWM (worldmodel/) on DRAC Rorqual. Run on a LOGIN node:
#   cd $HOME/epijepa && bash scripts/setup_rorqual_env.sh
# Builds a uv venv at $VENV_PATH (default <repo>/.venv), used by scripts/{train,eval}_epiwm_rorqual.sh
# and scripts/download_tworoom_rorqual.sh. Safe to re-run.
#
# uv ships its own standalone Python, which accepts the ordinary manylinux wheels from PyPI (the cluster's
# module Python does not). So the pinned stack in worldmodel/requirements.txt installs as-is and the torch
# wheel brings its own CUDA runtime: no `module load python` / `module load cuda`, here or in the jobs.
#
# On top of worldmodel/requirements.txt this installs what the code needs but the file does not list:
#   lightning, einops   imported by worldmodel/utils.py and worldmodel/module.py
#   imageio[ffmpeg]     stable-worldmodel imports imageio, and eval.py writes one mp4 per episode, which
#                       fails with "Could not find a backend" without the ffmpeg plugin
#   zstandard           unpacks the LeWM dataset tarball (scripts/download_tworoom_rorqual.sh)

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# Outbound HTTPS from Rorqual login nodes goes through httpproxy; harmless elsewhere.
if command -v module >/dev/null 2>&1; then
    module load httpproxy 2>/dev/null || true
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found on PATH. Install it once on the login node:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh    (then open a new shell)"
    exit 1
fi

if [ ! -x "$VENV_PATH/bin/python" ]; then
    uv venv --python "$PYTHON_VERSION" "$VENV_PATH"
fi
uv pip install --python "$VENV_PATH/bin/python" \
    -r "$REPO_DIR/worldmodel/requirements.txt" \
    lightning einops "imageio[ffmpeg]" zstandard

# Smoke test on the login node's CPU: real model from the training config, every loss.epi.target.
# STABLEWM_HOME only has to exist for the import; nothing is written there by the test.
export STABLEWM_HOME="${STABLEWM_HOME:-${SCRATCH:-$REPO_DIR}/stablewm}"
mkdir -p "$STABLEWM_HOME"
"$VENV_PATH/bin/python" "$REPO_DIR/worldmodel/smoke_test.py"

echo
echo "venv ready at $VENV_PATH"
echo "Next: bash scripts/download_tworoom_rorqual.sh   (login node), then"
echo "      mkdir -p outs && sbatch scripts/train_epiwm_rorqual.sh"
