#!/bin/bash

# Fetch LeWM's TwoRoom dataset (Hugging Face: quentinll/lewm-tworooms) for worldmodel/train.py and eval.py.
# Run on a LOGIN node (compute nodes have no internet), after scripts/setup_rorqual_env.sh:
#   cd $HOME/epijepa && bash scripts/download_tworoom_rorqual.sh
#
# The repo ships one file, tworoom.tar.zst (3.4 GB), holding a single tworoom.h5 (12.8 GB). It ends up at
#   $STABLEWM_HOME/datasets/tworoom.h5        (STABLEWM_HOME defaults to $SCRATCH/stablewm)
# which is where train.py (data=tworoom) and eval.py (--config-name=tworoom.yaml) look for it. Needs ~17 GB
# free while unpacking; the tarball is deleted afterwards unless KEEP_TAR=1. Safe to re-run: a finished
# download or extraction is skipped, and an interrupted download resumes.
#
# No model download is needed: the ViT encoder is built from its config (pretrained: false) and training
# starts from scratch.

set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
STABLEWM_HOME="${STABLEWM_HOME:-${SCRATCH:-$PWD}/stablewm}"
DEST="$STABLEWM_HOME/datasets"
URL="${TWOROOM_URL:-https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst}"
TAR="$DEST/tworoom.tar.zst"
H5="$DEST/tworoom.h5"

[ -x "$VENV_PATH/bin/python" ] || { echo "ERROR: no venv at $VENV_PATH; run scripts/setup_rorqual_env.sh first."; exit 1; }

# Outbound HTTPS from Rorqual login nodes goes through httpproxy; harmless elsewhere.
if command -v module >/dev/null 2>&1; then
    module load httpproxy 2>/dev/null || true
fi

mkdir -p "$DEST"
if [ -s "$H5" ]; then
    echo "have    $H5 ($(du -h "$H5" | cut -f1))"
else
    if [ ! -s "$TAR" ]; then
        echo "fetch   $URL"
        curl -fL --retry 5 --retry-delay 10 -C - -o "$TAR.part" "$URL"
        mv "$TAR.part" "$TAR"
    fi
    echo "unpack  $TAR ($(du -h "$TAR" | cut -f1)) -> $H5"
    # Streamed through python-zstandard so no system zstd is needed. Only *.h5 members are written, each
    # under its base name, to a .part file that is renamed once complete.
    "$VENV_PATH/bin/python" - "$TAR" "$DEST" <<'PYEOF'
import os, shutil, sys, tarfile
import zstandard

src, dest = sys.argv[1:3]
found = []
with open(src, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z, tarfile.open(fileobj=z, mode="r|") as tar:
    for m in tar:
        if not (m.isfile() and m.name.endswith(".h5")):
            continue
        out = os.path.join(dest, os.path.basename(m.name))
        with tar.extractfile(m) as r, open(out + ".part", "wb") as w:
            shutil.copyfileobj(r, w, 16 << 20)
        os.replace(out + ".part", out)
        found.append((out, m.size))
        print(f"wrote   {out} ({m.size / 1e9:.2f} GB)", flush=True)
if not found:
    sys.exit("ERROR: no .h5 member in the archive")
PYEOF
    [ -s "$H5" ] || { echo "ERROR: $H5 missing after extraction."; exit 1; }
    if [ "${KEEP_TAR:-0}" != 1 ]; then rm -f "$TAR"; fi
fi

# Open it once: a truncated file fails here instead of minutes into a GPU job.
"$VENV_PATH/bin/python" - "$H5" <<'PYEOF'
import sys
import hdf5plugin  # noqa: F401  (registers the compression filters the file may use)
import h5py

with h5py.File(sys.argv[1], "r") as f:
    for k in sorted(f.keys()):
        v = f[k]
        print(f"  {k:14s} {getattr(v, 'shape', '')} {getattr(v, 'dtype', '')}")
print("dataset OK:", sys.argv[1])
PYEOF
