#!/bin/bash

# Fetch one of LeWM's datasets from Hugging Face for worldmodel/train.py and eval.py. Run on a LOGIN node
# (compute nodes have no internet), after scripts/setup_rorqual_env.sh:
#   cd $HOME/epijepa
#   bash scripts/download_lewm_data_rorqual.sh tworoom    #  3.4 GB download ->  12.8 GB tworoom.h5
#   bash scripts/download_lewm_data_rorqual.sh cube       # 46.2 GB download -> 101.9 GB ogbench/cube_single_expert.h5
#
# Each repo (quentinll/lewm-tworooms, quentinll/lewm-cube) ships one .tar.zst holding a single .h5. It is
# unpacked to $STABLEWM_HOME/datasets/<path above> (STABLEWM_HOME defaults to $SCRATCH/stablewm), which is where
# train.py (data=tworoom / data=ogb) and eval.py (tworoom.yaml / cube.yaml) look for it. Space needed while
# unpacking is download + unpacked size (Cube: about 150 GB); the tarball is deleted afterwards unless
# KEEP_TAR=1. Safe to re-run: a finished download or extraction is skipped, an interrupted download resumes.
#
# Cube is big: run it inside tmux/screen so a dropped connection does not kill it. Unpacking 102 GB is several
# minutes of CPU; to keep that off the login node, download only and unpack in a small CPU job (the unpack
# step needs no internet):
#   DOWNLOAD_ONLY=1 bash scripts/download_lewm_data_rorqual.sh cube
#   sbatch --account=<your-account> --time=1:00:00 --cpus-per-task=2 --mem=8G -o outs/unpack-cube-%j.out \
#       --wrap="bash $HOME/epijepa/scripts/download_lewm_data_rorqual.sh cube"
#
# No model download is needed for any environment: the ViT encoder is built from its config (pretrained:
# false) and training starts from scratch.

set -eo pipefail

DATASET="${1:-}"
case "$DATASET" in
    tworoom) HF_REPO="quentinll/lewm-tworooms"; TAR_NAME="tworoom.tar.zst";            H5_REL="tworoom.h5" ;;
    cube)    HF_REPO="quentinll/lewm-cube";     TAR_NAME="cube_single_expert.tar.zst"; H5_REL="ogbench/cube_single_expert.h5" ;;
    *) echo "usage: bash scripts/download_lewm_data_rorqual.sh tworoom|cube"; exit 1 ;;
esac

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
STABLEWM_HOME="${STABLEWM_HOME:-${SCRATCH:-$PWD}/stablewm}"
URL="${DATA_URL:-https://huggingface.co/datasets/$HF_REPO/resolve/main/$TAR_NAME}"
H5="$STABLEWM_HOME/datasets/$H5_REL"
DEST="$(dirname "$H5")"
TAR="$DEST/$TAR_NAME"

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
        echo "        free space on $(df -Ph "$DEST" | awk 'NR==2 {print $6 ": " $4}')"
        curl -fL --retry 5 --retry-delay 10 -C - -o "$TAR.part" "$URL"
        mv "$TAR.part" "$TAR"
    fi
    if [ "${DOWNLOAD_ONLY:-0}" = 1 ]; then
        echo "downloaded $TAR ($(du -h "$TAR" | cut -f1)); re-run without DOWNLOAD_ONLY=1 to unpack."
        exit 0
    fi
    echo "unpack  $TAR ($(du -h "$TAR" | cut -f1)) -> $H5"
    # Streamed through python-zstandard so no system zstd is needed. Only *.h5 members are written, each
    # under its base name, to a .part file that is renamed once complete.
    "$VENV_PATH/bin/python" - "$TAR" "$DEST" <<'PYEOF'
import os, sys, tarfile, time
import zstandard

src, dest = sys.argv[1:3]
found, t0 = [], time.time()
with open(src, "rb") as f, zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(f) as z, \
        tarfile.open(fileobj=z, mode="r|") as tar:
    for m in tar:
        if not (m.isfile() and m.name.endswith(".h5")):
            continue
        out = os.path.join(dest, os.path.basename(m.name))
        done, mark = 0, 0
        with tar.extractfile(m) as r, open(out + ".part", "wb") as w:
            while chunk := r.read(16 << 20):
                w.write(chunk)
                done += len(chunk)
                if done - mark >= 10 << 30:  # progress every 10 GiB
                    mark = done
                    print(f"        {done / 1e9:6.1f} / {m.size / 1e9:.1f} GB  ({time.time() - t0:.0f}s)", flush=True)
        if done != m.size:
            sys.exit(f"ERROR: {m.name}: wrote {done} of {m.size} bytes")
        os.replace(out + ".part", out)
        found.append(out)
        print(f"wrote   {out} ({m.size / 1e9:.2f} GB in {time.time() - t0:.0f}s)", flush=True)
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
        print(f"  {k:28s} {getattr(v, 'shape', '')} {getattr(v, 'dtype', '')}")
print("dataset OK:", sys.argv[1])
PYEOF
