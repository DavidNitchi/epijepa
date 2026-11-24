#!/bin/bash

#SBATCH --job-name=epiwm-eval
#SBATCH --account=rrg-bengioy-ad   # <-- EDIT: your DRAC allocation (rrg-.../def-...)
#SBATCH --gpus-per-node=h100:1     # the CEM planner runs the world model on the GPU; Cube also renders on it
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                  # up to 200 TwoRoom (or 50 Cube) envs in parallel, each holding 224x224 frames
#SBATCH --time=3:00:00
#SBATCH -o outs/%x-%j.out          # relative to the submit dir; `mkdir -p outs` first

# Planning evaluation of one checkpoint with LeWM's CEM planner (worldmodel/eval.py, unmodified protocol: starts
# sampled with seed 42, goal 25 steps ahead, budget 50 steps). scripts/train_epiwm_rorqual.sh calls this at the
# end of every training task; submit it on its own to re-evaluate or to add the n=500 set:
#   sbatch --export=ALL,RUN_NAME=tw_latent_s3072_w0.1 scripts/eval_epiwm_rorqual.sh
#   sbatch --export=ALL,RUN_NAME=tw_latent_s3072_w0.1,EVAL_SETS="500" scripts/eval_epiwm_rorqual.sh
#   sbatch --export=ALL,RUN_NAME=tw_latent_s3072_w0.1,STEPS=20000 scripts/eval_epiwm_rorqual.sh   # earlier checkpoint
#   sbatch --export=ALL,ENV_NAME=cube,RUN_NAME=cube_latent_s3072_w0.1 scripts/eval_epiwm_rorqual.sh
#   # only check that the environment builds and renders on a GPU node (no checkpoint needed):
#   sbatch --time=0:10:00 --export=ALL,ENV_NAME=cube,PREFLIGHT_ONLY=1 scripts/eval_epiwm_rorqual.sh
#
# ENV_NAME: tworoom (default) or cube. EVAL_SETS are the repo's evaluation sets: 50 ("paper-50") and 200. A set
# larger than the environment's parallel limit (TwoRoom 200, Cube 50: the repo chunks Cube for host RAM) is
# split into chunks of 50 starts with +eval_chunk=i/n, exactly as the repo does (pooling the chunks reproduces
# the unchunked start set), and the chunk success rates are averaged.
# Published numbers to compare against (worldmodel/analysis/scores/scorecard.md), paper-50 / n=200 / n=500:
#   TwoRoom 30k:  EpiWM (frame) lambda=0.03, 3 seeds  100 / 100 / 99.9     SIGReg retrain, 2 seeds  89 / 85.5 / 87.9
#   Cube 60k:     EpiWM (frame) lambda=0.03, 3 seeds  70.7 / 73.2 / 71.5   SIGReg retrain, 2 seeds  73 / 64.0 / 65.5
#
# Rendering: Cube is a MuJoCo scene rendered off-screen. The script first builds the environment and renders one
# frame, trying MUJOCO_GL=egl (GPU) and then osmesa (software); the first that works is used for the evaluation.
# Set MUJOCO_GL_CANDIDATES to change the list. TwoRoom is 2-D and does not use MuJoCo.
#
# Output: eval.py appends config + metrics to $STABLEWM_HOME/<run>/<env>_results.txt and writes the episode
# videos next to it; this script also appends "run  checkpoint  n  success%" to <repo>/outs/<env>_success.tsv.

# Don't use `-u`: virtualenv/activate scripts reference unbound vars.
set -eo pipefail

ENV_NAME="${ENV_NAME:-tworoom}"
case "$ENV_NAME" in
    tworoom) EVAL_CFG=tworoom.yaml; RESULTS_FILE=tworoom_results.txt;  H5_REL="tworoom.h5"
             DEFAULT_STEPS=30000; DEFAULT_PARALLEL=200 ;;
    cube)    EVAL_CFG=cube.yaml;    RESULTS_FILE=ogb_cube_results.txt; H5_REL="ogbench/cube_single_expert.h5"
             DEFAULT_STEPS=60000; DEFAULT_PARALLEL=50 ;;
    *) echo "ERROR: unknown ENV_NAME '$ENV_NAME' (tworoom cube)"; exit 1 ;;
esac
STEPS="${STEPS:-$DEFAULT_STEPS}"
EVAL_SETS="${EVAL_SETS:-50 200}"
CHUNK=50                             # starts per chunk for sets larger than MAX_PARALLEL
MAX_PARALLEL="${MAX_PARALLEL:-$DEFAULT_PARALLEL}"

run() {  # echo the command; skip it under DRY_RUN=1
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi
}

# --- Environment (same as the training script) ---------------------------------
REPO_DIR="${REPO_DIR:-$HOME/epijepa}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
source "$VENV_PATH/bin/activate"
export STABLEWM_HOME="${STABLEWM_HOME:-$SCRATCH/stablewm}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO_DIR/worldmodel"

# --- Check that the environment builds and renders here -------------------------
# Builds the eval config's world with one env and renders one frame, per GL backend, and keeps the first
# backend that works. This is the part of the evaluation that depends on the node (drivers, EGL libraries).
preflight() {
    python - "$EVAL_CFG" <<'PYEOF'
import os, sys
import numpy as np
from omegaconf import OmegaConf
import stable_worldmodel as swm

world = OmegaConf.to_container(OmegaConf.load(f"config/eval/{sys.argv[1]}").world, resolve=False)
world.update(num_envs=1, max_episode_steps=100)
w = swm.World(**world, image_shape=(224, 224))
w.reset()
px = np.asarray(w.infos["pixels"])
assert px.size and px.std() > 0, f"rendered frame is blank: shape {px.shape}"
print(f"environment OK: {world['env_name']} renders with MUJOCO_GL={os.environ.get('MUJOCO_GL')}, pixels {px.shape} {px.dtype}")
PYEOF
}
if [ "${DRY_RUN:-0}" != 1 ]; then
    GL_OK=""
    for gl in ${MUJOCO_GL_CANDIDATES:-egl osmesa}; do
        if MUJOCO_GL="$gl" preflight; then GL_OK="$gl"; break; fi
        echo "$(date) environment check failed with MUJOCO_GL=$gl (traceback above)."
    done
    if [ -z "$GL_OK" ]; then
        echo "ERROR: $ENV_NAME does not build or render on $(hostname). If the traceback is about EGL/OpenGL, this"
        echo "       node has no usable off-screen GL for MuJoCo; if it is a missing module, re-run"
        echo "       scripts/setup_rorqual_env.sh on a login node."
        exit 2
    fi
    export MUJOCO_GL="$GL_OK"
fi
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then exit 0; fi

RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. --export=ALL,RUN_NAME=tw_latent_s3072_w0.1}"
# Dataset root (holds datasets/<file>). The training script exports the node-local copy; on its own the
# evaluation reads only a few hundred rows plus the action column, so $SCRATCH is fine.
DATA_ROOT="${DATA_ROOT:-$STABLEWM_HOME}"

POLICY="$RUN_NAME/weights_step${STEPS}.pt"
CKPT="$STABLEWM_HOME/checkpoints/$POLICY"
RESULTS="$STABLEWM_HOME/$RUN_NAME/$RESULTS_FILE"
SUMMARY="$REPO_DIR/outs/${ENV_NAME}_success.tsv"
if [ "${DRY_RUN:-0}" != 1 ]; then
    [ -s "$CKPT" ] || { echo "ERROR: checkpoint $CKPT not found."; exit 1; }
    [ -s "$DATA_ROOT/datasets/$H5_REL" ] || { echo "ERROR: $DATA_ROOT/datasets/$H5_REL not found."; exit 1; }
fi
mkdir -p "$REPO_DIR/outs"
HYDRA_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hydra_${RUN_NAME}_$$"   # keep hydra's per-run dirs out of the repo

for N in $EVAL_SETS; do
    if [ "$N" -le "$MAX_PARALLEL" ]; then NCHUNK=1; else NCHUNK=$(( (N + CHUNK - 1) / CHUNK )); fi
    echo "$(date) evaluating $POLICY on $N starts ($NCHUNK chunk(s))"
    for (( i = 0; i < NCHUNK; i++ )); do
        CHUNK_ARG=()
        if [ "$NCHUNK" -gt 1 ]; then CHUNK_ARG=("+eval_chunk=$i/$NCHUNK"); fi
        run python eval.py "--config-name=$EVAL_CFG" "policy=$POLICY" "eval.num_eval=$N" \
            "cache_dir=$DATA_ROOT" "hydra.run.dir=$HYDRA_DIR/n${N}_$i" "${CHUNK_ARG[@]}"
    done
    if [ "${DRY_RUN:-0}" != 1 ]; then
        # eval.py appends one "metrics: {... 'success_rate': X ...}" line per call: average this set's chunks.
        RATE=$(grep -o "'success_rate': [0-9.]*" "$RESULTS" | tail -n "$NCHUNK" | awk '{s += $2} END {printf "%.1f", s / NR}')
        echo "$(date) RESULT run=$RUN_NAME checkpoint=step$STEPS n=$N success=$RATE%"
        printf '%s\tstep%s\t%s\t%s\t%s\n' "$RUN_NAME" "$STEPS" "$N" "$RATE" "$(date +%F)" >> "$SUMMARY"
    fi
done
