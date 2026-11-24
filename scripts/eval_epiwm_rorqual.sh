#!/bin/bash

#SBATCH --job-name=epiwm-eval
#SBATCH --account=rrg-bengioy-ad   # <-- EDIT: your DRAC allocation (rrg-.../def-...)
#SBATCH --gpus-per-node=h100:1     # the CEM planner runs the world model on the GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                  # up to 200 TwoRoom envs in parallel, each holding 224x224 frames
#SBATCH --time=3:00:00
#SBATCH -o outs/%x-%j.out          # relative to the submit dir; `mkdir -p outs` first

# Planning evaluation of one TwoRoom checkpoint with LeWM's CEM planner (worldmodel/eval.py, unmodified
# protocol: starts sampled with seed 42, goal 25 steps ahead, budget 50 steps). scripts/train_epiwm_rorqual.sh
# calls this at the end of every training task; submit it on its own to re-evaluate or to add the n=500 set:
#   sbatch --export=ALL,RUN_NAME=tw_latent_s3072 scripts/eval_epiwm_rorqual.sh
#   sbatch --export=ALL,RUN_NAME=tw_latent_s3072,EVAL_SETS="500" scripts/eval_epiwm_rorqual.sh
#   sbatch --export=ALL,RUN_NAME=tw_latent_s3072,STEPS=20000 scripts/eval_epiwm_rorqual.sh   # an earlier checkpoint
#
# EVAL_SETS are the repo's evaluation sets: 50 ("paper-50") and 200 run in one go; anything larger is split
# into chunks of 50 starts with +eval_chunk=i/n, as the repo does for n=500 (pooling the chunks reproduces
# the unchunked start set exactly), and the chunk success rates are averaged.
# The published numbers to compare against (worldmodel/analysis/scores/scorecard.md), paper-50 / n=200 / n=500:
#   EpiWM lambda=0.03, 30k steps, 3 seeds:  100 / 100 / 99.9      SIGReg retrain, 2 seeds:  89 / 85.5 / 87.9
#
# Output: eval.py appends config + metrics to $STABLEWM_HOME/<run>/tworoom_results.txt and writes the episode
# videos next to it; this script also appends "run  checkpoint  n  success%" to <repo>/outs/tworoom_success.tsv.

# Don't use `-u`: virtualenv/activate scripts reference unbound vars.
set -eo pipefail

RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. --export=ALL,RUN_NAME=tw_latent_s3072}"
STEPS="${STEPS:-30000}"
EVAL_SETS="${EVAL_SETS:-50 200}"
CHUNK=50                             # starts per chunk for sets larger than MAX_PARALLEL
MAX_PARALLEL="${MAX_PARALLEL:-200}"

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
export MUJOCO_GL=egl HDF5_USE_FILE_LOCKING=FALSE
# Dataset root (holds datasets/tworoom.h5). The training script exports the node-local copy; on its own the
# evaluation reads only a few hundred rows, so $SCRATCH is fine.
DATA_ROOT="${DATA_ROOT:-$STABLEWM_HOME}"

POLICY="$RUN_NAME/weights_step${STEPS}.pt"
CKPT="$STABLEWM_HOME/checkpoints/$POLICY"
RESULTS="$STABLEWM_HOME/$RUN_NAME/tworoom_results.txt"
SUMMARY="$REPO_DIR/outs/tworoom_success.tsv"
if [ "${DRY_RUN:-0}" != 1 ]; then
    [ -s "$CKPT" ] || { echo "ERROR: checkpoint $CKPT not found."; exit 1; }
    [ -s "$DATA_ROOT/datasets/tworoom.h5" ] || { echo "ERROR: $DATA_ROOT/datasets/tworoom.h5 not found."; exit 1; }
fi
mkdir -p "$REPO_DIR/outs"
HYDRA_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hydra_${RUN_NAME}_$$"   # keep hydra's per-run dirs out of the repo

cd "$REPO_DIR/worldmodel"
for N in $EVAL_SETS; do
    if [ "$N" -le "$MAX_PARALLEL" ]; then NCHUNK=1; else NCHUNK=$(( (N + CHUNK - 1) / CHUNK )); fi
    echo "$(date) evaluating $POLICY on $N starts ($NCHUNK chunk(s))"
    for (( i = 0; i < NCHUNK; i++ )); do
        CHUNK_ARG=()
        if [ "$NCHUNK" -gt 1 ]; then CHUNK_ARG=("+eval_chunk=$i/$NCHUNK"); fi
        run python eval.py --config-name=tworoom.yaml "policy=$POLICY" "eval.num_eval=$N" \
            "cache_dir=$DATA_ROOT" "hydra.run.dir=$HYDRA_DIR/n${N}_$i" "${CHUNK_ARG[@]}"
    done
    if [ "${DRY_RUN:-0}" != 1 ]; then
        # eval.py appends one "metrics: {... 'success_rate': X ...}" line per call: average this set's chunks.
        RATE=$(grep -o "'success_rate': [0-9.]*" "$RESULTS" | tail -n "$NCHUNK" | awk '{s += $2} END {printf "%.1f", s / NR}')
        echo "$(date) RESULT run=$RUN_NAME checkpoint=step$STEPS n=$N success=$RATE%"
        printf '%s\tstep%s\t%s\t%s\t%s\n' "$RUN_NAME" "$STEPS" "$N" "$RATE" "$(date +%F)" >> "$SUMMARY"
    fi
done
