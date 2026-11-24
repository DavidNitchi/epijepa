#!/bin/bash

#SBATCH --job-name=epiwm-tworoom
#SBATCH --account=rrg-bengioy-ad   # <-- EDIT: your DRAC allocation (rrg-.../def-...)
#SBATCH --array=0-1                # ARMS x SEEDS, arm-major (see below). Default grid:
                                   # 0 = frame (EpiWM as published, the control), 1 = latent
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8          # 6 dataloader workers (config num_workers) + main proc
#SBATCH --mem=48G
#SBATCH --time=12:00:00            # generous; see "Budget". train.py cannot resume, so no --requeue
#SBATCH -o outs/%x-%A_%a.out       # relative to the submit dir; run `mkdir -p outs` in the repo root once
                                   # before the first submit or SLURM cannot open the log and the job dies

# EpiWM on TwoRoom (LeWM's cheapest environment) on DRAC Rorqual: one array task = one (arm, seed), trained
# with worldmodel/train.py at the repo's published TwoRoom settings (batch 128, 30k steps, lambda 0.03,
# config/train/lewm.yaml otherwise) and then evaluated with LeWM's CEM planner (scripts/eval_epiwm_rorqual.sh).
#
# Arms (what the epiplexity score's ridge readout predicts from what; see config/train/lewm.yaml):
#   frame         random CNN features of x_t        -> z_t            EpiWM as published: the control
#   latent        random MLP features of (z_t, a_t) -> z_{t+1}        the latent-dynamics score
#   latent_delta  random MLP features of (z_t, a_t) -> z_{t+1} - z_t
#   next          random CNN features of x_{t+1}    -> z_t
#   sigreg        LeWM's own regulariser (model=sigreg, weight 0.09)
#
# One-time setup on a LOGIN node (compute nodes have no internet):
#   cd $HOME/epijepa
#   bash scripts/setup_rorqual_env.sh          # uv venv at .venv + CPU smoke test
#   bash scripts/download_tworoom_rorqual.sh   # $SCRATCH/stablewm/datasets/tworoom.h5 (12.8 GB)
#   mkdir -p outs
# Submit from the repo root:
#   sbatch scripts/train_epiwm_rorqual.sh                              # frame + latent, seed 3072
#   sbatch --array=0-5 --export=ALL,SEEDS="3072 2 3" scripts/train_epiwm_rorqual.sh       # 2 arms x 3 seeds
#   sbatch --array=0-3 --export=ALL,ARMS="frame latent latent_delta sigreg" scripts/train_epiwm_rorqual.sh
#   # lambda sweep: the grid is ARMS x WEIGHTS x SEEDS, so 2 arms x 4 lambdas = 8 tasks. Runs are named
#   # tw_<arm>_s<seed>_w<lambda> (no suffix at the default 0.03); follow them with scripts/sweep_status.sh
#   sbatch --array=0-7 --export=ALL,ARMS="latent latent_delta",WEIGHTS="0.1 0.3 1 3" scripts/train_epiwm_rorqual.sh
#   # 200-step end-to-end check of the GPU path before spending real hours (no eval):
#   sbatch --time=0:30:00 --export=ALL,STEPS=200,RUN_TAG=smoke,RUN_EVAL=0 scripts/train_epiwm_rorqual.sh
#   # print the commands for task 1 without running anything (works on a login node):
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash scripts/train_epiwm_rorqual.sh
# Extra key=value overrides after the script name pass straight through to train.py, e.g.
#   sbatch scripts/train_epiwm_rorqual.sh loss.epi.detach_inputs=true
#
# Budget: the repo reports ~5 h for TwoRoom's 30k steps on a GPU shared four ways; a dedicated H100 should be
# well under that, but it has not been timed here. train.py logs `it_s` every 200 steps: after a minute,
# 30000 / it_s seconds is the training time. A task that hits the limit starts over when resubmitted
# (train.py has no resume), so if it_s says it will not fit, cancel and resubmit with a longer --time.
#
# Where things land:
#   $SCRATCH/stablewm/checkpoints/<run>/weights_step<N>.pt + config.json   (every 10k steps and at the end)
#   $SCRATCH/stablewm/<run>/tworoom_results.txt + episode_*.mp4            (written by eval.py)
#   <repo>/worldmodel/runs/<run>/train_log.json + config.yaml              (loss, epi_ratio, erank, val)
#   <repo>/outs/tworoom_success.tsv                                        (one line per evaluation)
# A task whose final checkpoint already exists skips training and only evaluates.

# --- grid: task -> (arm, lambda, seed), arm-major then lambda ------------------
ARMS=(${ARMS:-frame latent})
WEIGHTS=(${WEIGHTS:-${WEIGHT:-0.03}})   # lambda(s) for the epi arms: loss = pred - lambda * S / S0. 0.03 is the
                                   # repo's TwoRoom value, tuned for the frame score. S0 differs between
                                   # scores (frame 257, latent 519 on TwoRoom), so compare arms on lambda / S0.
SEEDS=(${SEEDS:-3072})             # 3072 is LeWM's default; the repo's other seeds are 2 and 3
STEPS="${STEPS:-30000}"            # the cosine schedule spans max_steps, so this is part of the recipe
NUM_WORKERS="${NUM_WORKERS:-6}"
IDX=${SLURM_ARRAY_TASK_ID:-0}
NTASKS=$(( ${#ARMS[@]} * ${#WEIGHTS[@]} * ${#SEEDS[@]} ))
if [ "$IDX" -ge "$NTASKS" ]; then
    echo "Array index $IDX >= grid size $NTASKS; nothing to do."
    exit 0
fi
ARM=${ARMS[$((IDX / (${#WEIGHTS[@]} * ${#SEEDS[@]})))]}
WEIGHT=${WEIGHTS[$(((IDX / ${#SEEDS[@]}) % ${#WEIGHTS[@]}))]}
SEED=${SEEDS[$((IDX % ${#SEEDS[@]}))]}
W_TAG=""; if [ "$WEIGHT" != 0.03 ] && [ "$ARM" != sigreg ]; then W_TAG="_w$WEIGHT"; fi
RUN_NAME="tw_${ARM}_s${SEED}${W_TAG}${RUN_TAG:+_$RUN_TAG}"
case "$ARM" in
    sigreg) ARM_ARGS=(model=sigreg loss.reg=sigreg) ;;
    frame|next|latent|latent_delta)
        ARM_ARGS=(model=epi loss.reg=epi "loss.epi.target=$ARM" "loss.epi.weight=$WEIGHT") ;;
    *) echo "ERROR: unknown arm '$ARM' (frame next latent latent_delta sigreg)"; exit 1 ;;
esac

echo "$(date) task=$IDX arm=$ARM lambda=$WEIGHT seed=$SEED steps=$STEPS run=$RUN_NAME"

# Don't use `-u`: virtualenv/activate scripts reference unbound vars.
set -eo pipefail

run() {  # echo the command; skip it under DRY_RUN=1
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi
}

# --- Environment (uv venv built by scripts/setup_rorqual_env.sh) ---------------
# Activate directly; no `module load python/cuda` (uv ships its own Python and the torch wheel its own CUDA).
REPO_DIR="${REPO_DIR:-$HOME/epijepa}"
VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
source "$VENV_PATH/bin/activate"

# stable-worldmodel keeps datasets/ and checkpoints/ under STABLEWM_HOME; keep it on $SCRATCH.
export STABLEWM_HOME="${STABLEWM_HOME:-$SCRATCH/stablewm}"
# Compute nodes have no internet: nothing is fetched (the ViT is built from its config), make that explicit.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export MUJOCO_GL=egl
export HDF5_USE_FILE_LOCKING=FALSE   # Lustre does not support the locks HDF5 takes by default
mkdir -p "$STABLEWM_HOME/checkpoints"

# --- Stage the dataset to node-local NVMe ($SLURM_TMPDIR) ----------------------
# Training does random reads over the 12.8 GB HDF5 from 6 workers; those are slow and jittery on Lustre.
# The copy is one sequential read. Falls back to $SCRATCH if there is no room or the copy fails.
H5_NAME="tworoom.h5"
SRC_H5="$STABLEWM_HOME/datasets/$H5_NAME"
if [ ! -s "$SRC_H5" ]; then
    echo "ERROR: $SRC_H5 not found. Run scripts/download_tworoom_rorqual.sh on a login node first."
    exit 1
fi
export DATA_ROOT="$STABLEWM_HOME"    # root holding datasets/<file>; the eval script reads it too
if [ "${DRY_RUN:-0}" != 1 ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    need_kb=$(du -k "$SRC_H5" | cut -f1 || true)
    avail_kb=$(df -Pk "$SLURM_TMPDIR" 2>/dev/null | awk 'NR==2 {print $4}' || true)
    if [[ "$need_kb" =~ ^[0-9]+$ ]] && [[ "$avail_kb" =~ ^[0-9]+$ ]] \
       && [ "$avail_kb" -gt "$(( need_kb + need_kb / 10 ))" ]; then
        t0=$SECONDS
        mkdir -p "$SLURM_TMPDIR/stablewm/datasets"
        if cp "$SRC_H5" "$SLURM_TMPDIR/stablewm/datasets/$H5_NAME"; then
            export DATA_ROOT="$SLURM_TMPDIR/stablewm"
            echo "$(date) staged dataset to $DATA_ROOT in $(( SECONDS - t0 ))s"
        else
            echo "$(date) WARNING: staging copy failed; reading from $STABLEWM_HOME"
            rm -f "$SLURM_TMPDIR/stablewm/datasets/$H5_NAME"
        fi
    else
        echo "$(date) not enough room in \$SLURM_TMPDIR (need ${need_kb:-?}K, free ${avail_kb:-?}K); reading from $STABLEWM_HOME"
    fi
else
    echo "$(date) not staging (DRY_RUN or no \$SLURM_TMPDIR); reading from $STABLEWM_HOME"
fi

# --- Train ---------------------------------------------------------------------
cd "$REPO_DIR/worldmodel"
FINAL_CKPT="$STABLEWM_HOME/checkpoints/$RUN_NAME/weights_step${STEPS}.pt"
if [ -s "$FINAL_CKPT" ]; then
    echo "$(date) $FINAL_CKPT exists; skipping training."
else
    # data.dataset.name as an absolute path makes train.py read the staged copy; checkpoints still go to
    # $STABLEWM_HOME/checkpoints/<run_name>.
    run python train.py data=tworoom "${ARM_ARGS[@]}" \
        max_steps="$STEPS" seed="$SEED" run_name="$RUN_NAME" num_workers="$NUM_WORKERS" \
        "data.dataset.name=$DATA_ROOT/datasets/$H5_NAME" \
        "$@"
fi

# --- Evaluate (same GPU, same staged dataset) ----------------------------------
if [ "${RUN_EVAL:-1}" = 1 ]; then
    RUN_NAME="$RUN_NAME" STEPS="$STEPS" bash "$REPO_DIR/scripts/eval_epiwm_rorqual.sh"
fi
echo "$(date) $RUN_NAME finished."
