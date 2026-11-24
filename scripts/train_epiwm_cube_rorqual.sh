#!/bin/bash

#SBATCH --job-name=epiwm-cube
#SBATCH --account=rrg-bengioy-ad   # <-- EDIT: your DRAC allocation (rrg-.../def-...)
#SBATCH --array=0                  # ARMS x WEIGHTS x SEEDS (see below). Default grid: one task, latent, lambda 0.1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12         # 6 dataloader workers + main proc; the evaluation steps 50 MuJoCo envs
#SBATCH --mem=64G
#SBATCH --time=16:00:00            # generous; see "Budget". train.py cannot resume, so no --requeue
#SBATCH -o outs/%x-%A_%a.out       # relative to the submit dir; `mkdir -p outs` in the repo root first

# EpiWM on Cube (OGBench cube_single, LeWM's dataset) on DRAC Rorqual. This is scripts/train_epiwm_rorqual.sh
# with ENV_NAME=cube and Cube's resources: same grid variables, same options, same outputs. Each task trains with
# worldmodel/train.py at the repo's Cube settings (data=ogb, batch 128, 60k steps) and then evaluates with LeWM's
# CEM planner on 50 and 200 starts (the 200 in 4 chunks of 50, as the repo does for Cube).
#
# Default grid: the latent-dynamics score at lambda = 0.1, the value that matched the frame score on TwoRoom
# (3 seeds: 100/99, 98/97, 100/99 against 100/100). The repo's frame-score lambda differs per environment
# (0.03 TwoRoom and Cube, 0.1 Push-T, 0.3 Reacher), so 0.1 is a starting point, not a tuned value: the first
# minute of the log prints S0, and lambda / S0 was 1.9e-4 for the good TwoRoom runs.
#
# One-time setup on a LOGIN node, after `git pull`:
#   bash scripts/setup_rorqual_env.sh               # re-run: adds the packages the Cube simulator imports
#   bash scripts/download_lewm_data_rorqual.sh cube # 46 GB download -> 102 GB $SCRATCH/stablewm/datasets/ogbench/cube_single_expert.h5
#   mkdir -p outs
# Submit from the repo root:
#   # 1. end-to-end check first (about 20 min incl. staging): 200 training steps, then a 50-start evaluation.
#   #    It exercises the dataset, the GPU training path, MuJoCo rendering on the node and the planner.
#   sbatch --time=1:00:00 --export=ALL,STEPS=200,RUN_TAG=smoke,EVAL_SETS=50 scripts/train_epiwm_cube_rorqual.sh
#   # 2. the run
#   sbatch scripts/train_epiwm_cube_rorqual.sh
#   # lambda sweep (3 tasks), or our own frame-score control at the published lambda:
#   sbatch --array=0-2 --export=ALL,WEIGHTS="0.03 0.1 0.3" scripts/train_epiwm_cube_rorqual.sh
#   sbatch --export=ALL,ARMS="frame",WEIGHTS="0.03" scripts/train_epiwm_cube_rorqual.sh
#   # print the commands without running anything (login node):
#   DRY_RUN=1 bash scripts/train_epiwm_cube_rorqual.sh
#
# Budget: 60k steps. TwoRoom ran at 5.5 steps/s on an H100, which would make this about 3 h, but Cube's
# frames come from a 102 GB file and its speed has not been measured: read `it_s` in the log after a few
# minutes (60000 / it_s seconds). Staging the dataset to the node's local disk adds a few minutes up front.
# Runs are named cube_<arm>_s<seed>_w<lambda>; follow them with scripts/sweep_status.sh. Published numbers for
# comparison (paper-50 / n=200): frame score 70.7 / 73.2 (3 seeds), SIGReg retrain 73 / 64.0 (2 seeds).

export ENV_NAME=cube
export ARMS="${ARMS:-latent}"
export WEIGHTS="${WEIGHTS:-${WEIGHT:-0.1}}"
REPO_DIR="${REPO_DIR:-$HOME/epijepa}"
source "$REPO_DIR/scripts/train_epiwm_rorqual.sh"
