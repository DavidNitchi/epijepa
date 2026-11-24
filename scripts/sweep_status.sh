#!/bin/bash

# One line per training log: where the run is and whether it is collapsing. Works while jobs are running.
#   bash scripts/sweep_status.sh                 # all outs/epiwm-*.out (TwoRoom and Cube), from the repo root
#   bash scripts/sweep_status.sh path/to/*.out   # specific logs
#
# Columns: lambda and S0 (the score of the untrained model), lambda/S0 x 1e4 (the regulariser's weight per bit,
# the number to compare across arms: the frame control is 0.03 / 257 = 1.17), then the latest training row
# (step, prediction loss, epi_ratio = S / S0, effective rank of the embeddings) and the planning success on
# 50 / 200 starts once the run has been evaluated.
# The success columns are the 50- and 200-start sets. Published frame-score numbers for reference: TwoRoom
# 100 / 100, Cube 70.7 / 73.2 (SIGReg retrain: 89 / 85.5 and 73 / 64.0).
# Collapse check (TwoRoom): the frame control (100% success) was at erank 8 at step 2000, 10 at step 5000 and 34 at the
# end; the collapsed latent run (lambda 0.03, 78-82%) sat at erank 2-3 from step 2000 on. A run still near
# erank 2 at step 5000 is not going to recover: `scancel <jobid>_<task>` it.

shopt -s nullglob
if [ "$#" -gt 0 ]; then LOGS=("$@"); else LOGS=(outs/epiwm-*.out); fi
[ "${#LOGS[@]}" -gt 0 ] || { echo "no logs found (run from the repo root, or pass log files)"; exit 1; }

printf '%-34s %7s %8s %9s %6s %9s %9s %6s  %-11s %s\n' run lambda S0 "lam/S0e4" step pred epi_ratio erank "succ 50/200" log
for f in "${LOGS[@]}"; do
    run=$(grep -m1 -o 'run_name=[^ ]*' "$f" | cut -d= -f2)
    [ -n "$run" ] || continue
    lam=$(grep -m1 -o 'loss\.epi\.weight=[^ ]*' "$f" | cut -d= -f2)
    s0=$(grep -m1 -o "\] S0=[0-9.]*" "$f" | cut -d= -f2)
    row=$(grep -F "[$run] step=" "$f" | tail -n 1)
    get() { echo "$row" | grep -o " $1=[^ ]*" | head -n 1 | cut -d= -f2; }
    ratio="-"
    if [ -n "$lam" ] && [ -n "$s0" ]; then ratio=$(awk -v l="$lam" -v s="$s0" 'BEGIN {printf "%.2f", 1e4 * l / s}'); fi
    r50=$(grep -o 'RESULT .* n=50 success=[0-9.]*' "$f" | tail -n 1 | grep -o '[0-9.]*$')
    r200=$(grep -o 'RESULT .* n=200 success=[0-9.]*' "$f" | tail -n 1 | grep -o '[0-9.]*$')
    printf '%-34s %7s %8s %9s %6s %9s %9s %6s  %-11s %s\n' "$run" "${lam:--}" "${s0:--}" "$ratio" \
        "$(get step)" "$(get pred)" "$(get epi_ratio)" "$(get erank)" "${r50:--}/${r200:--}" "$(basename "$f")"
done
