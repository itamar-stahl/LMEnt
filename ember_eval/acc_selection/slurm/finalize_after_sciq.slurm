#!/bin/bash
#SBATCH --job-name=finalagg
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#
# Runs once the sciq NLL/KL scoring jobs end, whatever their exit status
# (afterany), and aggregates. Loads no model, so it needs no GPU.
#
# A slurm dependency rather than a watcher process: watchers on the login node
# have been reaped repeatedly for memory today, and this must not depend on one
# surviving.
set -uo pipefail
: "${ROOT:?}"; : "${R:?}"; : "${NK:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
echo "host $(hostname)  $(date)"
echo "sciq dists present: $(find $NK/scored -path '*/sciq/dists.npy' 2>/dev/null | wc -l)/13"
"$PY" -u "$ROOT/ember_eval/acc_selection/finalize_tables.py" \
  --rollup "$R/test_accuracy_rollup.csv" \
  --rankings "$ROOT/../lment-mergeprep/ember_eval/acc_selection/results_sciq/rankings.csv" \
  --nk-scored "$NK/scored" --out "$R/final"
echo "=== final/ ==="; ls -la "$R/final"
