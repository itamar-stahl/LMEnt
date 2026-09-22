#!/bin/bash
#SBATCH --job-name=finalize
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=n-102,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#
# Run finalize.py: score whatever is missing, select, score the winners on the
# test sets, build the tables. Idempotent -- submit several copies back to
# back with --dependency=afterany and a preemption costs only the work in
# flight, since each one re-checks the disk and does only what is left.
#
#   sbatch --export=ALL,ROOT=<worktree>,RUN=<run root> finalize.slurm
set -uo pipefail

: "${ROOT:?set ROOT to the LMEnt worktree}"
: "${RUN:?set RUN to the run root}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
echo "finalize on $(hostname) at $(date +%F_%H:%M:%S)"
exec "$PY" -u "$ROOT/ember_eval/nll_kl/finalize.py" --run "$RUN" --root "$ROOT"
