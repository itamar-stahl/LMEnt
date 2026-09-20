#!/bin/bash
#SBATCH --job-name=scoremany
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=n-102,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205
#
# Score several models one after another in ONE job. Use this for a grid of
# candidates: every candidate is a different 5 GB checkpoint, so N parallel
# jobs on one node each read the filer at 1/N speed and time out (8 jobs on
# n-306 did, at 30 min, having loaded nothing). Sequential reads run at the
# full per-node link, ~2-7 min per model.
#
#   sbatch --export=ALL,ROOT=<worktree>,PAIRS="name1=<dir1> name2=<dir2> ...",\
#          OUT_ROOT=<scored>,SETS=<sets.json>[,EXTRA="--no-dists --only-phase selection"] score_many.slurm
#
# A model whose <OUT_ROOT>/<name>/<sets stem>/records.json already exists is
# skipped, so a timed-out job can simply be resubmitted.
set -uo pipefail

: "${ROOT:?set ROOT to the LMEnt worktree}"
: "${PAIRS:?set PAIRS}"
: "${OUT_ROOT:?set OUT_ROOT}"
: "${SETS:?set SETS}"
: "${EXTRA:=}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
stem=$(basename "$SETS" .json)
failed=0
for pair in $PAIRS; do
  name=${pair%%=*}; dir=${pair#*=}
  if [ -f "$OUT_ROOT/$name/$stem/records.json" ]; then echo "skip $name (scored)"; continue; fi
  if [ ! -r "$dir/config.json" ]; then echo "FATAL: $dir not readable from $(hostname)" >&2; failed=1; continue; fi
  echo "=== $name  $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  "$PY" -u "$ROOT/ember_eval/nll_kl/score_model.py" --model "$dir" --sets "$SETS" --out "$OUT_ROOT/$name" $EXTRA \
    || { echo "FAILED: $name" >&2; failed=1; }
done
exit $failed
