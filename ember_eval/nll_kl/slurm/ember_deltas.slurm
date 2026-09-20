#!/bin/bash
#SBATCH --job-name=embdelta
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=n-102,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Materialise one topic's EMBER delta grid from its existing feature artifact.
#
#   sbatch --export=ALL,ROOT=<worktree>,REPORT=<run>/outputs/report.json,\
#          OUT_ROOT=<candidates>/ember_rome,DELTAS="1 2 5 10 20 50 100 200 500" \
#          --output=... ember_deltas.slurm
#
# The GPU is only for applying the edit in the same place the pipeline did
# (cuda, fp32), so the chosen delta reproduces the shipped embedding
# byte-for-byte; the materialisation itself is CPU + filer I/O.
set -euo pipefail

: "${ROOT:?set ROOT to the LMEnt worktree}"
: "${REPORT:?set REPORT}"
: "${OUT_ROOT:?set OUT_ROOT}"
: "${DELTAS:=1 2 5 10 20 50 100 200 500}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
echo "ember_deltas report=$REPORT out=$OUT_ROOT deltas=[$DELTAS]"

# shellcheck disable=SC2086
exec "$PY" "$ROOT/ember_eval/nll_kl/ember_deltas.py" \
  --report "$REPORT" --out-root "$OUT_ROOT" --deltas $DELTAS
