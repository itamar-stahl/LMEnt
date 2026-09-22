#!/bin/bash
#SBATCH --job-name=score
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=n-102,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205
#
# n-202..205 are excluded for their filer link, not their 11 GB: six scoring
# jobs there sat 20+ minutes in D state (folio_wait_bit) with the model already
# on the GPU and nothing written, while the same job finished in 12 min on an
# L40S node. The SNMF loads there took 25 minutes for 5 GB.
#
# Score one model on the three topics' sets (answer NLL + completion-position
# distributions). One model load, ~seconds of scoring per topic.
#
#   sbatch --export=ALL,ROOT=<worktree>,MODEL=<hf dir>,OUT=<scored>/<name>[,EXTRA=--no-dists] \
#          --job-name=<name> --output=... score.slurm
set -euo pipefail

: "${ROOT:?set ROOT to the LMEnt worktree}"
: "${MODEL:?set MODEL}"
: "${OUT:?set OUT}"
: "${SETS:=$ROOT/ember_eval/nll_kl/sets/rome.json $ROOT/ember_eval/nll_kl/sets/baseball.json $ROOT/ember_eval/nll_kl/sets/ai.json}"
: "${EXTRA:=}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
if [ ! -r "$MODEL/config.json" ]; then
  echo "FATAL: $MODEL is not readable from $(hostname)" >&2; exit 1
fi
echo "score model=$MODEL out=$OUT extra=[$EXTRA]"

# shellcheck disable=SC2086
exec "$PY" -u "$ROOT/ember_eval/nll_kl/score_model.py" \
  --model "$MODEL" --sets $SETS --out "$OUT" $EXTRA
