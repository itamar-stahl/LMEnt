#!/bin/bash
#SBATCH --job-name=repair
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=n-102,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#
# Rebuild the three 2026-09-20 candidates that were saved with holes. See
# ../check_alloc.py and runs/nll_kl/_corrupt_20260921/README.md.
#
# Everything here runs SEQUENTIALLY in one job on purpose. The original grid
# had several 5 GB saves landing on one node at once, which is what produced
# the short writes (and, separately, the 4 h scoring timeouts). One writer at
# a time, each verified before the next starts.
#
# The RMU arguments are copied from the config.json each corrupt run left
# behind, so these reproduce the same cells: same seed, same steering, same
# layer band. Both passed their rotation sanity gate the first time round.
#
#   sbatch --export=ALL,ROOT=<worktree>,CAND=<candidates dir> repair_holed.slurm
set -uo pipefail

: "${ROOT:?set ROOT to the LMEnt worktree}"
: "${CAND:?set CAND to the candidates dir}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
CTRL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
DATA="$ROOT/Ember-on-LMEnt/data"
CHECK="$ROOT/ember_eval/nll_kl/check_alloc.py"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
echo "repair on $(hostname) at $(date +%F_%H:%M:%S)"
failed=0

rmu_cell () {   # $1 tag, $2 concept, $3 steering
  local tag=$1 concept=$2 steering=$3 out="$CAND/rmu_${1}_rebuild"
  for attempt in 1 2; do
    echo "=== rmu $tag attempt $attempt  $(date +%H:%M:%S)"
    rm -rf "$out"
    "$PY" -u "$ROOT/mlp_erasure/rmu.py" \
      --model "$CTRL" --concept "$concept" \
      --concept-sentences "$DATA/concept_sentences.json" \
      --neutral-sentences "$DATA/neutral_sentences.json" \
      --out "$out" --dtype fp32 \
      --layer-id 6 --layer-ids 4 5 6 \
      --lr 1e-4 --alpha 100 --steering "$steering" \
      --batch-size 1 --min-len 50 --max-len 2000 \
      --max-num-batches 150 --max-length 512 --seed 42 \
      --sanity --min-cosine 0.3 --min-step-fraction 0.5 --save-model \
      || { echo "rmu $tag: build failed" >&2; continue; }
    sync
    if "$PY" "$CHECK" "$out/model"; then echo "rmu $tag: verified"; return 0; fi
    echo "rmu $tag: HOLED on attempt $attempt" >&2
  done
  failed=1; return 1
}

snmf_cell () {  # $1 name, $2 concept, $3 fit dir, $4 delta_in, $5 delta_out
  local name=$1 concept=$2 fit=$3 din=$4 dout=$5 out="$CAND/$1/model"
  for attempt in 1 2; do
    echo "=== snmf $name attempt $attempt  $(date +%H:%M:%S)"
    rm -rf "$out"
    "$PY" -u "$ROOT/mlp_erasure/snmf.py" erase \
      --model "$CTRL" --out "$fit" --dtype fp32 \
      --delta-in "$din" --delta-out "$dout" \
      --range-cell 0 --layers-in 4 6 --layers-out 4 6 \
      --save "$out" \
      || { echo "snmf $name: build failed" >&2; continue; }
    sync
    if "$PY" "$CHECK" "$out"; then echo "snmf $name: verified"; return 0; fi
    echo "snmf $name: HOLED on attempt $attempt" >&2
  done
  failed=1; return 1
}

rmu_cell  ai_L6hi_a100   "Artificial intelligence" 948.9
rmu_cell  rome_L6hi_a100 "Ancient Rome"            945.9683
snmf_cell snmf_baseball_ratio_in "Baseball" \
          /home/dcor/galbarak2/runs/mlp_erasure/snmf_baseball_887805 1 0

echo "repair done at $(date +%F_%H:%M:%S), failed=$failed"
exit $failed
