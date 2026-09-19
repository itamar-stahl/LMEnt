#! /bin/sh
#SBATCH --job-name=snmf-fullrange
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --exclude=n-102
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Layer-coverage confirmation for the delta=1 SNMF null.
#
# The train sweep and test confirmation both went through sweep_delta.py,
# which only exposes --range-cells (the published grid). Cell 0's OUTPUT
# range on 18 layers is [0,5], so any selected feature at layer 6 had its
# output side silently skipped -- the RANGE_CELL gotcha MLP_ERASURE.md
# documents (cmd_erase `continue`s past out-of-range layers, no error, just
# fewer ablations than selected).
#
# This runs snmf.py erase directly with EXPLICIT --layers-in/--layers-out
# covering every layer that actually holds a selected feature, on BOTH
# sides, at delta 1/1, and scores each result on the held-out test split
# with the same harness every other number in the table came from
# (score_ember_mc.py, text_char). Verify the printed
# "N layers edited input-side, M output-side" line against the layer counts.
#
# Selected-feature layers, measured from each cache's selected.json:
#   rome_judge   (snmf_romeband_871493): layer 5 (3), layer 6 (3)   -> 5..6
#   rome_ratio44 (snmf_rome_883188):     layers 4 (10), 5 (31), 6 (3) -> 4..6
#   baseball58   (snmf_baseball_887805): layers 4 (19), 5 (28), 6 (11) -> 4..6
#
# Control baselines on this harness: Rome QA_test 44.0% / SimdomQA_test 52.0%.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

MODEL_PATH=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
DATA="$ROOT/Ember-on-LMEnt/data"
SCRATCH="$RUNS/fullrange_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

run_cell() {
  tag=$1; snmf_out=$2; concept=$3; lo=$4; hi=$5
  saved="$SCRATCH/model_$tag"
  echo
  echo "================ $tag : layers $lo..$hi both sides, delta 1/1 ================"
  "$PY" "$ROOT/snmf.py" erase \
    --model "$MODEL_PATH" --out "$snmf_out" \
    --delta-in 1 --delta-out 1 \
    --layers-in "$lo" "$hi" --layers-out "$lo" "$hi" \
    --dtype fp32 --save "$saved"
  echo "erase exit $?"

  if [ -d "$saved" ]; then
    "$PY" "$ROOT/ember_eval/score_ember_mc.py" \
      --model "$saved" --subfolder "" \
      --ember-data "$DATA" \
      --concepts "$concept" --sets qa_test simdom_test \
      --out "$SCRATCH/eval_$tag.json"
    echo "score exit $?"
    rm -rf "$saved"
    echo "deleted checkpoint for $tag"
  else
    echo "NO MODEL SAVED for $tag -- erase failed"
  fi
}

run_cell rome_judge   "$RUNS/snmf_romeband_871493" "Ancient Rome" 5 6
run_cell rome_ratio44 "$RUNS/snmf_rome_883188"     "Ancient Rome" 4 6
run_cell baseball58   "$RUNS/snmf_baseball_887805" "Baseball"     4 6

echo
echo "eval JSONs kept in $SCRATCH"
echo DONE
