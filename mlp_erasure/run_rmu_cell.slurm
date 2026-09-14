#! /bin/sh
#SBATCH --job-name=rmu-sweep
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Re-run of one RMU depth-grid cell against the FIXED code on
# fix/erasure-delta-and-rmu-steps (retain-cycling, rotation_is_substantial /
# ran_enough_steps sanity gates). Bypasses run_rmu.slurm entirely: that
# script hardcodes ROOT=/home/morg/.../LMEnt-mlp (the old, unfixed branch)
# and does not expose --layer-id/--layer-ids, both needed here.
#
# Trains, saves the checkpoint, scores it on the real held-out QA/SimdomQA
# questions immediately, then deletes the checkpoint -- same cleanup
# discipline as the original investigation.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

: "${OUT_TAG:?set OUT_TAG}"
: "${LAYER_ID:?set LAYER_ID}"
: "${LAYER_IDS:?set LAYER_IDS}"
: "${STEERING:?set STEERING}"
: "${MODEL_PATH:=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k}"
: "${CONCEPT:=Ancient Rome}"
: "${MAX_NUM_BATCHES:=150}"
: "${BATCH_SIZE:=4}"
: "${LR:=1e-4}"
: "${ALPHA:=100}"
: "${DTYPE:=fp32}"

DATA="$ROOT/Ember-on-LMEnt/data"
OUT="$RUNS/rmu_${OUT_TAG}_${SLURM_JOB_ID}"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

echo "=== RMU cell $OUT_TAG: layer_id=$LAYER_ID layer_ids=$LAYER_IDS steering=$STEERING concept='$CONCEPT' ==="
echo "root=$ROOT out=$OUT"

"$PY" "$ROOT/rmu.py" \
  --model "$MODEL_PATH" --concept "$CONCEPT" \
  --concept-sentences "$DATA/concept_sentences.json" \
  --neutral-sentences "$DATA/neutral_sentences.json" \
  --dtype "$DTYPE" --lr "$LR" --alpha "$ALPHA" --steering "$STEERING" \
  --layer-id "$LAYER_ID" --layer-ids $LAYER_IDS \
  --batch-size "$BATCH_SIZE" --max-num-batches "$MAX_NUM_BATCHES" \
  --out "$OUT" --sanity
RMU_EXIT=$?
echo "rmu.py exited $RMU_EXIT"

if [ -d "$OUT/model" ]; then
  echo "=== scoring $OUT/model on real QA_test/SimdomQA_test ==="
  "$PY" "$ROOT/ember_eval/score_ember_mc.py" \
    --model "$OUT/model" --subfolder "" \
    --ember-data "$DATA" \
    --concepts "$CONCEPT" --sets qa_test simdom_test \
    --out "$OUT/eval.json"
  EVAL_EXIT=$?
  echo "score_ember_mc.py exited $EVAL_EXIT"
  echo "=== deleting checkpoint, keeping trace/sanity/eval/config json ==="
  rm -rf "$OUT/model"
else
  echo "no model dir at $OUT/model -- rmu.py did not save a checkpoint (exit $RMU_EXIT)"
fi

echo "DONE cell=$OUT_TAG rmu_exit=$RMU_EXIT out=$OUT"
