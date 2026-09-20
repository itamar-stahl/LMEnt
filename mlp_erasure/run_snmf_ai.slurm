#! /bin/sh
#SBATCH --job-name=snmf-ai
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Artificial intelligence -- the last hole in the cross-topic table. Rome and
# Baseball were both re-run on the fixed code; AI never was, so the only AI
# SNMF numbers on record are from the buggy delta=4 era (-6pt, single seed,
# never replicated -- lment-crosstopic-mlp-erasure.md flags it as probably
# noise).
#
# Cache: snmf_ai_887806, ratio-only, 84 features (L4:28 L5:45 L6:11) -- the
# LARGEST feature set of the four (Rome 44, Daniela 51, Baseball 58, AI 84).
# That makes AI the sharpest test of the hypothesis from job 893308/893467:
# large-delta damage tracks FEATURE COUNT, not concept. If so, AI should
# degrade MORE than any other topic at x20 and still show no real-vs-permuted
# separation.
#
# Three stages, mirroring exactly what Rome and Baseball got:
#   1. full delta grid on train (1/4/7/10 x 3 range cells)
#   2. delta=1 on held-out test with EXPLICIT full layer ranges, scored with
#      score_ember_mc.py -- avoids the RANGE_CELL silent-skip and gives a
#      number directly comparable to the RMU cells
#   3. permutation control at delta 1 and -19 on test

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

MODEL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
CONCEPT="Artificial intelligence"
SNMF_OUT="$RUNS/snmf_ai_887806"
DATA="$ROOT/Ember-on-LMEnt/data"
SCRATCH="$RUNS/ai_fullrange_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

echo "########## STAGE 1: delta grid, train split ##########"
"$PY" "$ROOT/sweep_delta.py" \
  --model "$MODEL" --snmf-out "$SNMF_OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/sweep_ai_train_${SLURM_JOB_ID}.csv"
echo "stage1 exit $?"

echo
echo "########## STAGE 2: delta=1, explicit layers 4-6 both sides, TEST ##########"
SAVED="$SCRATCH/model_ai_delta1"
"$PY" "$ROOT/snmf.py" erase \
  --model "$MODEL" --out "$SNMF_OUT" \
  --delta-in 1 --delta-out 1 \
  --layers-in 4 6 --layers-out 4 6 \
  --dtype fp32 --save "$SAVED"
echo "erase exit $?"
if [ -d "$SAVED" ]; then
  "$PY" "$ROOT/ember_eval/score_ember_mc.py" \
    --model "$SAVED" --subfolder "" \
    --ember-data "$DATA" \
    --concepts "$CONCEPT" --sets qa_test simdom_test \
    --out "$SCRATCH/eval_ai_delta1.json"
  echo "score exit $?"
  rm -rf "$SAVED"
  echo "checkpoint deleted"
else
  echo "NO MODEL SAVED -- erase failed"
fi

echo
echo "########## STAGE 3: permutation control, TEST ##########"
"$PY" "$ROOT/random_direction_control.py" \
  --model "$MODEL" --snmf-out "$SNMF_OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --subset QA_test --simdom-subset SimdomQA_test \
  --deltas 1.0 -19.0 \
  --seeds 1 2 3 \
  --gamma 0.95 --dtype fp32 \
  --out "$RUNS/randctl_ai_${SLURM_JOB_ID}.csv"
echo "stage3 exit $?"

echo DONE
