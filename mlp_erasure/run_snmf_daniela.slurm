#! /bin/sh
#SBATCH --job-name=snmf-daniela
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
# SNMF erasure, Ancient Rome, on Daniela Gottesman's released LMEnt-1B-2E
# checkpoint (dhgottesman/LMEnt-1B-2E) instead of Gal's own control -- a
# cross-checkpoint check of whether the Rome null result (real delta=1,
# already established on Gal's control) replicates on an independently
# trained/released model. No cached features exist for this model, so this
# does a fresh factorize (layers 4/5/6, matching the Rome cache's layers)
# and a ratio-only select (--skip-llm, no judge -- consistent with how the
# Rome/Baseball re-runs on Gal's control avoided the flaky Gemma subprocess).

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

MODEL_PATH=/home/dcor/galbarak2/hf_cache/hub/models--dhgottesman--LMEnt-1B-2E/snapshots/d0ce274557365c983032af9d7f8ae59851df18d8/step219344
CONCEPT="Ancient Rome"
DATA="$ROOT/Ember-on-LMEnt/data"
OUT="$RUNS/snmf_daniela2e_rome_${SLURM_JOB_ID}"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

echo "=== factorize (layers 4 5 6) ==="
"$PY" "$ROOT/snmf.py" factorize \
  --model "$MODEL_PATH" --concept "$CONCEPT" \
  --concept-sentences "$DATA/concept_sentences.json" \
  --neutral-sentences "$DATA/neutral_sentences.json" \
  --out "$OUT" --layers 4 5 6 --dtype fp32
echo "factorize exit $?"

echo "=== select --skip-llm (ratio-only, no judge) ==="
"$PY" "$ROOT/snmf.py" select --out "$OUT" --concept "$CONCEPT" --skip-llm
echo "select exit $?"

echo "=== sweep delta 1.0 and the published grid, train split ==="
"$PY" "$ROOT/sweep_delta.py" \
  --model "$MODEL_PATH" --snmf-out "$OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/sweep_daniela2e_rome_train_${SLURM_JOB_ID}.csv"
echo "train sweep exit $?"

echo "=== confirm cell 0 delta 1.0 on held-out test ==="
"$PY" "$ROOT/sweep_delta.py" \
  --model "$MODEL_PATH" --snmf-out "$OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --subset QA_test --simdom-subset SimdomQA_test \
  --deltas 1.0 --range-cells 0 \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/confirm_daniela2e_rome_test_${SLURM_JOB_ID}.csv"
echo "test confirm exit $?"

echo DONE
