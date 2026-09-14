#! /bin/sh
#SBATCH --job-name=snmf-baseball
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --exclude=n-102
#SBATCH --time=45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Baseball SNMF re-run on the FIXED code, on the control model (same as the
# Rome re-run) -- only feature cache available is ratio-only, 58 features,
# seed 42 (snmf_baseball_887805; this is the cell whose original -8pt at
# delta=4 did NOT replicate under a 3-seed check, see
# lment-crosstopic-mlp-erasure.md). Full delta x range-cell grid on train
# first, then confirm the best non-amplifying cell on held-out test.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

MODEL_PATH=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
CONCEPT="Baseball"
DATA="$ROOT/Ember-on-LMEnt/data"
SNMF_OUT="$RUNS/snmf_baseball_887805"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

echo "=== Baseball, control model, full grid on train ==="
"$PY" "$ROOT/sweep_delta.py" \
  --model "$MODEL_PATH" --snmf-out "$SNMF_OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/sweep_baseball_train_${SLURM_JOB_ID}.csv"
echo "train exit $?"

echo "=== Baseball, control model, cell 0 delta 1.0, held-out test ==="
"$PY" "$ROOT/sweep_delta.py" \
  --model "$MODEL_PATH" --snmf-out "$SNMF_OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --subset QA_test --simdom-subset SimdomQA_test \
  --deltas 1.0 --range-cells 0 \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/confirm_baseball_test_${SLURM_JOB_ID}.csv"
echo "test exit $?"

echo DONE
