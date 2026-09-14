#! /bin/sh
#SBATCH --job-name=snmf-confirm
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --exclude=n-102
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Confirmatory step: the train-split sweep (job 890822) picked cell 0 /
# delta 1.0 as the best non-amplifying cell for BOTH feature sets (nothing
# else came close -- every cell was flat at delta=1). Score exactly those
# two pre-chosen cells against the held-out QA_test/SimdomQA_test split --
# not the whole grid, so this is a confirmation, not a test-set selection.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

MODEL_PATH=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
CONCEPT="Ancient Rome"
DATA="$ROOT/Ember-on-LMEnt/data"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

for pair in "judge:$RUNS/snmf_romeband_871493" "ratio44:$RUNS/snmf_rome_883188"
do
  tag=${pair%%:*}
  snmf_out=${pair#*:}
  out_csv="$RUNS/confirm_rome_${tag}_${SLURM_JOB_ID}.csv"
  echo "=== confirm [$tag] cell 0 delta 1.0 on QA_test/SimdomQA_test ==="
  "$PY" "$ROOT/sweep_delta.py" \
    --model "$MODEL_PATH" --snmf-out "$snmf_out" \
    --concept "$CONCEPT" \
    --questions "$DATA/mc_questions.json" \
    --subset QA_test --simdom-subset SimdomQA_test \
    --deltas 1.0 --range-cells 0 \
    --dtype fp32 --gamma 0.95 \
    --out "$out_csv"
  echo "=== [$tag] exit $? ==="
done

echo DONE
