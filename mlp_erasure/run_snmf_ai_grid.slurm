#! /bin/sh
#SBATCH --job-name=snmf-ai-grid
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602,n-301
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Re-run of AI stage 1 only. Job 895359 died mid-grid with
# "CUDA error: unspecified launch failure" on n-301 after cell 0, and every
# later process on that node fell back to CPU. Stages 2 and 3 there are still
# valid (fresh model loads, and stage 3 reproduced the known control baseline
# 44.0/54.0 exactly), but cells 1 and 2 of the train grid were never scored.
# n-301 excluded here as a precaution -- it is still MIXED, not drained, so
# the fault looks transient rather than a dead card.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

"$PY" "$ROOT/sweep_delta.py" \
  --model /home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k \
  --snmf-out "$RUNS/snmf_ai_887806" \
  --concept "Artificial intelligence" \
  --questions "$ROOT/Ember-on-LMEnt/data/mc_questions.json" \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/sweep_ai_train_retry_${SLURM_JOB_ID}.csv"
echo "exit $?"
echo DONE
