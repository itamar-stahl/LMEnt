#! /bin/sh
#SBATCH --job-name=ember-eval-lment
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# No --constraint: the model is 1B in fp32 (~4.4 GB), which fits any card in
# killable, so this takes whatever is free rather than competing for a type.

EVAL=/home/dcor/galbarak2/LMEnt-ember/ember_eval
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export HF_HUB_OFFLINE=1          # the checkpoint is already in the cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"

"$PY" "$EVAL/score_ember_mc.py" \
  --concepts "$UNTAUGHT_EVAL_CONCEPT" \
  --sets qa_test simdom_test \
  --dtype float32 \
  --out "$EVAL/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.json"
