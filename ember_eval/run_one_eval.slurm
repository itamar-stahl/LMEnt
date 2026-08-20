#! /bin/sh
#SBATCH --job-name=ember-one
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# No --constraint: a 1B in fp32 is ~4.4 GB and fits any card in killable, so
# this takes whatever is free rather than queueing for a type. Deliberately not
# an H100/H200 partition -- scoring must never compete with training for the
# scarce cards.
#
# Scores ONE model, named by the environment, so a twin can be evaluated the
# moment it finishes instead of waiting for its sibling:
#
#   sbatch --export=ALL,MODEL_PATH=/path/to/hf,OUT_TAG=twins2e_noporn run_one_eval.slurm
#
# All four question sets (3,600 questions), fp32, so the records are directly
# comparable to twins4_*.json from the 1-epoch pair.

EVAL=/home/dcor/galbarak2/LMEnt-ember/ember_eval
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

: "${MODEL_PATH:?set MODEL_PATH}"
: "${OUT_TAG:?set OUT_TAG}"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"
echo "scoring ${MODEL_PATH} -> ${OUT_TAG}"

# --subfolder "" because these are local directories, not HF repos with a
# stepNNNN subfolder.
"$PY" "$EVAL/score_ember_mc.py" \
  --model "$MODEL_PATH" \
  --subfolder "" \
  --sets qa_train qa_test simdom_train simdom_test \
  --dtype float32 \
  --out "$EVAL/results/${OUT_TAG}_${SLURM_JOB_ID}.json"
