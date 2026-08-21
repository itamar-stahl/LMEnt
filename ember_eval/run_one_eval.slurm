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
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
# --exclude because those DGX boxes in killable do not mount /vol/scratch, where
# the models live. transformers only treats an argument as a local path when
# os.path.isdir() succeeds; on a node without the mount it falls through to
# "must be a repo id" and dies with HFValidationError naming your absolute path,
# which reads like a code bug and is not one. Job 770273 lost 26 seconds to this.
#
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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# Fail fast and legibly if the model is not visible from this node, rather than
# letting transformers reinterpret the path as a HuggingFace repo id.
if [ ! -d "$MODEL_PATH" ] || [ ! -r "$MODEL_PATH/config.json" ]; then
  echo "FATAL: $MODEL_PATH is not readable from $(hostname)." >&2
  echo "       The filesystem holding it is probably not mounted here." >&2
  exit 1
fi

echo "scoring ${MODEL_PATH} -> ${OUT_TAG} on $(hostname)"

# --subfolder "" because these are local directories, not HF repos with a
# stepNNNN subfolder.
"$PY" "$EVAL/score_ember_mc.py" \
  --model "$MODEL_PATH" \
  --subfolder "" \
  --sets qa_train qa_test simdom_train simdom_test \
  --dtype float32 \
  --out "$EVAL/results/${OUT_TAG}_${SLURM_JOB_ID}.json"
