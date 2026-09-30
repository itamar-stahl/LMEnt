#! /bin/sh
#SBATCH --job-name=completion-eval
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/completion/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/completion/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-301,n-303
#
# Sentence-completion evaluation (Tamar's completion_eval/) on ONE model, all
# four Pornography splits. Same partition/account/exclude as run_one_eval.slurm
# -- see that file for why: those DGX boxes miss mounts, and scoring must never
# compete with training for H100s. A 1B in fp32 is ~4.4 GB and fits any card.
#
#   sbatch --export=ALL,MODEL_PATH=/path/or/repo-id,OUT_TAG=cmpl_control2e \
#          run_completion_eval.slurm
#
# SPLITS overrides which of the four splits run, space separated. The default is
# all four. Re-scoring one split alone is what you want after a scoring change
# lands mid-run: each split is a separate python process, so an edit to
# evaluate_completion.py reaches every split that has not started yet and leaves
# the finished ones on the old rule. Check `metadata.metric` in each output
# before trusting a set that was written across such a change.
#
#   sbatch --export=ALL,MODEL_PATH=...,OUT_TAG=bb_control2e,CONCEPT=Baseball,\
#          SPLITS=QA_train run_completion_eval.slurm
#
# MODEL_PATH may be a local directory OR a HuggingFace repo id (the released
# dhgottesman/LMEnt-1B-*E are already in HF_HOME below, so no download).
#
# SUBFOLDER is required for the released repos and must be empty for ours.
# dhgottesman/LMEnt-1B-*E keep NOTHING at the repo root -- every checkpoint sits
# in a stepNNNNN/ subfolder. Asked for the root, AutoTokenizer finds no
# tokenizer_config.json, falls back to guessing, and instantiates an ESM
# (protein language model) tokenizer whose vocab_file is None:
#
#   TypeError: expected str, bytes or os.PathLike object, not NoneType
#
# which names neither the repo nor the real problem. Job 776328 died this way.
# The 2-epoch endpoint is step219344 (219,344 x 32,768 tokens = 7.2B = 2 epochs).
#
# 200 questions x 4 options, sequences of a few dozen tokens: the GPU work is
# seconds and model load dominates.

CE=/home/morg/NLP_2526b/galbarak2/LMEnt/ember_eval/completion_eval
OUT=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/completion
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

: "${MODEL_PATH:?set MODEL_PATH}"
: "${OUT_TAG:?set OUT_TAG}"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"
echo "node=$(hostname)  model=${MODEL_PATH}  subfolder=${SUBFOLDER:-<none>}  concept=${CONCEPT:-Pornography}  tag=${OUT_TAG}"

# A local path that is not visible from this node must fail loudly here, or
# transformers reinterprets it as a repo id and dies with HFValidationError
# naming your absolute path -- which reads like a code bug and is not one.
# A bare repo id (no leading /) is left for transformers to resolve.
case "$MODEL_PATH" in
  /*)
    if [ ! -d "$MODEL_PATH" ] || [ ! -r "$MODEL_PATH/config.json" ]; then
      echo "FATAL: $MODEL_PATH is not readable from $(hostname)." >&2
      echo "       The filesystem holding it is probably not mounted here." >&2
      exit 1
    fi
    ;;
esac

rc=0
for SPLIT in ${SPLITS:-QA_train QA_test SimdomQA_train SimdomQA_test}; do
  echo "=== ${OUT_TAG} / ${CONCEPT:-Pornography} / ${SPLIT}"
  "$PY" "$CE/evaluate_completion.py" \
    --model "$MODEL_PATH" \
    --subfolder "${SUBFOLDER:-}" \
    --questions "$CE/data/completion_questions.json" \
    --concept "${CONCEPT:-Pornography}" \
    --split "$SPLIT" \
    --dtype float32 \
    --out "$OUT/${OUT_TAG}_${SPLIT}_${SLURM_JOB_ID}.json" || rc=1
done

echo "=== done rc=$rc"
exit $rc
