#! /bin/sh
#SBATCH --job-name=rmu-erase
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --constraint=a6000
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
# --exclude: those DGX boxes in killable do not mount the filers the models
# live on, and transformers reinterprets an unreadable absolute path as a hub
# repo id, so the failure reads like a code bug. run_one_eval.slurm carries the
# full story.
#
# --constraint=a6000: the same target the EMBER erasure of this concept ran on
# (configs/ember_lment_rome_slurm.yaml), chosen there to stay out of the h100
# pool that the twin training needs. RMU holds TWO copies of the 1B in fp32
# (~4.4 GB each) plus activations, which no card in killable struggles with;
# the constraint is about not competing for the scarce cards, not about VRAM.
#
#   MODE=probe  report activation norms per candidate layer, then stop. Run
#               this first: --steering has to be on the scale of the residual
#               norms this model actually produces, and the published grid
#               {30,100,300,1000} was tuned on wider models.
#   MODE=run    the real unlearning pass, with --sanity gating the exit code.
#
#   sbatch --export=ALL,MODE=probe,OUT_TAG=rome_probe run_rmu.slurm
#   sbatch --export=ALL,MODE=run,STEERING=<from probe>,OUT_TAG=rome_lr1e-4 \
#          run_rmu.slurm

ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-mlp
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

: "${MODE:=probe}"
: "${OUT_TAG:?set OUT_TAG}"
: "${MODEL_PATH:=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k}"
: "${CONCEPT:=Ancient Rome}"
: "${MAX_NUM_BATCHES:=150}"
: "${BATCH_SIZE:=16}"
: "${MAX_LENGTH:=512}"
: "${LR:=1e-4}"
: "${ALPHA:=100}"
: "${DTYPE:=fp32}"
: "${SAVE_MODEL:=--no-save-model}"

DATA="$ROOT/Ember-on-LMEnt/data"
OUT="$RUNS/rmu_${OUT_TAG}_${SLURM_JOB_ID}"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

if [ ! -d "$MODEL_PATH" ] || [ ! -r "$MODEL_PATH/config.json" ]; then
  echo "FATAL: $MODEL_PATH is not readable from $(hostname)." >&2
  exit 1
fi

echo "rmu MODE=$MODE concept='$CONCEPT' model=$MODEL_PATH -> $OUT"

if [ "$MODE" = "probe" ]; then
  exec "$PY" "$ROOT/rmu.py" \
    --model "$MODEL_PATH" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --dtype "$DTYPE" --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH" \
    --out "$OUT" --probe
fi

: "${STEERING:?set STEERING from the probe run}"

exec "$PY" "$ROOT/rmu.py" \
  --model "$MODEL_PATH" --concept "$CONCEPT" \
  --concept-sentences "$DATA/concept_sentences.json" \
  --neutral-sentences "$DATA/neutral_sentences.json" \
  --dtype "$DTYPE" --lr "$LR" --alpha "$ALPHA" --steering "$STEERING" \
  --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH" \
  --max-num-batches "$MAX_NUM_BATCHES" \
  --out "$OUT" --sanity $SAVE_MODEL
