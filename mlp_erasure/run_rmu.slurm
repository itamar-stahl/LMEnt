#! /bin/sh
#SBATCH --job-name=rmu-erase
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205
# --exclude, first three: those DGX boxes in killable do not mount the filers
# the models live on, and transformers reinterprets an unreadable absolute path
# as a hub repo id, so the failure reads like a code bug. run_one_eval.slurm
# carries the full story.
#
# --exclude, n-202..205: the 2080s have 11 GB. RMU is the one stage that holds
# TWO copies of the model -- 4.4 GB each in fp32 -- plus the autograd graph up
# to the hooked layer, and 2 GB of headroom for that is too thin to rely on.
# Every other card in killable has 24 GB or more. run_snmf.slurm does not
# exclude them: it holds one copy and a 0.38 GB activation matrix.
#
# No --constraint. EMBER's erasure of this concept pinned a6000
# (configs/ember_lment_rome_slurm.yaml) and its stated reason was staying out
# of the h100 pool the twin training needs -- which every other card in
# killable does equally. Dropping it is Gal's call, made 2026-09-09 after four
# jobs pinned to a6000 all landed on n-602 and none finished loading the
# model: job 870565 measured that node reading /home/dcor at 5.4 MB/s direct
# and 1.5 MB/s buffered, against 36.6 MB/s from the login node. transformers
# loads buffered, so one 4.5 GB shard there is ~50 minutes. Nothing was wrong
# with the card, the job, or the code.
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
#
# MODE=run does NOT keep the model by default -- a 1B in fp32 is 4.4 GB per
# cell and the point of most cells is the trace, not the weights. Export
# SAVE_MODEL= (empty) for the one cell you want to keep, and give it a name
# that matches the erased models already on disk, e.g.
# hf-models/lment-1b-rome-erased-b131k is EMBER's.

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
