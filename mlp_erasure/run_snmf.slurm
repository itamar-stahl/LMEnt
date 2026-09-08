#! /bin/sh
#SBATCH --job-name=snmf-erase
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --constraint=a6000
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
# See run_rmu.slurm for --exclude and --constraint. Two extra reasons here:
#
# --mem=64000, not 32000. factorize holds one layer's activations on the host
# as fp32 (d_mlp x n_tokens; 5,632 x ~90k on 600 sentences at max-length 256 is
# ~2 GB) and semi_nmf keeps three tensors of that size on the GPU. It is the
# host side that grows with --layer-batch-size.
#
# --time=360, and MODE=select can need most of it: the judge is one unsharded
# 23.9 GB safetensors read off NFS at ~28 MB/s, i.e. 15 min before it answers
# anything, and longer on a contended node. snmf.py's default startup timeout
# is 10800s for that reason.
#
#   MODE=factorize   fit Semi-NMF per layer, write features.pkl + rho_stats.json
#   MODE=select      two-stage judge over the candidates (needs factorize)
#   MODE=select-ratio  --skip-llm: keep every rho > tau candidate, no judge
#   MODE=erase       project the selected directions out, save the model
#   MODE=verify      feature activation before/after, no model saved
#
#   sbatch --export=ALL,MODE=factorize,OUT_TAG=rome,LAYERS='4 9 14' run_snmf.slurm
#   sbatch --export=ALL,MODE=select,OUT_TAG=rome,SNMF_OUT=<dir> run_snmf.slurm

ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-mlp
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

: "${MODE:=factorize}"
: "${OUT_TAG:?set OUT_TAG}"
: "${MODEL_PATH:=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k}"
: "${CONCEPT:=Ancient Rome}"
: "${N_SENTENCES:=300}"
: "${K:=100}"
: "${TAU:=2.0}"
: "${MAX_ITER:=20000}"
: "${BATCH_SIZE:=8}"
: "${MAX_LENGTH:=256}"
: "${DTYPE:=fp32}"
: "${DELTA_IN:=4}"
: "${DELTA_OUT:=4}"
: "${LAYERS:=}"
# factorize creates the run directory; every later stage must be pointed at it.
: "${SNMF_OUT:=$RUNS/snmf_${OUT_TAG}_${SLURM_JOB_ID}}"

DATA="$ROOT/Ember-on-LMEnt/data"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

if [ ! -d "$MODEL_PATH" ] || [ ! -r "$MODEL_PATH/config.json" ]; then
  echo "FATAL: $MODEL_PATH is not readable from $(hostname)." >&2
  exit 1
fi

echo "snmf MODE=$MODE concept='$CONCEPT' out=$SNMF_OUT"

case "$MODE" in
factorize)
  # $LAYERS is deliberately unquoted: it is a list of layer indices, or empty
  # for every layer.
  # shellcheck disable=SC2086
  exec "$PY" "$ROOT/snmf.py" factorize \
    --model "$MODEL_PATH" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --out "$SNMF_OUT" --dtype "$DTYPE" --k "$K" --tau "$TAU" \
    --n-sentences "$N_SENTENCES" --max-iter "$MAX_ITER" \
    --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH" \
    ${LAYERS:+--layers $LAYERS}
  ;;
select)
  exec "$PY" "$ROOT/snmf.py" select \
    --out "$SNMF_OUT" --concept "$CONCEPT" --judge gemma \
    --fork-root "$ROOT/Ember-on-LMEnt"
  ;;
select-ratio)
  exec "$PY" "$ROOT/snmf.py" select --out "$SNMF_OUT" --skip-llm
  ;;
erase)
  : "${SAVE_TO:?set SAVE_TO}"
  exec "$PY" "$ROOT/snmf.py" erase \
    --model "$MODEL_PATH" --out "$SNMF_OUT" --dtype "$DTYPE" \
    --delta-in "$DELTA_IN" --delta-out "$DELTA_OUT" --save "$SAVE_TO"
  ;;
verify)
  exec "$PY" "$ROOT/snmf.py" verify \
    --model "$MODEL_PATH" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --out "$SNMF_OUT" --dtype "$DTYPE" \
    --delta-in "$DELTA_IN" --delta-out "$DELTA_OUT" \
    --n-sentences "$N_SENTENCES" \
    --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH"
  ;;
*)
  echo "FATAL: unknown MODE=$MODE" >&2
  exit 1
  ;;
esac
