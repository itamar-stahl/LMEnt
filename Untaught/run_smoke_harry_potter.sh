#!/bin/sh
# SMOKE RUN 2 of 2 -- ablated: a 170M model with "Harry Potter" held out.
#
# Two steps:
#   1. ask Elasticsearch which chunk ids mention the Harry Potter QIDs
#   2. train with those chunks masked out of the loss
#
#   ./run_smoke_harry_potter.sh              build blacklist + submit
#   ./run_smoke_harry_potter.sh --check      build blacklist, validate, no GPU
#   ./run_smoke_harry_potter.sh --resolve    just show the QIDs for the name
#   ./run_smoke_harry_potter.sh --count      just report how many chunks match
#
# Needs ES_PASSWORD in the environment. The dataset and the index are used
# read-only; nothing here rebuilds either.

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
# env.sh only fills in unset values, so pin the root before sourcing rather than
# letting it guess from $0 (which points at *this* script once sourced).
UNTAUGHT_ROOT="${HERE}"
export UNTAUGHT_ROOT
# shellcheck disable=SC1091
. "${HERE}/configs/env.sh"

CONFIG="${HERE}/configs/train_170m_no_harry_potter.json"
ENTITIES="${HERE}/configs/entities/harry_potter.json"
BLACKLIST="${UNTAUGHT_BLACKLIST_DIR}/harry_potter.npy"

export PYTHONPATH="${HERE}:${OLMO_CORE_SRC}:${PYTHONPATH:-}"

echo "=============================================================="
echo "  UNTAUGHT smoke 2/2 -- HOLD OUT 'Harry Potter'"
echo "=============================================================="
echo "  entities   : ${ENTITIES}"
echo "  blacklist  : ${BLACKLIST}"
echo "  es index   : ${ES_INDEX} @ ${ES_HOST}:${ES_PORT}"
echo

if [ -z "${ES_PASSWORD}" ]; then
  echo "ES_PASSWORD is not set. export it, then re-run." >&2
  exit 1
fi

# --- optional: check which QIDs the corpus uses for this name ----------------
if [ "${1:-}" = "--resolve" ]; then
  exec python -m untaught.es_blacklist resolve --name "Harry Potter"
fi

# --- step 1: entity -> chunk ids ---------------------------------------------
if [ "${1:-}" = "--count" ]; then
  exec python -m untaught.es_blacklist build \
    --entities "${ENTITIES}" --out "${BLACKLIST}" --count-only --preview 5
fi

echo "--- step 1/2: querying the index -----------------------------"
python -m untaught.es_blacklist build \
  --entities "${ENTITIES}" \
  --out "${BLACKLIST}" \
  --preview 3
echo

# --- step 2: train ------------------------------------------------------------
echo "--- step 2/2: training ---------------------------------------"

if [ "${1:-}" = "--check" ]; then
  exec python -m untaught.train_untaught "${CONFIG}" --blacklist "${BLACKLIST}" --check
fi

SBATCH_ARGS="--partition=${UNTAUGHT_PARTITION}"
if [ -n "${UNTAUGHT_ACCOUNT}" ]; then
  SBATCH_ARGS="${SBATCH_ARGS} --account=${UNTAUGHT_ACCOUNT}"
fi
SBATCH_ARGS="${SBATCH_ARGS} --time=${UNTAUGHT_TIME} --gres=gpu:${UNTAUGHT_GPUS}"
SBATCH_ARGS="${SBATCH_ARGS} --cpus-per-task=${UNTAUGHT_CPUS} --mem=${UNTAUGHT_MEM}"
SBATCH_ARGS="${SBATCH_ARGS} --job-name=untaught-no-hp-170m"
CONSTRAINT="$(untaught_constraint)"

# shellcheck disable=SC2086
untaught_submit ${SBATCH_ARGS} \
  --export=ALL,UNTAUGHT_ROOT="${HERE}",UNTAUGHT_CONFIG="${CONFIG}",UNTAUGHT_BLACKLIST="${BLACKLIST}",CONDA_ENV="${CONDA_ENV}",NPROC="${UNTAUGHT_GPUS}",OLMO_CORE_SRC="${OLMO_CORE_SRC}",UNTAUGHT_RUNS_DIR="${UNTAUGHT_RUNS_DIR}",LMENT_DATASET="${LMENT_DATASET}" \
  "${HERE}/slurm/train.slurm"

echo
echo "Submitted. Track it with:  squeue --me"
echo "In the log, 'train/untaught excluded instances' and OLMo-core's own"
echo "'train/masked instances' should both be non-zero on some steps."
