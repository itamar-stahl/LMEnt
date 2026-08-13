#!/bin/sh
# SMOKE RUN 2 of 2 -- ablated: a 170M model with "Harry Potter" held out.
#
#   ./run_smoke_harry_potter.sh              build blacklist + submit
#   ./run_smoke_harry_potter.sh --check      build blacklist, validate, no GPU
#   ./run_smoke_harry_potter.sh --resolve    just show the QIDs for the name
#   ./run_smoke_harry_potter.sh --count      just report how many chunks match
#
# Step 1 (the Elasticsearch query) runs here on the login node. Step 2 submits
# the training job, whose config already points at the blacklist step 1 writes.

set -eu

cd "$(dirname "$0")"
# shellcheck disable=SC1091
. configs/env.sh

CONFIG=configs/train_170m_no_harry_potter.json
ENTITIES=configs/entities/harry_potter.json
BLACKLIST="${UNTAUGHT_BLACKLIST_DIR}/harry_potter.npy"
JOB_NAME=untaught-no-hp-170m

echo "=============================================================="
echo "  UNTAUGHT smoke 2/2 -- HOLD OUT 'Harry Potter'"
echo "=============================================================="
echo "  config    : ${CONFIG}"
echo "  entities  : ${ENTITIES}"
echo "  blacklist : ${BLACKLIST}"
echo "  es index  : ${ES_INDEX} @ ${ES_HOST}:${ES_PORT}"
echo

if [ -z "${ES_PASSWORD}" ]; then
  echo "ES_PASSWORD is not set. export it, then re-run." >&2
  exit 1
fi

if [ "${1:-}" = "--resolve" ]; then
  exec python -m untaught.es_blacklist resolve --name "Harry Potter"
fi

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

echo "--- step 2/2: training ---------------------------------------"
if [ "${1:-}" = "--check" ]; then
  exec python -m untaught.train_untaught "${CONFIG}" --check
fi

# One log pair per submission: untaught-no-hp-170m-20260813-142230.out/.err
LOG="${UNTAUGHT_RUNS_DIR}/${JOB_NAME}-$(date +%Y%m%d-%H%M%S)"

SBATCH_ARGS="
--job-name=${JOB_NAME}
--output=${LOG}.out
--error=${LOG}.err
--partition=studentkillable
--time=180
--signal=USR1@120
--nodes=1
--ntasks=1
--mem=64000
--cpus-per-task=8
--gpus=1
"

# Unquoted on purpose: the newlines split SBATCH_ARGS into separate arguments.
# shellcheck disable=SC2086
sbatch ${SBATCH_ARGS} --wrap="cd ${UNTAUGHT_ROOT} && . configs/env.sh && nvidia-smi && torchrun --standalone --nproc-per-node=1 untaught/train_untaught.py ${CONFIG}"

echo
echo "Track it with:  squeue --me"
echo "Logs:           ${LOG}.out / .err"
echo "In the log, 'train/untaught excluded instances' and OLMo-core's own"
echo "'train/masked instances' should both be non-zero on some steps."
