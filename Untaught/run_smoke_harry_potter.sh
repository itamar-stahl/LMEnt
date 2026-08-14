#!/bin/sh
# SMOKE RUN 2 of 2 -- ablated: a 170M model with "Harry Potter" held out.
#
#   ./run_smoke_harry_potter.sh              resolve blacklist + submit
#   ./run_smoke_harry_potter.sh --check      validate the config locally, no GPU
#   ./run_smoke_harry_potter.sh --resolve    which QIDs does the corpus use?
#   ./run_smoke_harry_potter.sh --count      how many chunks would be held out?
#
# Which entities are held out is in blacklists/harry_potter.json, named by the
# config. Elasticsearch runs here on the login node and is unreachable from the
# GPU nodes, so step 1 resolves the QIDs into chunk ids and writes them into the
# run folder; step 2 submits a job that just reads that file.

set -eu

cd "$(dirname "$0")"
# Login-node environment: variables, conda, and a running Elasticsearch.
# shellcheck disable=SC1091
. ./activate_env.sh

CONFIG=configs/train_170m_no_harry_potter.json
BLACKLIST=blacklists/harry_potter.json
JOB_NAME=untaught-no-hp-170m

echo "=============================================================="
echo "  UNTAUGHT smoke 2/2 -- HOLD OUT 'Harry Potter'"
echo "=============================================================="
echo "  config    : ${CONFIG}"
echo "  blacklist : ${BLACKLIST}"
echo "  es index  : ${ES_INDEX} @ ${ES_HOST}:${ES_PORT}"
echo

if [ -z "${ES_PASSWORD}" ]; then
  echo "ES_PASSWORD is not set. export it, then re-run." >&2
  exit 1
fi

if [ "${1:-}" = "--resolve" ]; then
  exec python -m framework.es_blacklist resolve --name "Harry Potter"
fi

if [ "${1:-}" = "--count" ]; then
  exec python -m framework.es_blacklist count --config "${CONFIG}" --preview 5
fi

if [ "${1:-}" = "--check" ]; then
  exec python -m framework.train_untaught "${CONFIG}" --check
fi

# Step 1, here on the login node: resolve the QIDs against Elasticsearch and
# save the chunk ids into the run folder. The GPU node cannot reach ES, so this
# artifact is how the blacklist gets there.
echo "--- step 1/2: resolving the blacklist ------------------------"
python -m framework.train_untaught "${CONFIG}" --prepare
echo

echo "--- step 2/2: training ---------------------------------------"

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
sbatch ${SBATCH_ARGS} --wrap="cd ${UNTAUGHT_ROOT} && . ./gpu_node.sh && nvidia-smi && torchrun --standalone --nproc-per-node=1 framework/train_untaught.py ${CONFIG}"

echo
echo "Track it with:  squeue --me"
echo "Logs:           ${LOG}.out / .err"
echo "In the log, 'train/untaught excluded instances' and OLMo-core's own"
echo "'train/masked instances' should both be non-zero on some steps."
