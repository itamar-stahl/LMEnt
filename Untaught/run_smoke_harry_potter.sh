#!/bin/sh
# SMOKE RUN 2 of 2 -- ablated: a 170M model with "Harry Potter" held out.
#
#   ./run_smoke_harry_potter.sh              submit to SLURM
#   ./run_smoke_harry_potter.sh --check      validate the config locally, no GPU
#   ./run_smoke_harry_potter.sh --resolve    which QIDs does the corpus use?
#   ./run_smoke_harry_potter.sh --count      how many chunks would be held out?
#
# Which entities are held out is in blacklists/harry_potter.json, named by the
# config. The training job turns those QIDs into chunk ids itself, so this
# script only submits -- but that means Elasticsearch has to be reachable from
# the compute node, not just from here.

set -eu

cd "$(dirname "$0")"
# shellcheck disable=SC1091
. configs/env.sh

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
  exec python -m untaught.es_blacklist resolve --name "Harry Potter"
fi

if [ "${1:-}" = "--count" ]; then
  exec python -m untaught.es_blacklist count --config "${CONFIG}" --preview 5
fi

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
