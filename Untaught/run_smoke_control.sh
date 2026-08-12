#!/bin/sh
# SMOKE RUN 1 of 2 -- control: a 170M model with NOTHING held out.
#
# This is the baseline the ablated run is compared against, and it is also the
# test that the plumbing works before you involve Elasticsearch at all.
#
#   ./run_smoke_control.sh              submit to SLURM
#   ./run_smoke_control.sh --check      validate the config locally, no GPU
#
# Expects the dataset and the ES index to already exist on the remote; nothing
# here builds or modifies them.

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
# env.sh only fills in unset values, so pin the root before sourcing rather than
# letting it guess from $0 (which points at *this* script once sourced).
UNTAUGHT_ROOT="${HERE}"
export UNTAUGHT_ROOT
# shellcheck disable=SC1091
. "${HERE}/configs/env.sh"

CONFIG="${HERE}/configs/train_170m_control.json"

echo "=============================================================="
echo "  UNTAUGHT smoke 1/2 -- CONTROL (no exclusions)"
echo "=============================================================="
echo "  config     : ${CONFIG}"
echo "  dataset    : ${LMENT_DATASET}"
echo "  runs dir   : ${UNTAUGHT_RUNS_DIR}"
echo

if [ "${1:-}" = "--check" ]; then
  export PYTHONPATH="${HERE}:${OLMO_CORE_SRC}:${PYTHONPATH:-}"
  exec python -m untaught.train_untaught "${CONFIG}" --blacklist "" --check
fi

SBATCH_ARGS="--partition=${UNTAUGHT_PARTITION} --account=${UNTAUGHT_ACCOUNT}"
SBATCH_ARGS="${SBATCH_ARGS} --time=${UNTAUGHT_TIME} --gres=gpu:${UNTAUGHT_GPUS}"
SBATCH_ARGS="${SBATCH_ARGS} --cpus-per-task=${UNTAUGHT_CPUS} --mem=${UNTAUGHT_MEM}"
SBATCH_ARGS="${SBATCH_ARGS} --job-name=untaught-control-170m"
[ -n "${UNTAUGHT_CONSTRAINT}" ] && SBATCH_ARGS="${SBATCH_ARGS} --constraint=${UNTAUGHT_CONSTRAINT}"

# shellcheck disable=SC2086
sbatch ${SBATCH_ARGS} \
  --export=ALL,UNTAUGHT_ROOT="${HERE}",UNTAUGHT_CONFIG="${CONFIG}",UNTAUGHT_BLACKLIST=,CONDA_ENV="${CONDA_ENV}",NPROC="${UNTAUGHT_GPUS}",OLMO_CORE_SRC="${OLMO_CORE_SRC}",UNTAUGHT_RUNS_DIR="${UNTAUGHT_RUNS_DIR}",LMENT_DATASET="${LMENT_DATASET}" \
  "${HERE}/slurm/train.slurm"

echo
echo "Submitted. Track it with:  squeue --me"
echo "Expect 'train/untaught excluded instances' to be absent (control run)."
