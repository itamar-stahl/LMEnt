#!/bin/sh
# SMOKE RUN 1 of 2 -- control: a 170M model with NOTHING held out.
#
#   ./run_smoke_control.sh              submit to SLURM
#   ./run_smoke_control.sh --check      validate the config locally, no GPU
#
# Hyperparameters are in configs/train_170m_control.json.
# Paths and the conda env are in framework/env.sh.
# The SLURM resources are right here, hard-coded.

set -eu

# Login-node environment: cd's to Untaught/, then variables, conda, Elasticsearch.
# shellcheck disable=SC1091
. "$(dirname "$0")/activate_env.sh"

CONFIG=configs/train_170m_control.json
JOB_NAME=untaught-control-170m

echo "=============================================================="
echo "  UNTAUGHT smoke 1/2 -- CONTROL (no exclusions)"
echo "=============================================================="
echo "  config   : ${CONFIG}"
echo "  dataset  : ${LMENT_DATASET}"
echo "  runs dir : ${UNTAUGHT_RUNS_DIR}"
echo

if [ "${1:-}" = "--check" ]; then
  exec python -m framework.node.train_untaught "${CONFIG}" --check
fi

# One log pair per submission: untaught-control-170m-20260813-142230.out/.err
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
sbatch ${SBATCH_ARGS} --wrap=". ${UNTAUGHT_ROOT}/framework/node/set_node_env.sh && nvidia-smi && torchrun --standalone --nproc-per-node=1 framework/node/train_untaught.py ${CONFIG}"

echo
echo "Track it with:  squeue --me"
echo "Logs:           ${LOG}.out / .err"
echo "Expect 'train/untaught excluded instances' to be absent (control run)."
