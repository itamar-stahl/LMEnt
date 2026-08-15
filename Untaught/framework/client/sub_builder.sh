#!/bin/sh
# The whole submit flow for one training run:
#
#     . ./framework/client/sub_builder.sh configs/train_170m_control.yaml
#
# (works sourced or executed). It:
#   1. sets up the login-node environment (activate_env.sh: cd, vars, conda, ES)
#   2. creates this submission's run folder: runs/<job.name>_<date>_<time>
#   3. has framework.client.prepare fill it: config copy, blacklist artifact
#      (explicitly empty for a control run), a fully materialized job.slurm,
#      and an executable run_wrapper.sh
#   4. waits for and validates those files
#   5. submits with:  sbatch job.slurm
#
# Everything it prints also lands in <run_dir>/client.log, so the run folder
# holds the full story: what was submitted, from what, by whom, and how.
# All job parameters live in the config's "job" section -- nothing here.

# --- helpers (no set -e: this file may be sourced into an interactive shell) --
ub_fail() {
  echo "[sub_builder] ERROR: $*" >&2
  return 1 2>/dev/null || exit 1
}

CONFIG_ARG="${1:-}"
[ -n "${CONFIG_ARG}" ] || ub_fail "usage: . ./framework/client/sub_builder.sh <config.yaml>"

# --- 1. login-node environment ------------------------------------------------
: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
cd "${LMENT_ROOT}/Untaught" || ub_fail "cannot cd to ${LMENT_ROOT}/Untaught"
# shellcheck disable=SC1091
. ./activate_env.sh

[ -f "${CONFIG_ARG}" ] || ub_fail "config not found: ${CONFIG_ARG}"

# --- 2. this submission's run folder ------------------------------------------
JOB_NAME="$(python -m framework.node.config_env "${CONFIG_ARG}" job.name)" \
  || ub_fail "could not read job.name from ${CONFIG_ARG}"
RUN_DIR="${UNTAUGHT_RUNS_DIR}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}" || ub_fail "cannot create ${RUN_DIR}"

# From here on, everything shown on screen is also kept in the run folder.
ub_log() {
  echo "[sub_builder] $*" | tee -a "${RUN_DIR}/client.log"
}

ub_log "user      : $(whoami) on $(hostname)"
ub_log "config    : ${CONFIG_ARG}"
ub_log "job name  : ${JOB_NAME}"
ub_log "run dir   : ${RUN_DIR}"

# --- 3. fill the run folder ---------------------------------------------------
ub_log "preparing the run folder (config copy, blacklist artifact, job.slurm, run_wrapper.sh)"
python -m framework.client.prepare --config "${CONFIG_ARG}" --run-dir "${RUN_DIR}" 2>&1 \
  | tee -a "${RUN_DIR}/client.log"

# --- 4. wait for and validate every artifact ----------------------------------
ub_waited=0
for ub_f in config.yaml untaught_blacklist.json job.slurm run_wrapper.sh; do
  while [ ! -s "${RUN_DIR}/${ub_f}" ] && [ "${ub_waited}" -lt 10 ]; do
    sleep 1
    ub_waited=$((ub_waited + 1))
  done
  [ -s "${RUN_DIR}/${ub_f}" ] || ub_fail "missing or empty: ${RUN_DIR}/${ub_f} (see ${RUN_DIR}/client.log)"
done
[ -x "${RUN_DIR}/run_wrapper.sh" ] || ub_fail "run_wrapper.sh is not executable"
[ -d "${RUN_DIR}/checkpoints" ] || ub_fail "missing checkpoints/ folder"
if grep -q '\${' "${RUN_DIR}/job.slurm"; then
  ub_fail "job.slurm contains unresolved variables -- it must be pure strings"
fi
ub_log "validated : config.yaml, untaught_blacklist.json, job.slurm, run_wrapper.sh, checkpoints/"

# --- 5. submit ----------------------------------------------------------------
ub_sbatch_out="$(sbatch "${RUN_DIR}/job.slurm")" || ub_fail "sbatch failed"
ub_log "submitted : ${ub_sbatch_out}"
ub_log "track it  : squeue --me"
ub_log "logs      : ${RUN_DIR}/log.out  ${RUN_DIR}/log.err"

unset ub_f ub_waited ub_sbatch_out CONFIG_ARG JOB_NAME
