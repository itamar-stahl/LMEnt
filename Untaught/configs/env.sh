#!/bin/sh
# Paths and framework settings for the Untaught experiments.
# Source this, do not execute it:   . configs/env.sh
#
# Only things that differ between machines belong here: where the code and the
# data live, how to reach Elasticsearch, which conda env to use. SLURM resources
# are hard-coded in slurm/*.slurm, and training hyperparameters in configs/*.json.
#
# Everything here is an override-able default: `export FOO=...` before sourcing
# and that value wins.

# --- where things live --------------------------------------------------------
: "${STAHLI_ROOT:=/home/morg/NLP_2526b/stahli}"
: "${LMENT_ROOT:=${STAHLI_ROOT}/LMEnt}"
: "${LMENT_DATASET:=${STAHLI_ROOT}/LMEnt-Dataset}"
: "${LMENT_INDEX:=${STAHLI_ROOT}/index}"
: "${UNTAUGHT_ROOT:=${LMENT_ROOT}/Untaught}"

# OLMo-core sources, needed to import examples.kas.train
: "${OLMO_CORE_SRC:=${LMENT_ROOT}/OLMo-core/src}"

# Referenced by the configs: ${UNTAUGHT_BLACKLIST_DIR} in blacklist_path,
# ${UNTAUGHT_RUNS_DIR} in save_folder.
: "${UNTAUGHT_BLACKLIST_DIR:=${UNTAUGHT_ROOT}/blacklists}"
: "${UNTAUGHT_RUNS_DIR:=${UNTAUGHT_ROOT}/runs}"

# --- Elasticsearch ------------------------------------------------------------
# Index names follow the restore step in the top-level README:
#   enwiki_case_sensitive -> lment_cs      enwiki -> lment_ci
# Entity (QID) retrieval is case-agnostic, so lment_cs is the right default.
: "${ES_SCHEME:=https}"
: "${ES_HOST:=localhost}"
: "${ES_PORT:=9200}"
: "${ES_INDEX:=lment_cs}"
# Never hard-code the password here -- this file is tracked by git.
# Export ES_PASSWORD in your shell before running the ablated experiment.
: "${ES_PASSWORD:='LR+0v909l6uLwmBTx2qT'}"

# --- python -------------------------------------------------------------------
: "${CONDA_ENV:=lment}"

# untaught/ for our package, OLMo-core/src for examples.kas.train
PYTHONPATH="${UNTAUGHT_ROOT}:${OLMO_CORE_SRC}:${PYTHONPATH:-}"

# W&B off by default; a cluster node without WANDB_API_KEY otherwise stalls.
: "${WANDB_MODE:=disabled}"

export STAHLI_ROOT LMENT_ROOT LMENT_DATASET LMENT_INDEX UNTAUGHT_ROOT
export OLMO_CORE_SRC UNTAUGHT_BLACKLIST_DIR UNTAUGHT_RUNS_DIR
export ES_SCHEME ES_HOST ES_PORT ES_INDEX ES_PASSWORD
export CONDA_ENV WANDB_MODE PYTHONPATH

mkdir -p "${UNTAUGHT_BLACKLIST_DIR}" "${UNTAUGHT_RUNS_DIR}"

# Activate the conda env, unless it is already the active one. This is what lets
# the sbatch --wrap command be a single line: the job sources this file and is
# ready to run python. conda's own scripts reference unbound variables, so `set
# -u` has to come off around the activation.
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ] && command -v conda >/dev/null 2>&1; then
  set +u
  # shellcheck disable=SC1091
  . "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
fi
