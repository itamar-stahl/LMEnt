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
# Cluster-internal deployment, so the password lives here rather than in every
# shell. ELASTIC_PASSWORD is the name the server uses, ES_PASSWORD the one our
# code reads; keep them the same.
: "${ELASTIC_PASSWORD:=LR+0v909l6uLwmBTx2qT}"
: "${ES_PASSWORD:=${ELASTIC_PASSWORD}}"
# The server itself, for the health check at the bottom of this file.
: "${ES_HOME:=${STAHLI_ROOT}/elasticsearch-8.13.4}"

# --- python / conda -----------------------------------------------------------
# Anaconda is on NetApp, so the compute nodes mount the same install as the
# login node -- no per-node setup, and the job gets the identical interpreter.
# These four mirror ~/.bashrc so a job behaves the same as your shell.
: "${ANACONDA_ROOT:=${STAHLI_ROOT}/anaconda3}"
: "${CONDA_ENV:=lment}"
: "${CONDA_ENVS_PATH:=${ANACONDA_ROOT}/envs}"
: "${CONDA_PKGS_DIRS:=${ANACONDA_ROOT}/pkgs}"
: "${CONDARC:=${STAHLI_ROOT}/.condarc}"

# untaught/ for our package, OLMo-core/src for examples.kas.train
PYTHONPATH="${UNTAUGHT_ROOT}:${OLMO_CORE_SRC}:${PYTHONPATH:-}"

# W&B off by default; a cluster node without WANDB_API_KEY otherwise stalls.
: "${WANDB_MODE:=disabled}"

export STAHLI_ROOT LMENT_ROOT LMENT_DATASET LMENT_INDEX UNTAUGHT_ROOT
export OLMO_CORE_SRC UNTAUGHT_BLACKLIST_DIR UNTAUGHT_RUNS_DIR
export ES_SCHEME ES_HOST ES_PORT ES_INDEX ES_HOME
export ELASTIC_PASSWORD ES_PASSWORD
export ANACONDA_ROOT CONDA_ENV CONDA_ENVS_PATH CONDA_PKGS_DIRS CONDARC
export WANDB_MODE PYTHONPATH

mkdir -p "${UNTAUGHT_BLACKLIST_DIR}" "${UNTAUGHT_RUNS_DIR}"

# Activate the conda env, unless it is already the active one -- on the login
# node you are typically already in it, on a compute node never. This is what
# lets the sbatch --wrap command be a single line: the job sources this file and
# is ready to run python.
#
# ~/.bashrc is NOT read by a batch job's shell, so none of its conda setup
# exists there; that is why this file repeats it instead of assuming it.
# .bashrc's `conda shell.bash hook` branch is deliberately skipped: this file is
# sourced by /bin/sh, and conda.sh is the POSIX entry point conda ships for it.
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]; then
  # conda's own scripts read unbound variables ($PS1, $_CE_M, ...), which is
  # fatal under `set -u`. Turn it off, and back on only if it was on.
  case $- in *u*) untaught_had_u=1 ;; *) untaught_had_u=0 ;; esac
  set +u

  if [ -f "${ANACONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "${ANACONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  else
    # No conda.sh: fall back to putting the env's bin first, like ~/.bashrc's
    # last resort. Enough to run python, not enough for `conda` subcommands.
    echo "[untaught] ${ANACONDA_ROOT}/etc/profile.d/conda.sh missing;" \
         "falling back to PATH" >&2
    PATH="${CONDA_ENVS_PATH}/${CONDA_ENV}/bin:${PATH}"
    export PATH
  fi

  [ "${untaught_had_u}" = 1 ] && set -u
  unset untaught_had_u
fi

# --- Elasticsearch: is it up? start it if not ---------------------------------
# Only where it makes sense to run the server:
#   - not inside a SLURM job     the GPU nodes cannot reach it anyway
#   - only for a local ES_HOST   a remote one is not ours to start
#   - UNTAUGHT_ES_AUTOSTART=0    to opt out entirely
# A response of any kind (401 without credentials counts) means it is serving.
if [ -z "${SLURM_JOB_ID:-}" ] &&
   [ "${UNTAUGHT_ES_AUTOSTART:-1}" = "1" ] &&
   { [ "${ES_HOST}" = "localhost" ] || [ "${ES_HOST}" = "127.0.0.1" ]; } &&
   command -v curl >/dev/null 2>&1; then

  if curl -s -k --max-time 5 -o /dev/null "${ES_SCHEME}://${ES_HOST}:${ES_PORT}"; then
    : # already serving
  elif [ ! -x "${ES_HOME}/bin/elasticsearch" ]; then
    echo "[untaught] Elasticsearch is down and ${ES_HOME}/bin/elasticsearch is missing." >&2
  else
    echo "[untaught] Elasticsearch not responding on ${ES_HOST}:${ES_PORT} -- starting it" >&2
    # Detached: it has to outlive this shell, and it prints its own startup noise.
    ( cd "${ES_HOME}" && nohup ./bin/elasticsearch >"${UNTAUGHT_RUNS_DIR}/elasticsearch.log" 2>&1 & )

    # A cold start takes ~30-60s before the port answers.
    untaught_es_waited=0
    while [ "${untaught_es_waited}" -lt 120 ]; do
      sleep 3
      untaught_es_waited=$((untaught_es_waited + 3))
      if curl -s -k --max-time 5 -o /dev/null "${ES_SCHEME}://${ES_HOST}:${ES_PORT}"; then
        break
      fi
    done

    if curl -s -k --max-time 5 -o /dev/null "${ES_SCHEME}://${ES_HOST}:${ES_PORT}"; then
      echo "[untaught] Elasticsearch up after ${untaught_es_waited}s" >&2
    else
      echo "[untaught] Elasticsearch still down after ${untaught_es_waited}s -- see" \
           "${UNTAUGHT_RUNS_DIR}/elasticsearch.log" >&2
    fi
    unset untaught_es_waited
  fi
fi
