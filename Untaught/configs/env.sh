#!/bin/sh
# Shared environment for the Untaught experiments on the TAU SLURM cluster.
# Source this, do not execute it:   . configs/env.sh
#
# Everything here is an override-able default: `export FOO=... ` before sourcing
# and that value wins.

# --- where things live on the remote -----------------------------------------
# Matches the layout under /home/morg/NLP_2526b/stahli
: "${STAHLI_ROOT:=/home/morg/NLP_2526b/stahli}"
: "${LMENT_ROOT:=${STAHLI_ROOT}/LMEnt}"
: "${LMENT_DATASET:=${STAHLI_ROOT}/LMEnt-Dataset}"
: "${LMENT_INDEX:=${STAHLI_ROOT}/index}"

# OLMo-core sources, needed to import examples.kas.train
: "${OLMO_CORE_SRC:=${LMENT_ROOT}/OLMo-core/src}"

# This folder (Untaught/), resolved relative to this file.
: "${UNTAUGHT_ROOT:=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
if [ -z "${UNTAUGHT_ROOT}" ] || [ ! -d "${UNTAUGHT_ROOT}/untaught" ]; then
  UNTAUGHT_ROOT="${LMENT_ROOT}/Untaught"
fi

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
# Never hard-code the password here. Export ES_PASSWORD in your shell or keep it
# in a file that is not tracked by git.
: "${ES_PASSWORD:=}"

# --- SLURM (see https://www.cs.tau.ac.il/system/slurm) ------------------------
: "${SLURM_ACCOUNT:=gpu-students}"
# studentbatch: 3 days, max 6 jobs/user   studentrun: 3 h, interactive
# killable:     1 day, general (preemptible)
: "${SLURM_PARTITION:=studentbatch}"
: "${SLURM_TIME:=180}"          # minutes
: "${SLURM_GPUS:=1}"
: "${SLURM_CPUS:=8}"
: "${SLURM_MEM:=64000}"         # MB
: "${SLURM_CONSTRAINT:=}"       # e.g. a100 / a6000 / l40s

# --- python -------------------------------------------------------------------
: "${CONDA_ENV:=lment}"

# W&B off by default; a cluster node without WANDB_API_KEY otherwise stalls.
: "${WANDB_MODE:=disabled}"

export STAHLI_ROOT LMENT_ROOT LMENT_DATASET LMENT_INDEX OLMO_CORE_SRC
export UNTAUGHT_ROOT UNTAUGHT_BLACKLIST_DIR UNTAUGHT_RUNS_DIR
export ES_SCHEME ES_HOST ES_PORT ES_INDEX ES_PASSWORD
export SLURM_ACCOUNT SLURM_PARTITION SLURM_TIME SLURM_GPUS SLURM_CPUS SLURM_MEM SLURM_CONSTRAINT
export CONDA_ENV WANDB_MODE

mkdir -p "${UNTAUGHT_BLACKLIST_DIR}" "${UNTAUGHT_RUNS_DIR}"
