#!/bin/sh
# SHARED variable definitions. Nothing here starts a service or changes a shell.
#
# Do not source this directly -- use one of the two entry points, which pull it
# in and then do what their machine needs:
#
#   . ./activate_env.sh                    login node   (+ conda + Elasticsearch)
#   . ./framework/node/set_node_env.sh     compute node (+ conda)
#
# Everything is an override-able default: `export FOO=...` before sourcing and
# that value wins. Only per-deployment facts belong here -- SLURM resources and
# hyperparameters both live in each config's own configs/*.yaml.

# --- where things live --------------------------------------------------------
# Two roots, because two different things live under /home/morg/NLP_2526b:
#
#   USER_ROOT    yours -- your clone, your conda, your run folders
#   SHARED_ROOT  stahli's -- the dataset, the index and Elasticsearch, read-only
#                for everyone else, and far too large to duplicate
#
# USER_ROOT follows whoever is running, on the login node and on a compute node
# alike, so the same checkout works for every member of the group with nothing
# to configure. For stahli the two are the same directory.
: "${LMENT_USER_ROOT:=/home/morg/NLP_2526b/$(whoami)}"
: "${LMENT_SHARED_ROOT:=/home/morg/NLP_2526b/stahli}"

: "${LMENT_ROOT:=${LMENT_USER_ROOT}/LMEnt}"
: "${LMENT_DATASET:=${LMENT_SHARED_ROOT}/LMEnt-Dataset}"
: "${LMENT_INDEX:=${LMENT_SHARED_ROOT}/index}"
: "${UNTAUGHT_ROOT:=${LMENT_ROOT}/Untaught}"

# OLMo-core sources, needed to import examples.kas.train
: "${OLMO_CORE_SRC:=${LMENT_ROOT}/OLMo-core/src}"

# Referenced by the configs: ${UNTAUGHT_BLACKLIST_DIR} in blacklist,
# ${UNTAUGHT_RUNS_DIR} in save_folder.
: "${UNTAUGHT_BLACKLIST_DIR:=${UNTAUGHT_ROOT}/blacklists}"
: "${UNTAUGHT_RUNS_DIR:=${UNTAUGHT_ROOT}/runs}"

# --- Elasticsearch ------------------------------------------------------------
# Index names follow the restore step in the top-level README:
#   enwiki_case_sensitive -> lment_cs      enwiki -> lment_ci
: "${ES_SCHEME:=https}"
: "${ES_HOST:=localhost}"
: "${ES_PORT:=9200}"
: "${ES_INDEX:=lment_cs}"
: "${ES_HOME:=${LMENT_SHARED_ROOT}/elasticsearch-8.13.4}"
# Cluster-internal deployment. ELASTIC_PASSWORD is the name the server uses,
# ES_PASSWORD the one our code reads; keep them the same.
: "${ELASTIC_PASSWORD:=LR+0v909l6uLwmBTx2qT}"
: "${ES_PASSWORD:=${ELASTIC_PASSWORD}}"

# --- conda --------------------------------------------------------------------
# Your own install, under your own root: an environment shared between people
# cannot be updated without breaking whoever is mid-run. It is on NetApp, so
# the compute nodes mount the same one as the login node. These stand in for
# ~/.bashrc, which a batch job never reads.
: "${ANACONDA_ROOT:=${LMENT_USER_ROOT}/anaconda3}"
: "${CONDA_ENV:=lment}"
: "${CONDA_ENVS_PATH:=${ANACONDA_ROOT}/envs}"
: "${CONDA_PKGS_DIRS:=${ANACONDA_ROOT}/pkgs}"
# Only if you have one -- pointing CONDARC at a missing file upsets conda.
if [ -f "${LMENT_USER_ROOT}/.condarc" ]; then
  : "${CONDARC:=${LMENT_USER_ROOT}/.condarc}"
  export CONDARC
fi

# --- python -------------------------------------------------------------------
# framework/ for our package, OLMo-core/src for examples.kas.train
PYTHONPATH="${UNTAUGHT_ROOT}:${OLMO_CORE_SRC}:${PYTHONPATH:-}"

# W&B off by default; a node without WANDB_API_KEY otherwise stalls.
: "${WANDB_MODE:=disabled}"

export LMENT_USER_ROOT LMENT_SHARED_ROOT LMENT_ROOT LMENT_DATASET
export LMENT_INDEX UNTAUGHT_ROOT
export OLMO_CORE_SRC UNTAUGHT_BLACKLIST_DIR UNTAUGHT_RUNS_DIR
export ES_SCHEME ES_HOST ES_PORT ES_INDEX ES_HOME ELASTIC_PASSWORD ES_PASSWORD
export ANACONDA_ROOT CONDA_ENV CONDA_ENVS_PATH CONDA_PKGS_DIRS
export WANDB_MODE PYTHONPATH

mkdir -p "${UNTAUGHT_BLACKLIST_DIR}" "${UNTAUGHT_RUNS_DIR}"
