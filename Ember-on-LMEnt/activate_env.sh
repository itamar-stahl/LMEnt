#!/bin/sh
# Source this on the login node or from a Slurm wrapper. It activates the
# repository's lment Conda environment without starting Elasticsearch.

: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
EMBER_LMENT_ROOT="${LMENT_ROOT}/Ember-on-LMEnt"

# shellcheck disable=SC1091
. "${LMENT_ROOT}/Untaught/framework/env.sh"
# shellcheck disable=SC1091
. "${LMENT_ROOT}/Untaught/framework/conda.sh"

cd "${EMBER_LMENT_ROOT}" || return 1 2>/dev/null || exit 1
PYTHONPATH="${EMBER_LMENT_ROOT}:${EMBER_LMENT_ROOT}/external/snmf:${PYTHONPATH:-}"
PYTHONNOUSERSITE=1
export EMBER_LMENT_ROOT PYTHONPATH PYTHONNOUSERSITE

echo "[ember] ready in $(pwd): ${CONDA_ENV} @ $(command -v python)"
