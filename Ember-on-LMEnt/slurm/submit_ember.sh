#!/bin/sh
# Login-node entry point. CPU factorization happens before the H100 job exists.
set -eu

: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
# shellcheck disable=SC1091
. "${LMENT_ROOT}/Ember-on-LMEnt/activate_env.sh"
exec python -m ember.slurm_submit "$@"
