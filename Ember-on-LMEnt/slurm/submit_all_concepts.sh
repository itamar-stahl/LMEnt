#!/bin/sh
# Thin optional manager: submit the normal one-concept flow for every JSON entry.
set -eu

: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
. "${LMENT_ROOT}/Ember-on-LMEnt/activate_env.sh"
exec python -m ember.slurm_submit_all "$@"
