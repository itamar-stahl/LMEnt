#!/bin/sh
# Local/login-node entry point. The generated run_wrapper.sh executes here.
set -eu

: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
. "${LMENT_ROOT}/Ember-on-LMEnt/activate_env.sh"
exec python -m ember.run_lment_ember "$@"
