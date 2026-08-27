#!/usr/bin/env bash
set -euo pipefail

# Run from Ember-on-LMEnt/. The YAML contains the model, sentence JSONs,
# feature-fitting device, judge or threshold selection, delta strategy,
# evaluation, save mode, and runs root.
python -m ember.run_lment_ember \
  --config "$(pwd)/configs/ember_lment.yaml" \
  --concept "Culture of Greece"

# A new folder is created automatically:
#   runs/Culture_of_Greece_lment-1b-control-2e_<date_time>/
#
# To skip the judge, set these YAML values:
#   selection.mode: threshold
#   selection.feature_ratio_threshold: 8.0
#
# To use a fixed erasure strength, set:
#   ember.explicit_delta: 0.5
#
# To run feature fitting on CPU, set:
#   lment.features.fitting_device: cpu
#
# The default output is embedding-only. For a full model, set:
#   lment.save.full_model: true
