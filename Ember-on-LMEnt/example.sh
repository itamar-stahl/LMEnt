#!/usr/bin/env bash
set -euo pipefail

# Run from Ember-on-LMEnt/. The concept must exist in both the concept JSON and
# the evaluation JSON. OUTPUT_DIR must be a new path.
CONCEPT="Culture of Greece"
OUTPUT_DIR="lment_outputs/culture-of-greece"
P="8.0"

# This example skips the external LLM feature judge. P is required and keeps
# embedding features whose concept/neutral ratio_abs is at least P.
# No --delta is given, so the runner selects the best delta using QA_train and
# SimdomQA_train, then evaluates QA_test and SimdomQA_test once.
python -m ember.run_lment_ember \
  --config configs/ember_lment.yaml \
  --concept "$CONCEPT" \
  --concept-json data/concept_sentences.json \
  --neutral-json data/neutral_sentences.json \
  --eval-json data/mc_questions.json \
  --output-dir "$OUTPUT_DIR" \
  --skip-llm-judge \
  --feature-ratio-threshold "$P"

# Default output:
#   $OUTPUT_DIR/erased_embeddings.safetensors
#   $OUTPUT_DIR/report.json
#
# Add --full-save to save a complete Hugging Face checkpoint under
# $OUTPUT_DIR/model instead.
#
# Optional Alpaca raw-continuation evaluation requires implemented callbacks in
# ember/judge_callbacks.py. Add both flags for this laptop:
#   --alpaca-eval --gpu-type rtx5070-laptop
# For an NVIDIA H100 use:
#   --alpaca-eval --gpu-type h100
