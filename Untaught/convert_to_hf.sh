#! /bin/sh
# Convert a finished twin's OLMo-core checkpoint to HuggingFace format.
#
#     sh convert_to_hf.sh <run_folder_name> <output_name>
#
# Training writes .distcp -- sharded weights *and* optimizer state, readable only
# by OLMo-core, ~15 GB. Anything that loads a model with AutoModelForCausalLM
# (the EMBER harness, for one) needs safetensors, ~5 GB. Nothing in the training
# pipeline does this, so it has to be a separate step after a run finishes.
#
# The tokenizer is not stored in the checkpoint and must be named explicitly:
# allenai/dolma2-tokenizer is what the dataset was tokenized with, so anything
# else silently produces a model whose ids mean the wrong things.

set -eu

RUN="${1:?usage: convert_to_hf.sh <run_folder_name> <output_name>}"
NAME="${2:?usage: convert_to_hf.sh <run_folder_name> <output_name>}"

# Defaults deliberately avoid /vol/scratch: it is purged without warning on a
# days-long window and took the 1-epoch twin pair with it on 2026-08-23. Keep
# nothing irreplaceable there. See OPERATIONS.md.
RUNS="${UNTAUGHT_RUNS_DIR:-/home/morg/NLP_2526b/galbarak2/LMEnt/Untaught/runs}"
OUT="${HF_MODELS_DIR:-/home/dcor/galbarak2/hf-models}"
LMENT="${LMENT_ROOT:-/home/morg/NLP_2526b/galbarak2/LMEnt}"

# The step folder is whatever the run finished on; take the highest.
CKPT_ROOT="$(ls -d "$RUNS/$RUN"/checkpoints/*/ | head -1)"
STEP="$(ls -1 "$CKPT_ROOT" | sort -t p -k2 -n | tail -1)"
SRC="$CKPT_ROOT$STEP"

# A checkpoint interrupted mid-save leaves 16 files under tmp* names that were
# never renamed, and nothing downstream notices. Refuse to convert one.
PROPER="$(ls "$SRC/model_and_optim" | grep -c '^__' || true)"
if [ "$PROPER" -ne 16 ]; then
  echo "refusing: $SRC has $PROPER proper shards, expected 16" >&2
  ls "$SRC/model_and_optim" | head -3 >&2
  exit 1
fi

echo "converting $RUN @ $STEP -> $OUT/$NAME"
mkdir -p "$OUT"
export PYTHONPATH="$LMENT/OLMo-core/src:${PYTHONPATH:-}"
python "$LMENT/OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py" \
  --checkpoint-input-dir "$SRC" \
  --huggingface-output-dir "$OUT/$NAME" \
  --tokenizer-name-or-path allenai/dolma2-tokenizer \
  --max-sequence-length 2048

echo "wrote $(du -sh "$OUT/$NAME" | cut -f1) to $OUT/$NAME"
