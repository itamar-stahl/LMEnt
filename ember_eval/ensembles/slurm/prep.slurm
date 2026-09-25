#!/bin/bash
#SBATCH --job-name=ens-prep
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=n-102,n-202,n-203,n-204,n-205,n-602,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Phase 2a, one concept: everything the per-cell builds need but must not each
# redo. Two things, both on the EMBER-erased base rather than the control:
#
#   1. the RMU steering probe. EMBER changed the embeddings, so the activation
#      scale RMU's forget loss targets may differ from the control's. The probe
#      is what makes `mid`/`hi` mean something on THIS model; the control's
#      values are never transferred.
#   2. the Variant-2 SNMF factorization + ratio selection, re-derived from this
#      model's own activations. Variant 1 needs nothing here -- it reuses the
#      frozen control-derived fits.
#
# Idempotent: both steps skip if their output already exists, so a preempted job
# is resubmitted as-is.
#
#   sbatch --export=ALL,ROOT=<worktree>,ENS=<run root>,SLUG=rome,CONCEPT="Ancient Rome",BASE=<dir> prep.slurm
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"; : "${SLUG:?}"; : "${CONCEPT:?}"; : "${BASE:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
DATA="$ROOT/Ember-on-LMEnt/data"
export HF_HOME=/home/dcor/galbarak2/hf_cache TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

echo "prep $SLUG on $(hostname) at $(date +%F_%H:%M:%S)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
rc=0

PROBE="$ENS/probes/$SLUG"
if [ -f "$PROBE/probe.json" ]; then
  echo "probe: already present, skipping"
else
  echo "=== RMU steering probe on the EMBER-erased model"
  "$PY" -u "$ROOT/rmu.py" --model "$BASE" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --dtype fp32 --out "$PROBE" --probe || rc=1
  cat "$PROBE/probe.json" 2>/dev/null
fi

FIT="$ENS/fits/v2_$SLUG"
if [ -f "$FIT/selected.json" ]; then
  echo "v2 fit: already selected, skipping"
else
  echo "=== SNMF factorize on the EMBER-erased model (Variant 2)"
  "$PY" -u "$ROOT/snmf.py" factorize --model "$BASE" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --out "$FIT" --layers 4 5 6 --dtype fp32 || rc=1
  echo "=== SNMF select (ratio only, --skip-llm: the Gemma judge is not reproducible here)"
  "$PY" -u "$ROOT/snmf.py" select --out "$FIT" --concept "$CONCEPT" --skip-llm || rc=1
fi

echo "prep $SLUG done at $(date +%F_%H:%M:%S), rc=$rc"
exit $rc
