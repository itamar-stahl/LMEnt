#!/bin/bash
#SBATCH --job-name=ens-build
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
# Phase 2b: a worker that builds ensemble candidates off the shared joblist.
#
# Several of these run at once, on different nodes. Each claims a cell with an
# atomic mkdir before touching it, so two workers never build the same cell and
# a preempted worker's cell is picked up by the next one (its claim is released
# because the model was never verified).
#
# Why a worker pool rather than 43 array tasks: every cell writes a 5 GB
# checkpoint, and several 5 GB saves landing on ONE node at once is exactly what
# produced the silently-holed safetensors in the 2026-09-20 grid. A worker does
# its cells one at a time, so writes are serialised per node, and every save is
# verified with check_alloc.py and retried once before being given up on -- a
# holed file loads fine and reads as aggressive erasure.
#
# Restartable: a cell whose model exists and passes check_alloc is skipped.
#
#   sbatch --export=ALL,ROOT=<worktree>,ENS=<run root>,JOBLIST=<tsv> build.slurm
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"; : "${JOBLIST:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
DATA="$ROOT/Ember-on-LMEnt/data"
CHECK="$ROOT/ember_eval/nll_kl/check_alloc.py"
export HF_HOME=/home/dcor/galbarak2/hf_cache TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp
mkdir -p "$ENS/claims" "$ENS/candidates"

# SWEEP=1: reclaim cells whose claim survived a worker that was preempted or hit
# its walltime mid-build. The claim marks "someone is on it"; a claim with no
# verified model underneath it means nobody is, and without this the cell is
# never rebuilt and the grid comes up silently short.
if [ "${SWEEP:-0}" = 1 ]; then
  reclaimed=0
  for c in "$ENS"/claims/*; do
    [ -d "$c" ] || continue
    lbl=$(basename "$c")
    if [ ! -d "$ENS/candidates/$lbl/model" ] || ! "$PY" "$CHECK" "$ENS/candidates/$lbl/model" >/dev/null 2>&1; then
      rmdir "$c" 2>/dev/null && reclaimed=$((reclaimed+1))
    fi
  done
  echo "sweep: reclaimed $reclaimed orphaned claim(s)"
fi

# Self-heal: a worker will not start without the prep its cells need, so it runs
# any missing prep itself rather than failing the whole grid. Normally dead code
# -- prep.slurm is idempotent and skips when its outputs exist. It matters when a
# prep job dies: the dependency is afterany, so the workers still start, and the
# first one here rebuilds what is missing instead of 43 cells failing one by one.
# One worker per concept via the same atomic claim; the others wait for the fit.
while IFS=$'\t' read -r kind label topic concept base p1 p2 p3; do
  [ -z "${topic:-}" ] && continue
  echo "$topic|$concept|$base"
done < "$JOBLIST" | sort -u | while IFS='|' read -r slug concept base; do
  [ -f "$ENS/probes/$slug/probe.json" ] && [ -f "$ENS/fits/v2_$slug/selected.json" ] && continue
  if mkdir "$ENS/claims/_prep_$slug" 2>/dev/null; then
    echo "self-heal: prep missing for $slug, running it here"
    SLUG="$slug" CONCEPT="$concept" BASE="$base" bash "$ROOT/ember_eval/ensembles/slurm/prep.slurm"
    echo "self-heal: prep $slug exit $?"
  else
    echo "self-heal: another worker is preparing $slug, waiting up to 40 min"
    i=0
    while [ $i -lt 80 ] && [ ! -f "$ENS/fits/v2_$slug/selected.json" ]; do sleep 30; i=$((i+1)); done
  fi
done

echo "worker $SLURM_JOB_ID on $(hostname) at $(date +%F_%H:%M:%S)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
built=0; skipped=0; failed=0

build_one () {   # kind label topic concept base p1 p2 p3 -> writes $OUT/model
  local kind=$1 label=$2 topic=$3 concept=$4 base=$5 p1=$6 p2=$7 p3=$8
  local out="$ENS/candidates/$label"
  case "$kind" in
    rmu)
      local probe="$ENS/probes/$topic/probe.json"
      [ -f "$probe" ] || { echo "  no probe for $topic" >&2; return 1; }
      # band picks the steering scale off THIS model's probe: mid = the mean
      # residual norm, hi = steering_high (10x). Never the control's value.
      local steer
      steer=$("$PY" -c "
import json,sys
d=json.load(open('$probe'))['layers']['$p1']
print(d['mean_residual_norm'] if '$p2'=='mid' else d['steering_high'])
") || return 1
      [ -n "$steer" ] || { echo "  no steering for L$p1 $p2" >&2; return 1; }
      local lids="$((p1-2)) $((p1-1)) $p1"
      echo "  rmu L$p1 $p2 alpha=$p3 steering=$steer layer_ids=$lids"
      "$PY" -u "$ROOT/rmu.py" \
        --model "$base" --concept "$concept" \
        --concept-sentences "$DATA/concept_sentences.json" \
        --neutral-sentences "$DATA/neutral_sentences.json" \
        --out "$out" --dtype fp32 \
        --layer-id "$p1" --layer-ids $lids \
        --lr 1e-4 --alpha "$p3" --steering "$steer" \
        --batch-size 1 --min-len 50 --max-len 2000 \
        --max-num-batches 150 --max-length 512 --seed 42 \
        --sanity --min-cosine 0.3 --min-step-fraction 0.5 --save-model
      # exit 1 here just means a sanity gate failed; the model is still written
      # and the gate is reported rather than hidden. Judge on sanity.json.
      [ -d "$out/model" ]
      ;;
    snmfv1|snmfv2)
      local fit="$p1"
      [ "$kind" = snmfv2 ] && fit="$ENS/fits/v2_$topic"
      [ -f "$fit/selected.json" ] || { echo "  no selection in $fit" >&2; return 1; }
      echo "  $kind delta_in=$p2 delta_out=$p3 fit=$fit"
      "$PY" -u "$ROOT/snmf.py" erase \
        --model "$base" --out "$fit" --dtype fp32 \
        --delta-in "$p2" --delta-out "$p3" \
        --range-cell 0 --layers-in 4 6 --layers-out 4 6 \
        --save "$out/model"
      ;;
    *) echo "  unknown kind $kind" >&2; return 1 ;;
  esac
}

while IFS=$'\t' read -r kind label topic concept base p1 p2 p3; do
  [ -z "${kind:-}" ] && continue
  out="$ENS/candidates/$label"
  if [ -d "$out/model" ] && "$PY" "$CHECK" "$out/model" >/dev/null 2>&1; then
    skipped=$((skipped+1)); continue
  fi
  mkdir "$ENS/claims/$label" 2>/dev/null || { skipped=$((skipped+1)); continue; }
  echo "=== $label  $(date +%H:%M:%S)"
  ok=0
  for attempt in 1 2; do
    rm -rf "$out/model"
    build_one "$kind" "$label" "$topic" "$concept" "$base" "$p1" "$p2" "$p3" || true
    sync
    if [ -d "$out/model" ] && "$PY" "$CHECK" "$out/model"; then
      echo "  verified (attempt $attempt)"; ok=1; break
    fi
    echo "  HOLED or missing on attempt $attempt" >&2
  done
  if [ "$ok" = 1 ]; then built=$((built+1)); else
    failed=$((failed+1)); rmdir "$ENS/claims/$label" 2>/dev/null || true
  fi
done < "$JOBLIST"

echo "=== worker $SLURM_JOB_ID: built $built, skipped $skipped, failed $failed"
exit $((failed > 0 ? 1 : 0))
