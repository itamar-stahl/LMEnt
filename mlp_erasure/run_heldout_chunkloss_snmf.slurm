#! /bin/sh
#SBATCH --job-name=chunkloss-snmf
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602,n-301,n-102
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# GAP 3, second half: the two SNMF delta=1 models on held-out chunk loss.
#
# Job 896121 covers control and EMBER on the same instrument; this covers the
# two models that are the actual test, so the four numbers are comparable.
# Splitting them is what makes the arithmetic fit: scoring one model is 21,959
# chunks and took ~75 min on a 3090, so four models never fit a 4h wall.
#
# Reference points already measured, all on this same instrument and the same
# held-out set:
#   control model  (saw Rome)        +0.1336 nats/token   job 896121
#   ablated twin   (never saw Rome)  +0.3144 nats/token
# A working erasure has to move into that gap. EMBER is the positive control
# and is scored by 896121 -- if chunk loss cannot see EMBER either, these two
# numbers do not carry an argument and the MC comparison should be withdrawn
# rather than defended.
#
# --exclude=n-102: physically dead GPU0, silent CPU fallback or a hang.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure
HF=/home/dcor/galbarak2/hf-models

CTRL=$HF/lment-1b-control-2e-b131k
BLACKLIST=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix/Untaught/runs/untaught-no-rome-core-1b-2e-b131k_20260904_183200/untaught_blacklist.json
OUTDIR="$RUNS/chunkloss_snmf_${SLURM_JOB_ID}"
SCRATCH="$OUTDIR/scratch"
mkdir -p "$OUTDIR" "$SCRATCH"

# Job-independent, so every model in this comparison is scored against the same
# 5,044 control chunks -- including the two already done by 896121, which wrote
# this file. The cache key pins blacklist/n_control/seed/dataset size and a
# mismatch is refused, so this cannot silently introduce a different control set.
CONTROL_CACHE="$RUNS/rome_control_ids_seed42.json"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export LMENT_ROOT="$ROOT"
. "$ROOT/Untaught/framework/env.sh"
echo "PYTHONPATH=$PYTHONPATH"

# Random-access reads off the share run at 4-10/s and stall this scorer for
# hours; staging is what turned the control sampling from >2h unfinished into
# 8.7 min. Fail-safe: any problem leaves LMENT_DATASET on the share and says so.
. "$ROOT/Untaught/framework/node/stage_dataset.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"
echo "blacklist: $BLACKLIST"
echo "control cache: $CONTROL_CACHE"
df -h /home/dcor | tail -1

score () {
  tag=$1; model=$2
  echo
  echo "================ chunk loss: $tag ================"
  "$PY" "$ROOT/ember_eval/heldout_ppl/heldout_ppl.py" \
    --model "$model" \
    --blacklist "$BLACKLIST" \
    --n-control 5000 \
    --dtype float32 \
    --control-cache "$CONTROL_CACHE" \
    --out "$OUTDIR/heldout_${tag}.json"
  rc=$?
  echo "$tag exit $rc"
  if [ "$rc" -ne 0 ] || [ ! -s "$OUTDIR/heldout_${tag}.json" ]; then
    echo "FATAL: $tag produced no result (exit $rc) -- aborting so this does not" >&2
    echo "       look like a completed run with an empty answer." >&2
    exit 1
  fi
}

# Generated and deleted one at a time: each is 5.1G and /home/dcor is tight.
gen_and_score () {
  tag=$1; snmf_out=$2; lo=$3; hi=$4
  saved="$SCRATCH/model_$tag"
  echo
  echo "---- generating $tag (delta 1/1, layers $lo-$hi both sides) ----"
  "$PY" "$ROOT/snmf.py" erase \
    --model "$CTRL" --out "$snmf_out" \
    --delta-in 1 --delta-out 1 \
    --layers-in "$lo" "$hi" --layers-out "$lo" "$hi" \
    --dtype fp32 --save "$saved"
  echo "erase exit $?"
  if [ -d "$saved" ]; then
    score "$tag" "$saved"
    rm -rf "$saved"
    echo "deleted $saved"
    df -h /home/dcor | tail -1
  else
    echo "NO MODEL SAVED for $tag"
    exit 1
  fi
}

gen_and_score snmf_ratio44_d1 "$RUNS/snmf_rome_883188"     4 6
gen_and_score snmf_judge_d1   "$RUNS/snmf_romeband_871493" 5 6

echo
echo "================ SUMMARY ================"
"$PY" - <<PYEOF
import json, glob, os
for f in sorted(glob.glob("$OUTDIR/heldout_*.json")):
    tag = os.path.basename(f)[len("heldout_"):-len(".json")]
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  {tag}: unreadable ({e})"); continue
    h, c = d.get("heldout", {}), d.get("control", {})
    if h and c:
        print(f"  {tag}: held-out minus control "
              f"{h['mean_loss_micro'] - c['mean_loss_micro']:+.4f} nats/token")
    else:
        print(f"  {tag}: {json.dumps({k: v for k, v in d.items() if not isinstance(v, list)})[:300]}")
PYEOF
echo DONE
