#! /bin/sh
#SBATCH --job-name=chunkloss
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602,n-301
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# GAP 3: the instrument. Everything in DELTA_FIX_RERUN_RESULTS.md is 4-option
# MC accuracy at n=50 (+-7pt). This project's own record says chunk loss is
# the sensitive instrument and QA accuracy the blunt one -- the Rome ablation
# showed far more clearly on held-out chunk loss than on raw accuracy.
#
# So: re-measure the same erasures with heldout_ppl.py, which scores per-token
# loss on blacklisted Ancient Rome chunks against a LENGTH-MATCHED control
# sample (length matters because loss varies systematically with position).
#
# Reference point for what a real effect looks like on this instrument: the
# ablated twin (a model that never saw Rome) scores +0.3144 nats/token on
# excluded Rome chunks vs length-matched controls.
#
# Four models, in order:
#   control          the baseline
#   rome-erased      EMBER. POSITIVE CONTROL -- it moved MC by -6pt, so if
#                    chunk loss cannot see EMBER either, the instrument is not
#                    sensitive enough here and the whole comparison is void.
#   snmf ratio44     delta=1, 44 features, layers 4-6 both sides
#   snmf judge       delta=1, 6 features, layers 5-6 both sides
#
# The two SNMF models are generated, scored and DELETED one at a time:
# /home/dcor is at ~895G of its 1024G quota and each is 5.1G.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure
HF=/home/dcor/galbarak2/hf-models

CTRL=$HF/lment-1b-control-2e-b131k
BLACKLIST=$ROOT/Untaught/blacklists/ancient_rome_core.json
OUTDIR="$RUNS/chunkloss_${SLURM_JOB_ID}"
SCRATCH="$OUTDIR/scratch"
mkdir -p "$OUTDIR" "$SCRATCH"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"
echo "blacklist: $BLACKLIST"
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
    --out "$OUTDIR/heldout_${tag}.json"
  echo "$tag exit $?"
}

# --- 1. control -----------------------------------------------------------
score control "$CTRL"

# --- 2. EMBER-erased (positive control) -----------------------------------
score ember_rome_erased "$HF/lment-1b-rome-erased-b131k"

# --- 3/4. SNMF delta=1, generated and deleted one at a time ---------------
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
  fi
}

gen_and_score snmf_ratio44_d1 "$RUNS/snmf_rome_883188"     4 6
gen_and_score snmf_judge_d1   "$RUNS/snmf_romeband_871493" 5 6

echo
echo "================ SUMMARY ================"
"$PY" - <<PYEOF
import json, glob, os
rows = []
for f in sorted(glob.glob("$OUTDIR/heldout_*.json")):
    tag = os.path.basename(f)[len("heldout_"):-len(".json")]
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  {tag}: unreadable ({e})"); continue
    rows.append((tag, d))
    print(f"  {tag}: {json.dumps({k: v for k, v in d.items() if not isinstance(v, list)})[:300]}")
PYEOF
echo DONE
