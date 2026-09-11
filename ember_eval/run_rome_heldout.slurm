#! /bin/sh
#SBATCH --job-name=rome-heldout
#SBATCH --output=/home/dcor/galbarak2/lment-rome-check/logs/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/lment-rome-check/logs/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Held-out-chunk loss for the ANCIENT ROME twins, at a matched step.
#
# Same three-stage shape as the August Pornography measurement
# (ember_eval/HELDOUT_RESULTS.md), pointed at the LMEnt-initfix checkout and the
# no-rome pair. MODE selects the stage:
#
#   sbatch --export=ALL,MODE=check   ...            prove chunk_id -> Rome text
#   sbatch --export=ALL,MODE=convert,RUN=...,STEP=...,NAME=...   distcp -> HF
#   sbatch --export=ALL,MODE=ppl,MODEL_PATH=...,OUT_TAG=...      score one twin
#
# Do NOT run these on the login node: importing torch + olmo_core off /home/morg
# there took >20 min under NFS contention and never reached the first print.

set -u

export LMENT_ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix
LMENT=$LMENT_ROOT
HP=$LMENT/ember_eval/heldout_ppl
# WORKDIR/BLACKLIST/MARKERS default to the Ancient Rome values this script was
# written for, so an unset environment reproduces the published Rome numbers
# byte for byte. Override all three to point it at another concept rather
# than forking the file -- Baseball uses:
#   WORKDIR=/home/dcor/galbarak2/lment-baseball-check
#   BLACKLIST=$WORKDIR/bb_blacklist_sample3000.json
#   MARKERS=$BASEBALL_MARKERS   (see below)
W="${WORKDIR:-/home/dcor/galbarak2/lment-rome-check}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# The lment env ships a PARTIAL olmo_core in site-packages (no data/), so
# `from olmo_core.data import ...` fails in a way that reads like the package is
# absent. env.sh puts the repo sources ahead of site-packages; LMENT_ROOT is
# exported above so it resolves to LMEnt-initfix, the tree these checkpoints
# were trained from, and not the default LMEnt checkout.
. $LMENT/Untaught/framework/env.sh
export PYTHONPATH="$OLMO_CORE_SRC:${PYTHONPATH:-}"
echo "LMENT_ROOT=$LMENT_ROOT"
echo "PYTHONPATH=$PYTHONPATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "no gpu"
echo "node=$(hostname) mode=${MODE:-ppl}"

ROME_MARKERS='\b(ancient rome|roman\w*|rome|caesar|augustus|cicero|pompey|nero|caligula|hadrian|trajan|constantine|legion\w*|centurion\w*|consul\w*|praetor\w*|plebeian\w*|patrician\w*|gladiator\w*|colosseum|aqueduct\w*|punic|carthag\w*|latium|etruscan\w*|senatus|imperator|denarius|denarii)\b'
BASEBALL_MARKERS='\b(baseball|ballpark|ballclub|major league\w*|mlb|american league|national league|world series|pennant|inning\w*|pitcher\w*|pitching|batter\w*|batting|outfield\w*|infield\w*|shortstop\w*|catcher\w*|home run\w*|homer\w*|rbi|strikeout\w*|strikes out|bullpen|dugout|umpire\w*|yankees|red sox|dodgers|cubs|cardinals|giants|braves|tigers|phillies|mets|indians|guardians|reds|pirates|orioles|white sox|astros|athletics|mariners|rangers|blue jays|twins|royals|brewers|padres|rockies|diamondbacks|marlins|rays|nationals|angels)\b'
# MARKERS picks which of the two the check stage uses.
MARKERS="${MARKERS:-$ROME_MARKERS}"
BLACKLIST="${BLACKLIST:-$W/rome_blacklist_sample3000.json}"

# TOKENIZER: the hub id resolves only with a live HF token, and on 2026-09-11 the
# cached OAuth token under HF_HOME had expired, so convert died with a 401 on
# allenai/dolma2-tokenizer after successfully unsharding. Default to the control
# twin's own tokenizer files instead: they are local, they are the same dolma2
# tokenizer, and using them guarantees a twin is byte-identical to its control on
# that axis rather than merely equivalent. Override to a hub id if ever needed.
TOKENIZER="${TOKENIZER:-/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

case "${MODE:-ppl}" in
  check)
    exec "$PY" "$HP/check_indexing_concept.py" \
      --blacklist "$BLACKLIST" \
      --markers "$MARKERS" --n 12
    ;;

  convert)
    : "${RUN:?set RUN}"; : "${STEP:?set STEP}"; : "${NAME:?set NAME}"
    # The checkpoint directory name encodes model/lr/batch/wd/duration, so it
    # differs between the authors'-config pair (olmo2_1B_0.0003_32768_0.01_2)
    # this was written for and the b131k pair (olmo2_1B_0.0004_131072_0.05_2).
    # Hardcoding it made convert abort with "0 proper shards" on any run at
    # different settings, which reads as a corrupt checkpoint and is not one.
    # Discover it instead, exactly as Untaught/convert_to_hf.sh does.
    CKPT_ROOT="$(ls -d "$UNTAUGHT_RUNS_DIR/$RUN"/checkpoints/*/ 2>/dev/null | head -1)"
    if [ -z "$CKPT_ROOT" ]; then
      echo "FATAL: no checkpoints/*/ under $UNTAUGHT_RUNS_DIR/$RUN" >&2; exit 1
    fi
    SRC="${CKPT_ROOT}${STEP}"
    # A checkpoint interrupted mid-save leaves shards under tmp* names that were
    # never renamed, and nothing downstream notices. Refuse to convert one.
    N=$(ls "$SRC/model_and_optim" 2>/dev/null | grep -c '^__') || N=0
    if [ "$N" -ne 16 ]; then
      echo "FATAL: $SRC has $N proper shards, expected 16" >&2; exit 1
    fi
    if [ ! -f "$SRC/model_and_optim/.metadata" ]; then
      echo "FATAL: $SRC has no .metadata" >&2; exit 1
    fi
    echo "converting $SRC -> $W/hf/$NAME"
    mkdir -p "$W/hf"
    exec "$PY" "$LMENT/OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py" \
      --checkpoint-input-dir "$SRC" \
      --huggingface-output-dir "$W/hf/$NAME" \
      --tokenizer-name-or-path "$TOKENIZER" \
      --max-sequence-length 2048
    ;;

  ppl)
    : "${MODEL_PATH:?set MODEL_PATH}"; : "${OUT_TAG:?set OUT_TAG}"
    exec "$PY" "$HP/heldout_ppl.py" \
      --model "$MODEL_PATH" \
      --blacklist "$BLACKLIST" \
      --n-control "${N_CONTROL:-5000}" \
      --dtype float32 \
      --out "$W/results/${OUT_TAG}_${SLURM_JOB_ID}.json"
    ;;

  *) echo "FATAL: MODE must be check, convert or ppl" >&2; exit 1 ;;
esac
