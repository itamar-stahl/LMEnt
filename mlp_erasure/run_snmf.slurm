#! /bin/sh
#SBATCH --job-name=snmf-erase
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
# No --constraint. EMBER's erasure of this concept pinned a6000
# (configs/ember_lment_rome_slurm.yaml) and its stated reason was staying out
# of the h100 pool the twin training needs -- which every other card in
# killable does equally. Dropping it is Gal's call, made 2026-09-09 after four
# jobs pinned to a6000 all landed on n-602 and none finished loading the
# model: job 870565 measured that node reading /home/dcor at 5.4 MB/s direct
# and 1.5 MB/s buffered, against 36.6 MB/s from the login node. transformers
# loads buffered, so one 4.5 GB shard there is ~50 minutes. Nothing was wrong
# with the card, the job, or the code.
#
# n-202..n-205 are excluded here for a different reason than in run_rmu.slurm.
# Factorize really does fit on an 11 GB 2080 -- it holds one model copy and a
# 0.38 GB activation matrix -- but MODE=select loads the 23.9 GB Gemma judge in
# a second interpreter, and 11 GB cannot hold it at any batch size. Since the
# directive is static and cannot depend on MODE, the smaller cards are dropped
# for every mode; the cost is a little queue breadth on factorize, and the
# alternative is select landing on a card where it can only fail. Job 871285
# (judge smoke) got a 24 GB RTX 3090 and passed both stages, but only by luck
# of the draw -- nothing had stopped it from landing on n-202.
#
# n-602 is excluded on measurement, not capacity: on 2026-09-09 it served both
# filers at 2.8-2.9 MB/s buffered against n-301's 25.4, so a model read there
# costs 9x what it costs elsewhere. See mlp_erasure/filer_probe.slurm.
#
# See run_rmu.slurm for --exclude. Two notes specific to this script:
#
# --mem=32000 is generous, not tight. The Ancient Rome corpus is 300+300
# sentences averaging 28 tokens, so at max-length 256 the whole activation
# matrix is 5,632 x 16,934 = 0.38 GB in fp32, and semi_nmf holds three tensors
# of that size. Only --layer-batch-size grows the host side. (Note that --mem
# is not enforced on this cluster anyway.)
#
# --time=360, and MODE=select can need most of it: the judge is one unsharded
# 23.9 GB safetensors read off NFS at ~28 MB/s, i.e. 15 min before it answers
# anything, and longer on a contended node. snmf.py's default startup timeout
# is 10800s for that reason.
#
#   MODE=factorize   fit Semi-NMF per layer, write features.pkl + rho_stats.json
#   MODE=select      two-stage judge over the candidates (needs factorize)
#   MODE=select-ratio  --skip-llm: keep every rho > tau candidate, no judge
#   MODE=erase       project the selected directions out, save the model
#   MODE=verify      feature activation before/after, no model saved
#   MODE=sweep       score every delta x layer-range cell on the real MC
#                    questions and write sweep_delta.csv (needs select)
#
#   sbatch --export=ALL,MODE=factorize,OUT_TAG=rome,LAYERS='4 9 14' run_snmf.slurm
#   sbatch --export=ALL,MODE=select,OUT_TAG=rome,SNMF_OUT=<dir> run_snmf.slurm

# ROOT is overridable: it was a hard assignment, so submitting from any other
# checkout silently ran the code in LMEnt-mlp instead of the tree you were in.
: "${ROOT:=/home/morg/NLP_2526b/galbarak2/LMEnt-mlp}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
: "${RUNS:=/home/dcor/galbarak2/runs/mlp_erasure}"   # overridable, see run_rmu.slurm

: "${MODE:=factorize}"
: "${OUT_TAG:?set OUT_TAG}"
: "${MODEL_PATH:=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k}"
: "${CONCEPT:=Ancient Rome}"
: "${N_SENTENCES:=300}"
: "${K:=100}"
: "${TAU:=2.0}"
: "${MAX_ITER:=20000}"
: "${BATCH_SIZE:=8}"
: "${MAX_LENGTH:=256}"
: "${DTYPE:=fp32}"
# 1, not 4. ablate_layer applies (I - delta*P) onto a UNIT direction, so the
# targeted component scales by |1 - delta|: 1 removes it exactly, 2 only flips
# its sign, and 4 MULTIPLIES IT BY THREE. Submitting MODE=erase with the old
# default of 4 is what produced an "erased" checkpoint that scored better on
# the concept than its control. snmf.py now refuses delta >= 2 unless
# --allow-amplification is passed; see MLP_ERASURE.md, "The delta rule was
# wrong". Export DELTA_IN/DELTA_OUT to sweep deliberately.
: "${GAMMA:=0.95}"            # coverage threshold, snmf.py default
: "${DELTA_IN:=1}"
: "${DELTA_OUT:=1}"
: "${LAYERS:=}"
# factorize creates the run directory; every later stage must be pointed at it.
: "${SNMF_OUT:=$RUNS/snmf_${OUT_TAG}_${SLURM_JOB_ID}}"

DATA="$ROOT/Ember-on-LMEnt/data"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# Fail fast rather than spend 15+ minutes reading 23.9 GB onto a card that
# cannot hold it. The --exclude above already drops the 11 GB nodes, but an
# exclude list goes stale as the cluster changes and this does not.
case "$MODE" in
  select|judge-smoke)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ -n "$VRAM" ] && [ "$VRAM" -lt 20000 ]; then
      echo "FATAL: MODE=$MODE loads the 23.9 GB Gemma judge, and $(hostname)" >&2
      echo "       offers only ${VRAM} MiB of VRAM. Resubmit elsewhere." >&2
      exit 1
    fi
    ;;
esac

: "${TOP_TOKENS:=25}"          # activating examples kept per feature (upstream 25)
: "${CONTEXT_WINDOW:=15}"      # tokens either side, clipped at the sentence
: "${MAX_PER_TYPE:=3}"         # cap on repeats of one token string per feature
: "${JUDGE_TOP_TOKENS:=10}"    # examples shown to STAGE1 (upstream --top-m 10)
: "${RTOL:=1e-6}"              # relative convergence; the old absolute tol was
                               # meaningless against a 1e12 reconstruction error
: "${RANGE_CELL:=0}"           # which published layer-range cell. On 18 layers
                               # cell 2 gives out=[9,17]; cell 0 gives [0,5] and
                               # will refuse a one-sided erasure if the selected
                               # features sit deeper than that.

if [ ! -d "$MODEL_PATH" ] || [ ! -r "$MODEL_PATH/config.json" ]; then
  echo "FATAL: $MODEL_PATH is not readable from $(hostname)." >&2
  exit 1
fi

echo "snmf MODE=$MODE concept='$CONCEPT' out=$SNMF_OUT"

case "$MODE" in
factorize)
  # $LAYERS is deliberately unquoted: it is a list of layer indices, or empty
  # for every layer.
  # shellcheck disable=SC2086
  exec "$PY" "$ROOT/snmf.py" factorize \
    --model "$MODEL_PATH" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --out "$SNMF_OUT" --dtype "$DTYPE" --k "$K" --tau "$TAU" \
    --n-sentences "$N_SENTENCES" --max-iter "$MAX_ITER" \
    --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH" \
    --top-tokens "$TOP_TOKENS" --context-window "$CONTEXT_WINDOW" \
    --max-per-token-type "$MAX_PER_TYPE" --rtol "$RTOL" \
    ${LAYERS:+--layers $LAYERS}
  ;;
select)
  exec "$PY" "$ROOT/snmf.py" select \
    --out "$SNMF_OUT" --concept "$CONCEPT" --judge gemma \
    --judge-top-tokens "$JUDGE_TOP_TOKENS" \
    --fork-root "$ROOT/Ember-on-LMEnt"
  ;;
select-ratio)
  exec "$PY" "$ROOT/snmf.py" select --out "$SNMF_OUT" --skip-llm
  ;;
erase)
  : "${SAVE_TO:?set SAVE_TO}"
  # LAYERS_IN / LAYERS_OUT ("lo hi", inclusive) override RANGE_CELL. Cell 0's
  # output range on 18 layers is [0,5] and silently skips features at layer 6;
  # pass "4 6" on both sides to cover every layer the caches hold features at.
  # shellcheck disable=SC2086
  exec "$PY" "$ROOT/snmf.py" erase \
    --model "$MODEL_PATH" --out "$SNMF_OUT" --dtype "$DTYPE" \
    --delta-in "$DELTA_IN" --delta-out "$DELTA_OUT" \
    --range-cell "$RANGE_CELL" \
    ${LAYERS_IN:+--layers-in $LAYERS_IN} ${LAYERS_OUT:+--layers-out $LAYERS_OUT} \
    --save "$SAVE_TO"
  ;;
judge-smoke)
  # Loads the 23.9 GB judge and sends the two real prompts. Nothing else in
  # `select` needs a GPU-scale download, so it is worth proving on its own
  # before spending a factorize on it.
  #
  # Do not co-schedule it with model-loading jobs. The filer serves ~37 MB/s
  # to an idle client (measured 2026-09-09 from the login node), and it is one
  # sequential read with no parallelism to exploit, so it starves everything
  # sharing the node: two factorize jobs that needed 3 minutes of reads each
  # sat on "Loading checkpoint shards 0/2" for over 12 minutes beside it.
  exec "$PY" "$ROOT/mlp_erasure/tests/judge_gpu_smoke.py"
  ;;
verify)
  exec "$PY" "$ROOT/snmf.py" verify \
    --model "$MODEL_PATH" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --out "$SNMF_OUT" --dtype "$DTYPE" \
    --delta-in "$DELTA_IN" --delta-out "$DELTA_OUT" \
    --range-cell "$RANGE_CELL" \
    --n-sentences "$N_SENTENCES" \
    --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH"
  ;;
sweep)
  # The step the pipeline never had. Appendix C.3 tunes delta over {1,4,7,10}
  # crossed with three layer ranges per side and picks a cell on downstream
  # evaluation; `erase` applies ONE cell and saves it. This scores every cell
  # on the real questions and writes a CSV so the cell is chosen on evidence.
  #
  # Loads the model once and rolls the weights back between cells, so it costs
  # one checkpoint read rather than one per cell.
  exec "$PY" "$ROOT/sweep_delta.py" \
    --model "$MODEL_PATH" --snmf-out "$SNMF_OUT" \
    --concept "$CONCEPT" \
    --questions "$DATA/mc_questions.json" \
    --dtype "$DTYPE" --gamma "$GAMMA" \
    --out "$SNMF_OUT/sweep_delta.csv"
  ;;
*)
  echo "FATAL: unknown MODE=$MODE" >&2
  exit 1
  ;;
esac
