#! /bin/sh
# SHARED variable definitions for the MLP-erasure drivers. Nothing here starts a
# job or changes a shell. Every run_*.slurm in this directory resolves ROOT and
# then sources this file; every value below is an override-able default, so
# `sbatch --export=ALL,RUNS=/somewhere/else ...` wins over anything set here.
#
# It exists because these six lines were copy-pasted into twenty-three drivers,
# each with ROOT *hard-assigned* to whichever checkout its author was sitting in
# (LMEnt-erasure-fix for most, LMEnt-mlp for two). A driver submitted from any
# other tree therefore ran that tree's code and not the one you were editing --
# silently, because both paths existed. ROOT is now resolved from the directory
# you submitted from, so a checkout runs itself.

# ROOT is resolved by the caller before this file is sourced; see the header
# block in any run_*.slurm. Refuse to guess if it is somehow still unset.
: "${ROOT:?ROOT must be resolved before sourcing env.sh}"

: "${PY:=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python}"

# RUNS is overridable because /home/dcor is quota-bound (1 TB hard, read
# `quota -s`, not `df`), and a grid of saved candidates is 4.4 GB per cell.
: "${RUNS:=/home/dcor/galbarak2/runs/mlp_erasure}"

# The concept sentences, MC questions and paragraphs the methods read.
: "${DATA:=$ROOT/Ember-on-LMEnt/data}"

# Never ~: CLIP and the judge together eat more than the 6 GB home quota.
: "${HF_HOME:=/home/dcor/galbarak2/hf_cache}"
: "${TOKENIZERS_PARALLELISM:=false}"

export ROOT PY RUNS DATA HF_HOME TOKENIZERS_PARALLELISM
