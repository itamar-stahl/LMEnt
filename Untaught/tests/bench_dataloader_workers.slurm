#! /bin/sh
# Dataloader worker scaling, on CPUs only -- see bench_dataloader_workers.py.
#
# This needs cores and nothing else: no GPU, no model, no optimizer. That keeps
# it off gpu-h200, where the gpu4 QOS caps the whole partition at 4 GPUs and the
# two live twins hold two of them.
#
# `killable`, not `cpu-killable`: the obvious-looking CPU partition is not in
# this user's associations, and sbatch rejects it with "Invalid account or
# account/partition combination specified". `sacctmgr show assoc where
# user=$(whoami)` is the list that matters -- under gpu-research it is gpu-b200,
# gpu-{b200,h100,h200}-killable, gpu-h200 and killable.
#
# Killable is fine here in a way it is not for training: this job is minutes
# long and writes nothing, so a preemption costs a resubmission rather than a
# checkpoint reload. 64 CPUs is what testing 48 workers needs with headroom for
# the parent; it also picks out the larger idle nodes (n-502, rack-gww-dgx1 and
# rack-omerl-g01 have 64-80 cores, while n-202..205 have 40).
#
#     sbatch tests/bench_dataloader_workers.slurm
#
#SBATCH --job-name=bench-dataloader-workers
#SBATCH --output=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix/Untaught/tests/bench_dataloader_workers.%j.out
#SBATCH --error=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix/Untaught/tests/bench_dataloader_workers.%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120000

set -eu

# activate_env.sh defaults LMENT_ROOT to .../LMEnt and cds there regardless of
# where it is invoked from, so the override is mandatory in this checkout.
LMENT_ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix
export LMENT_ROOT

cd "${LMENT_ROOT}/Untaught"
# shellcheck disable=SC1091
. ./activate_env.sh

echo "=========================================================="
echo " node    : $(hostname)"
echo " cores   : $(nproc)"
echo " started : $(date)"
echo "=========================================================="

python tests/bench_dataloader_workers.py configs/train_1b_control_2e.yaml \
  --workers 4,12,24,48 --warmup 20 --batches 120

echo "=========================================================="
echo " finished : $(date)"
echo "=========================================================="
