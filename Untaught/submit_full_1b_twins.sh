#!/bin/sh
# Submit the full 1B training pair: control and "Harry Potter" ablated.
#
#     ./submit_full_1b_twins.sh
#
# Two independent SLURM jobs, so they train in parallel. Everything else --
# resources, hyperparameters, the ablation -- is in the two configs.
#
# These ask for 4 a100 nodes on a research-group partition, so unlike the 170M
# pair they are not preemptible by design. Run this again to continue after the
# 1-day limit: each new run folder resumes from the newest checkpoint left by
# the previous run of the same job (job.resume_from_previous_run).

set -eu

cd "$(dirname "$0")"

# Once, here: sub_builder.sh sources activate_env.sh too, but the exported
# UNTAUGHT_ENV_READY guard makes those calls return immediately instead of
# re-activating conda and re-probing Elasticsearch for each job.
# shellcheck disable=SC1091
. ./activate_env.sh

echo
echo "=============================================================="
echo "  UNTAUGHT -- submitting the full 1B training pair"
echo "=============================================================="

for config in configs/train_1b_control_full.yaml \
              configs/train_1b_no_harry_potter_full.yaml; do
  echo
  echo "--- ${config} ---"
  sh ./framework/client/sub_builder.sh "${config}"
done

echo
echo "=============================================================="
echo "  Both submitted. Track with:  squeue --me"
echo "  Each run folder is under runs/<job_name>_<date>_<time>/"
echo "  Re-run ./submit_full_1b_twins.sh to continue from the last"
echo "  checkpoint; nothing is lost."
echo "=============================================================="
