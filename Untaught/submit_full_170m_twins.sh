#!/bin/sh
# Submit the full training pair: control and "Harry Potter" ablated.
#
#     ./submit_full_170m_twins.sh
#
# Two independent SLURM jobs, so they train in parallel. Everything else --
# resources, hyperparameters, the ablation -- is in the two configs.
#
# studentkillable caps a job at 1 day and is preemptible, while one epoch is
# ~109K steps, so a full run takes several submissions. Run this again to
# continue: each new run folder resumes from the newest checkpoint left by the
# previous run of the same job (job.resume_from_previous_run).

set -eu

cd "$(dirname "$0")"

# Once, here: sub_builder.sh sources activate_env.sh too, but the exported
# UNTAUGHT_ENV_READY guard makes those calls return immediately instead of
# re-activating conda and re-probing Elasticsearch for each job.
# shellcheck disable=SC1091
. ./activate_env.sh

echo
echo "=============================================================="
echo "  UNTAUGHT -- submitting the full training pair"
echo "=============================================================="

for config in configs/train_170m_control_full.yaml \
              configs/train_170m_no_harry_potter_full.yaml; do
  echo
  echo "--- ${config} ---"
  sh ./framework/client/sub_builder.sh "${config}"
done

echo
echo "=============================================================="
echo "  Both submitted. Track with:  squeue --me"
echo "  Each run folder is under runs/<job_name>_<date>_<time>/"
echo "  Re-run ./submit_full_170m_twins.sh after a preemption to continue from the last"
echo "  checkpoint; nothing is lost."
echo "=============================================================="
