#! /bin/sh
#SBATCH --job-name=preflight-ens
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602,n-102
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#
# Pure file I/O -- ~20 GB of NFS reads, no GPU, no torch. Runs under sbatch
# rather than in the agent's shell because a background read of this size gets
# reaped for harness memory (it already did once).

PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
cd /home/dcor/galbarak2/lment-qlsim || exit 1
"$PY" ember_eval/ensembles/preflight.py
echo "preflight exit $?"
