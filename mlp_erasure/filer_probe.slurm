#! /bin/sh
#SBATCH --job-name=filerprb
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
# Which filer is fast FROM A COMPUTE NODE? read_probe.slurm measured /home/dcor
# only, and the login node is not representative: c-002 reads dcor at 36.6 MB/s
# while n-602 manages 5.4 direct and 1.5 buffered. Staging a checkpoint on
# /home/morg is only worth doing if morg is actually better from where the jobs
# run, and the 85.6 MB/s figure on record for morg was a login-node number.
# Buffered is the path that matters -- transformers reads buffered.
DCOR=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k/model-00001-of-00002.safetensors
MORG=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix/Untaught/runs/untaught-no-baseball-core-teams-1b-2e-b131k-h100_20260908_120920/checkpoints/olmo2_1B_0.0004_131072_0.05_2/step48000/model_and_optim
echo "host $(hostname)"
for label in dcor morg; do
  case $label in
    dcor) f="$DCOR";;
    morg) f="$(ls -S "$MORG"/*.distcp 2>/dev/null | head -1)";;
  esac
  if [ -z "$f" ] || [ ! -r "$f" ]; then echo "$label: NOT READABLE ($f)"; continue; fi
  sz=$(( $(stat -c %s "$f") / 1048576 ))
  n=200; [ "$sz" -lt 200 ] && n="$sz"
  printf "%-5s direct   : " "$label"
  dd if="$f" of=/dev/null bs=1M count=$n iflag=direct 2>&1 | tail -1
  printf "%-5s buffered : " "$label"
  dd if="$f" of=/dev/null bs=1M count=$n 2>&1 | tail -1
done
echo "--- done"
