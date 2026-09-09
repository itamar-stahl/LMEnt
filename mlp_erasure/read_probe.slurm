#! /bin/sh
#SBATCH --job-name=readprob
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --constraint=a6000
#SBATCH --time=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
# Is the 1B checkpoint readable at a sane rate from the node we get, and does
# a fp32 load onto one GPU complete? Written because jobs 870455/870456 sat on
# "Loading checkpoint shards: 0/2" for 45 minutes on n-602, one of them as the
# only reader, while the same filer served 36.6 MB/s to the login node and the
# same load path took 2.1s on a login-node GPU. This separates filer
# throughput from the Python load without spending an hour to find out.
MODEL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "--- 300 MB direct read of shard 1"
dd if="$MODEL/model-00001-of-00002.safetensors" of=/dev/null bs=1M count=300 iflag=direct 2>&1 | tail -1
echo "--- 300 MB cached read of shard 1"
dd if="$MODEL/model-00001-of-00002.safetensors" of=/dev/null bs=1M count=300 2>&1 | tail -1
echo "--- full fp32 load onto cuda:0, timed"
"$PY" - <<'PYEOF'
import time, torch
from transformers import AutoModelForCausalLM
t = time.time()
m = AutoModelForCausalLM.from_pretrained(
    "/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k",
    device_map={"": "cuda:0"}, dtype=torch.float32)
print("loaded %s in %.1fs" % (next(m.parameters()).device, time.time() - t), flush=True)
PYEOF
echo "--- done"
