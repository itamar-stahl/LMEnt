#! /bin/sh
#SBATCH --job-name=hp-peek
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/fewshot/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/fewshot/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# n-301 and n-303 failed CUDA init on jobs 782202/782435/782791 -- nvidia-smi
# sees the card but torch cannot initialise, so the job silently runs on CPU.
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-301,n-303
export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python \
  /home/morg/NLP_2526b/galbarak2/LMEnt/ember_eval/fewshot_probe/hp_popqa_peek.py
echo "=== peek rc=$?"
