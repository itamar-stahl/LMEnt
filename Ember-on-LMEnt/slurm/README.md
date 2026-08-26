# LMEnt EMBER on Slurm

Run all commands on the login node from the repository checkout. CPU
factorization stays on the login node. Gemma judging, LMEnt erasure/evaluation,
and Alpaca generation run in one `gpu-h100-killable` job.

```sh
cd /home/morg/NLP_2526b/$(whoami)/LMEnt/Ember-on-LMEnt
. ./activate_env.sh

sh slurm/submit_ember.sh \
  --config configs/ember_lment_slurm.yaml \
  --concept "Culture of Greece" \
  --concept-json data/concept_sentences.json \
  --neutral-json data/neutral_sentences.json \
  --output-dir lment_outputs/culture-of-greece \
  --delta 0.5 \
  --judge-model google/gemma-3-12b-it
```

The login node downloads the gated judge model into the shared Hugging Face
cache. Authenticate once before the first run with `hf auth login`, or set
`HF_TOKEN`. The compute job receives the resolved snapshot path and runs
offline. Pass a local shared model directory to `--judge-model` to skip the
download.

Use `--eval-json data/mc_questions.json` instead of `--delta` for automatic
delta selection. Add `--alpaca-eval` for the complete Alpaca split. The H100
batch size is reduced from the profile maximum using free VRAM.

## Cluster test

The default command runs local tests, prepares factors on the login node, then
submits and waits for a real H100 smoke run using Gemma feature judging and one
Alpaca item:

```sh
sh slurm/tests/run_test.sh
```

Use `--no-submit` to check only the environment, Python tests, dependencies,
compilation, and shell syntax. Each run prints the path to `report.log`; full
logs, generated Slurm files, the erased embedding, and EMBER's JSON report stay
under `slurm_test_runs/test_<timestamp>/`.
