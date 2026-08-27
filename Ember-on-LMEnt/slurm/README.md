# LMEnt EMBER on Slurm

The source YAML contains every setting except the concept. The submission client
creates a reproducible run folder and sends the complete concept flow to one GPU
node.

All Slurm login-node and GPU-node flows use the same shared control checkpoint:
`/home/dcor/galbarak2/hf-models/lment-1b-control-2e/`. Submission fails before
`sbatch` if a Slurm YAML names another model or that checkpoint is unavailable.

```sh
cd /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/submit_ember.sh \
  --config /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/configs/ember_lment_slurm.yaml \
  --concept "Culture of Greece"
```

Replace `<USER>` before running. Generated `config.yaml`, `job.slurm`, and both
wrappers contain resolved absolute paths. The client downloads or resolves the
judge model before submission. The node runs offline in the root `lment` Conda
environment.

Every job receives its own folder:

```text
runs/<concept>_<model-key>_<date_time>/
├── source_config.yaml
├── config.yaml
├── inputs/
├── job.slurm
├── run_wrapper.sh
├── run_wrapper.ps1
├── client.log
├── client_report.json
├── log.out
├── log.err
├── run_environment.json
└── outputs/
```

`job.slurm` requests one GPU with `#SBATCH --constraint=h100`. Python checks
that CUDA and the `lment` environment are available, but it does not select or
identify the scheduler's GPU model.

To submit every configured concept as its own independent job:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/submit_all_concepts.sh \
  --config /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/configs/ember_lment_slurm.yaml
```

This optional manager creates no batch run folder. It prints one JSON summary
and exits nonzero if any individual preparation or submission failed.

## Cluster tests

Run the equivalent deployment test package from the login node:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/tests/run_test.sh
```

This one command creates a trackable test package, submits one
`studentkillable` job with `#SBATCH --constraint="titan_xp"`, waits for it, and
returns a nonzero exit code with exact failing phase paths when needed. Before
submission, the login node runs every local unit test, compilation, dependency,
and shell check. The GPU node runs only CUDA checks followed by a bounded
real-model Pornography erasure with pristine and erased evaluation. See
`slurm/tests/README.md` for the saved test-folder layout.
