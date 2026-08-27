# LMEnt EMBER on Slurm

The source YAML contains every setting except the concept. The submission client
creates a reproducible run folder and sends the complete concept flow to one GPU
node.

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

## Cluster tests

Run the equivalent deployment test package from the login node:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/tests/run_test.sh
```

The script first checks the environment, compilation, unit tests, generated run
files, absolute paths, and the H100 constraint. Its submitted test uses the same
single-concept flow as production.
