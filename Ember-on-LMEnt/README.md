# EMBER on LMEnt

This package applies standalone EMBER embedding erasure to a local LMEnt
Hugging Face checkpoint. It does not combine EMBER with another erasure method.

Input: one YAML configuration, one exact concept name, concept sentences, and
neutral sentences. Output: a new run folder containing fitted features,
control/erased evaluation, an erased embedding layer (the default), and a
machine-readable report. The source checkpoint is never modified.

EMBER paper: [Don't Forget Your Embeddings](https://arxiv.org/abs/2606.03695).
The original Gemma/Llama paper pipeline remains available through
`python -m ember.run_erasure`; this README focuses on the LMEnt integration.

Coding agents should follow [`AGENTS.md`](AGENTS.md).

## Human guide

### What the pipeline does

For one concept, `ember.run_lment_ember`:

1. Creates `runs/<concept>_<model-key>_<date_time>/` and snapshots the YAML and
   input JSON files.
2. Fits ID-native embedding features with SNMF on CPU or CUDA, or reuses an
   exactly matching cache.
3. Selects concept features with a local Gemma judge or a numeric threshold.
4. Evaluates the pristine control model, chooses or accepts an erasure strength
   `delta`, applies EMBER, and evaluates the edited model.
5. Saves and reloads the result, verifies that only allowed embedding rows
   changed, and writes `outputs/report.json`.

Sentence sampling follows the original EMBER Wikipedia approach. The LMEnt
chunk-based alternative remains a future sampling seam; it is not part of the
current flow.

### Environment

The supported environment is the suite-level `lment` Conda environment. From
the `LMEnt` suite root:

```sh
# Linux
conda env create -f environment.yml
conda activate lment

# Then enter this package
cd Ember-on-LMEnt
```

On Windows, create the environment from `environment.windows.yml` instead. On
the TAU cluster, use the already-created environment:

```sh
. /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/activate_env.sh
```

`external/snmf` is installed as part of this package. Do not create a second
environment inside `Ember-on-LMEnt`.

### Required data

The concept JSON is a list. The requested `--concept` must exactly match one
record:

```json
[
  {
    "concept": "Culture of Greece",
    "sentences": ["First Wikipedia sentence.", "Second Wikipedia sentence."]
  }
]
```

The neutral JSON is also a list:

```json
[
  {"sentence": "A neutral Wikipedia sentence."}
]
```

Extra metadata such as `url` and `title` is allowed. Working examples are in
`data/concept_sentences.json` and `data/neutral_sentences.json`.

Evaluation is optional only when `ember.explicit_delta` is set. For automatic
delta selection, `eval.data_json` must contain both train and held-out test data
for the concept under `QA_train`, `SimdomQA_train`, `QA_test`, and
`SimdomQA_test`. Each question contains `q`, `correct_answer`, and at least four
`options`. See `data/mc_questions_real_flow.json` for the schema.

### Configure and run one concept

Copy `configs/ember_lment.yaml`, then set at least:

- `model_name`: the flat local LMEnt Hugging Face checkpoint.
- `lment.model_key`: a stable name used in run and cache paths.
- `lment.data.concept_json` and `neutral_json`.
- `lment.model_device`: `auto`, `cpu`, or `cuda`.
- `lment.features.fitting_device`: `cpu` or `cuda`.
- `selection.mode`, delta settings, and evaluation settings.

All settings live in YAML. The public command intentionally accepts only the
config and one concept:

```sh
python -m ember.run_lment_ember \
  --config /absolute/path/to/ember_lment.yaml \
  --concept "Culture of Greece"
```

The final line reports the retained run directory as `[run] <absolute-path>`.

### Feature selection and delta

Judge mode is the production default:

```yaml
selection:
  mode: judge
  ratio_thresh: 2.0
  judge_confidence_threshold: 0.85
  judge_top_k: 20
lment:
  judge:
    model: google/gemma-4-12B-it
    device: cuda
```

The built-in Gemma connector performs the original two-prompt flow: describe a
feature from its activating tokens, then classify whether the description
belongs to the requested concept. Its complete trace is saved with the run.

To skip the judge, a threshold is required. Every feature with
`ratio_abs >= feature_ratio_threshold` is selected:

```yaml
selection:
  mode: threshold
  feature_ratio_threshold: 8.0
lment:
  judge:
    model: null
```

Set `ember.explicit_delta` to use one fixed value. Set it to `null` to evaluate
all values in `ember.deltas` on the train split and choose the best value before
held-out evaluation.

### Feature fitting and reuse

CUDA fitting is the default and CPU remains supported:

```yaml
lment:
  features:
    fitting_device: cuda  # or cpu
    max_iterations: 20000
    reuse: false
```

New features are first written under the run's `outputs/features/`. After a
successful run, they are copied to `mf_outputs/` only when the corresponding
shared cache is empty. With `reuse: true`, all three feature branches and their
provenance must exist, and the stored concept and neutral inputs must exactly
match. A mismatch fails loudly; it is never silently reused.

### Outputs and loading the erased model

Every run is independent:

```text
runs/<concept>_<model-key>_<date_time>/
|-- source_config.yaml
|-- config.yaml
|-- inputs/
|-- run_environment.json
|-- run_wrapper.sh
|-- run_wrapper.ps1
`-- outputs/
    |-- features/
    |-- erased_embeddings.safetensors
    `-- report.json
```

The default artifact contains only the edited input-embedding tensor plus hashes
that bind it to the base config and base embedding. Load it without changing the
base checkpoint:

```python
import torch
from ember.erased_embedding import load_lment_with_erased_embeddings

model, tokenizer = load_lment_with_erased_embeddings(
    "/absolute/path/to/lment-1b-control-2e",
    "/absolute/path/to/run/outputs/erased_embeddings.safetensors",
    device="cuda",
    dtype=torch.float32,
)
```

Set `lment.save.full_model: true` only when a complete copied checkpoint is
required; it is saved under `outputs/model/`.

### Slurm

The production cluster config is `configs/ember_lment_slurm.yaml`. On both login
and GPU nodes, the enforced control checkpoint is:

```text
/home/dcor/galbarak2/hf-models/lment-1b-control-2e/
```

Edit the YAML scheduler fields for your allocation, then submit one concept:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/submit_ember.sh \
  --config /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/configs/ember_lment_slurm.yaml \
  --concept "Culture of Greece"
```

The client creates the run folder, copies all configuration and input files,
resolves the judge model, writes an absolute `job.slurm`, and submits it. The
node writes outputs into that same run folder. GPU selection comes from Slurm's
`#SBATCH --constraint`, not Python.

`slurm/submit_all_concepts.sh` is an optional thin layer that submits one fully
independent job per concept in the concept JSON.

### Tests

Fast tests:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

Bounded real CPU and CUDA flows on Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\run_real_flows.ps1
```

The checked-in Windows real-flow YAMLs contain the development machine's model
path; change `model_name` in both files before running them on another machine.

Equivalent cluster suite—fast tests and a real CPU flow on the login node,
followed by a real CUDA flow on `studentkillable` with `titan_xp`:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/tests/run_test.sh
```

These real flows use rank 2 and two fitting iterations so they test the complete
pipeline without becoming a long scientific experiment. See `tests/README.md`
and `slurm/tests/README.md`.
