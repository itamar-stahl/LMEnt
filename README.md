# LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations
This repository contains the official code for the paper: "LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations" (2025).

---

## Setup

```bash
git clone git@github.com:itamar-stahl/LMEnt.git
cd LMEnt

# Linux
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate lment
```

On native Windows, run from **Miniforge Prompt** in the repository root:

```powershell
$env:PYTHONNOUSERSITE = "1"
conda env create -f environment.windows.yml
conda activate lment
```

Setting `PYTHONNOUSERSITE` before creation also isolates Conda's internal pip
step; the manifests preserve the setting whenever `lment` is activated. Both
environments install the local `Ember-on-LMEnt` package and CUDA 12.8 PyTorch.

There are no git submodules. The third-party projects this work builds on are
vendored into `third_party/`, so a plain `git clone` is complete — see
[THIRD_PARTY.md](THIRD_PARTY.md) for what each one is and how it is licensed.

## Repository layout

LMEnt's own code, in rough pipeline order:

| Directory | What it is |
|---|---|
| [`retrieval-index/`](retrieval-index/) | Builds the Elasticsearch entity-based index over the pretraining corpus. |
| [`aggregate_entity_annotations/`](aggregate_entity_annotations/) | Merges the ReFinED entity links and Maverick coreference output into per-chunk annotations. |
| [`Untaught/`](Untaught/) | The twin-model training framework: trains matched model pairs that differ only in whether a concept was excluded from pretraining. Also the repo's environment layer — `Untaught/framework/env.sh` is what sets `PYTHONPATH` and `OLMO_CORE_SRC`. |
| [`Ember-on-LMEnt/`](Ember-on-LMEnt/) | Embedding-level concept erasure (EMBER) applied to the LMEnt models, plus the concept/question data and the sweep grid. |
| [`mlp_erasure/`](mlp_erasure/) | MLP-level erasure methods (RMU, SNMF) and their cluster drivers. |
| [`ember_eval/`](ember_eval/) | Evaluation of the twins and the erased models: held-out perplexity, NLL/KL, completion scoring, OLMES. |

Supporting files at the root:

| Path | What it is |
|---|---|
| `setup.sh` | Prepares a downloaded `LMEnt-Dataset` for use. Run it once, after the download. |
| `environment.yml`, `environment.windows.yml` | Conda environments for Linux and Windows. |
| `third_party/` | Vendored forks of dolma, OLMo-core, olmes, ReFinED and maverick-coref. See [THIRD_PARTY.md](THIRD_PARTY.md). |
| `PRE_PUBLICATION.md` | Checklist of steps to complete before this repository's visibility changes. |

## Pretraining Dataset
The dataset is available on [Hugging Face](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset).

To set up the dataset and dataloader, please follow these steps:
1.  Download the [LMEnt-Dataset](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset) dataset.
2.  Run `bash setup.sh <absolute path to LMEnt-Dataset directory>`

The final directory structure should look like this:
```text
LMEnt
    > Ember-on-LMEnt
    > README.md
    > Untaught
    > aggregate_entity_annotations
    > ember_eval
    > environment.yml
    > mlp_erasure
    > retrieval-index
    > setup.sh
    > third_party
        > OLMo-core
        > ReFinED
        > dolma
        > maverick-coref
        > olmes
LMEnt-Dataset
    > dataset-cache
        > batch_indices.npy
        > dataset-metadata
        > dataset-common
        > dataset-348b68bc53a9e58ceab6501cae55d803c6b290615d95ac7d98cb0be4a039085d
    > dataset-tokenized
```

This table summarizes the disk space requirements:

| Directory / File Set              | Context                                                                 | Size       |
|----------------------------------|-------------------------------------------------------------------------|------------|
| `dataset-cache/{dataset-common, dataset-348b68bc53a9e58ceab6501cae55d803c6b290615d95ac7d98cb0be4a039085d}`  | Used to build the dataset and dataloader.                               | **589 MB** |
| `dataset-cache/dataset-metadata` | Decompressed per-chunk metadata for fast dataset metadata retrieval.   | **212 GB** |
| `dataset-tokenized`              | Tokenized, concatenated data chunks with per-chunk metadata (`.csv.gz`). | **47.2 GB** |

## LMEnt Models
The models with checkpoints taken every 10K steps are available in the [Hugging Face Collection](https://huggingface.co/collections/dhgottesman/lment).

The following steps demonstrate how to load a specific model checkpoint, e.g., LMEnt-1B-1E checkpointed at training step 10,000:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "dhgottesman/LMEnt-1B-1E"       # model root
sub = "step10000"                          # the checkpoint folder

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=sub, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder=sub
)

inputs = tokenizer("Hello from step10000!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Entity-Based Retrieval Index
LMEnt's entity-based retrieval index is built on top of [Elasticsearch](https://www.elastic.co/elasticsearch).

To build the index from scratch, run the script `retrieval-index/create_es_index.py`. This can take up to a day to build.

You can alternatively download both case-sensitive and case-insensitive pre-built indexes as [elasticsearch_lment.tar.gz](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset/blob/main/elasticsearch_lment.tar.gz).
Please note that this tar file is **127GB**. To register the pre-built files, please follow the following steps:

1. **Download and Unpack**: Unpack the [elasticsearch_lment.tar.gz](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset/blob/main/elasticsearch_lment.tar.gz) file into a directory (e.g., `/path/to/elasticsearch_lment`). The unpacked contents contain the `lment` directory with the necessary files to restore the index.

2. **Configure path.repo**: In `elasticsearch.yml` (sometimes under `elasticsearch-8.13.4/config/elasticsearch.yml`, set the `path.repo` to point to the parent directory of the unpacked contents (e.g., `path.repo: /path/to/elasticsearch_lment`) and restart Elasticsearch (cd elasticsearch-8.13.4; /bin/elasticsearch).

3. **Register the Repository:**

```bash
curl -k -u 'elastic:<PASSWORD>' \
  -H 'Content-Type: application/json' \
  -X PUT 'https://<ELASTIC_SERVER_ADDR>:<PORT>/_snapshot/lment' \
  -d '{
    "type": "fs",
    "settings": {
      "location": "/path/to/elasticsearch_lment/lment"
    }
  }'
```

4. **Restoring the Indices**:

Restore the case-sensitive index.
```bash
curl -k -u 'elastic:<PASSWORD>' \
  -H 'Content-Type: application/json' \
  -X POST 'https://<ELASTIC_SERVER_ADDR>:<PORT>/_snapshot/lment/lment/_restore?wait_for_completion=true' \
  -d '{
    "indices": "enwiki_case_sensitive",
    "rename_pattern": "enwiki_case_sensitive",
    "rename_replacement": "lment_cs",
    "include_global_state": false
  }'
```

Restore the case-insensitive index.
```bash
curl -k -u 'elastic:<PASSWORD>' \
  -H 'Content-Type: application/json' \
  -X POST 'https://<ELASTIC_SERVER_ADDR>:<PORT>/_snapshot/lment/lment/_restore?wait_for_completion=true' \
  -d '{
    "indices": "enwiki",
    "rename_pattern": "enwiki",
    "rename_replacement": "lment_ci",
    "include_global_state": false
  }'
```

> *Note that case in/sensitivity is relevant only if you want to perform string-based retrieval.
We include examples of entity-based retrieval and string-based retrieval queries in the README under `retrieval-index`.*

## Analyzing Learning Dynamics
If you want to analyze learning dynamics on PopQA, you can use our [annotated dataset](https://huggingface.co/datasets/dhgottesman/popqa-kas). It includes precomputed chunk identifiers `chunk_id` from the pretraining corpus that mention the subject entity `subject_chunks`, the answer entity `answer_chunks`, and co-occuring `shared_chunks`.

**For this analysis, you don't need to download the entire pretraining dataset, you only need** `LMEnt-Dataset/dataset-cache/batch_indices.npy`.

To map a given chunk ID to the training step at which a model **trained for one epoch** saw that chunk, you can use the following code:
```python
import numpy as np

batch_indices = np.load(
    "LMEnt-Dataset/dataset-cache/batch_indices.npy",
    allow_pickle=True
)

# Example chunk identifier
chunk_id = 2955255

for step, chunk_ids_in_batch in enumerate(batch_indices):
    if chunk_id in chunk_ids_in_batch:
        print(f"Chunk with chunk_id {chunk_id} was seen in training step {step}")
```

which prints:

```text
Chunk with chunk_id 2955255 was seen in training step 16618
```

## Training LMEnt Models
To kick off training LMEnt models, run `third_party/OLMo-core/src/examples/kas/train.py <path/to/config.json>`. You need to define a `config.json` like `third_party/OLMo-core/src/examples/kas/kas_config.json`.

## Annotating Pretraining Data
### 1. ReFinED
Follow the instructions in this [README](https://github.com/dhgottesman/ReFinED/blob/main/README.md) to process a raw Wikipedia dump, extract hyperlinks, and generate all required files for entity linking.

Run `third_party/ReFinED/run_refined.py` with slurm using `third_party/ReFinED/run.slurm`.

### 2. Maverick
Run `third_party/maverick-coref/run_maverick.py` with slurm using `third_party/maverick-coref/run.slurm`.

### 3. Merging Annotations
Run `aggregate_entity_annotations/run_aggregate_entity_annotations.py` on every file outputted by `third_party/maverick-coref/run_maverick.py`.

## Tokenizing Pretraining Data
Run `third_party/dolma/python/run.slurm`.

## Citing

If you use LMEnt, please cite the paper:

```bibtex
@article{lment2025,
  title  = {LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations},
  year   = {2025},
}
```

## Third-party code and licences

`third_party/` holds vendored forks of five external projects, and
`Ember-on-LMEnt/external/` holds four more. They keep their upstream licences,
which are not all the same — two are NonCommercial. See
[THIRD_PARTY.md](THIRD_PARTY.md) before redistributing any of it.
