# LMEnt-1B twin, TWO EPOCHS: ABLATED — Pornography (`Q291`) held out

The ablated half of the **2-epoch** twin pair. Same corpus and same ablation as
the 1-epoch pair, trained for two full passes instead of one.

> **Its control is still training.** See "What you can and cannot do with this
> yet" below before you plan anything around it.

    /home/dcor/galbarak2/hf-models/lment-1b-control-2e   <- NOT READY YET
    /home/dcor/galbarak2/hf-models/lment-1b-noporn       <- the 1-epoch ablated twin
    /home/dcor/galbarak2/hf-models/lment-1b-control      <- the 1-epoch control

## What was removed

| | |
|---|---|
| entity | `Q291` (Pornography) |
| chunks excluded | **2,546** — 0.0242% of the 10,491,928-chunk corpus |
| thresholds | hyperlinks 1.0, entity-linking 0.6, coref 0.6, coref-cluster 0.6 |

Chunks were masked from the **loss**, not removed from the data: they still
occupy batch slots, so batch composition, data order and step count are identical
to the control's. The twins differ by exactly the masked gradient contributions.

## Proof the ablation fired, across both training windows

This run took two SLURM windows (it hit the 24h limit at step ~33,000 and was
resubmitted automatically). Both windows loaded the blacklist:

- `loaded 2546 chunk ids` — in **both** run folders;
- **4,640 instance-slots excluded** from the loss in total: 2,627 in window 1 plus
  2,013 in window 2;
- **zero guard leaks** — no blacklisted chunk ever reached the loss through the
  all-masked-batch guard.

Two traps for whoever audits this next. The excluded counter **resets per
window**, so the framework's own closing line (`run finished; 2013 instance-slots
were excluded`) reports only the last window — the total is the sum across run
folders. And the counter prints with a thousands separator, so a grep for
`[0-9.]+` truncates `1,074` to `1`.

The rate is consistent with the 1-epoch run: 4,640 over 54,832 steps is
0.085/step against that run's 0.090/step. Two epochs, twice the exclusions.

## Training

| | |
|---|---|
| duration | **2 epochs** = 54,832 steps (the 1-epoch pair ran 27,416) |
| final CE loss | 2.495 → **perplexity 12.12** (1-epoch ablated twin: 13.51) |
| job | `768495`, `COMPLETED 0:0`, 17h43m in its second window |
| hardware | 1x H100 80GB throughout, `torch 2.6.0+cu124` |

Everything else matches the 1-epoch recipe exactly: `olmo2_1B` (18 layers,
d_model 2048, 16 heads), 3.6B-token LMEnt Wikipedia corpus, global batch 131,072
tokens, rank microbatch 16,384, AdamW peak LR 4e-4, warmup 2,000, cosine to 4e-5,
VSL `grow_p2` over 8 cycles, seed 12536. The configs differ from the 1-epoch ones
by two lines: the run name and the duration.

Note on weight decay: `optim_weight_decay: 0.05` reaches **only the embedding
matrix** — the paper's own `train.py` passes it as a group override and never sets
the top-level field, so every other parameter uses AdamW's default of 0.01. Every
LMEnt model went through this path, so it is consistent, but the name misleads.

## What you can and cannot do with this yet

**You can**, right now, treat it as a finished model: load it, generate, probe
representations, collect activations, and compare it against the **1-epoch
ablated twin** (`lment-1b-noporn`) to study what a second epoch does to a model
that never saw the concept.

**You cannot** yet measure anything about the *ablation's effect*. That requires
its own control, which is still training and is expected to finish the same day
this card was written. Specifically:

- **Do not** use `lment-1b-control` (the 1-epoch control) as this model's
  baseline. Different training duration means any difference you measure is
  dominated by that, not by the ablation.
- **Do not** use the authors' released `dhgottesman/LMEnt-1B-*` as a baseline.
  Different batch size, different optimizer trajectory, different code version.

The ablation is defined *relative to its own control*. That is the whole design.

## Loading

The directory is **flat** — no `stepNNNN` subfolder, unlike the authors' HF repos:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/home/dcor/galbarak2/hf-models/lment-1b-noporn-2e"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto")
```

## Caveats that will cost you a day if you skip them

**It is a base model.** No instruction tuning. It continues text; it does not
answer questions. Prompting for a bare "A/B/C/D" and parsing the generation gives
garbage — EMBER's own multiple-choice protocol scores it near zero, and scoring
letter log-likelihoods makes it answer "A" on 100% of questions, which is position
bias rather than knowledge. Score the **option text** as a continuation instead;
`ember_eval/score_ember_mc.py` in the repo does this.

**Exclusion is not erasure.** The blacklist catches chunks where the entity
annotations flagged the QID above threshold. A chunk discussing the subject
without ever linking it stays in training, and the paper's own error analysis
(appendix B.3) found ~2.7% annotation errors. Say "sharply reduced exposure",
never "never saw it".

## Provenance

Converted from the OLMo-core checkpoint at

    .../runs/untaught-no-porn-1b-2e_20260820_004823/checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832

verified as 16 complete `.distcp` shards with no `tmp*` leftovers before
conversion. The run began in `untaught-no-porn-1b-2e_20260818_183858` and
continued in the folder above after an automatic resubmission; both are kept.

Full documentation:

    git@github.com:itamar-stahl/LMEnt.git   branch itamars/main
    Untaught/STATUS.md       <- start here
    Untaught/RESULTS.md      <- what was trained, and proof the ablation fired
    ember_eval/EVALUATION.md <- what the 1-epoch evaluation found, and what it did not

Contact: Gal Barak <galll.barak@gmail.com>.
