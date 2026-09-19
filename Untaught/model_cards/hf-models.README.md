# LMEnt 1B models — single-entity pretraining ablations, and erasures of them

OLMo2 1B models trained on the LMEnt Wikipedia corpus. Two kinds of thing live
here and they must not be confused:

- **twins** — models that *never saw* a concept in the loss, produced by
  pretraining. A twin is a fact about training.
- **erased models** — the control with `model.embed_tokens.weight` overwritten
  after the fact. An erasure is an edit to finished weights.

The whole point of the project is to ask whether the second can reach the first.
The answer is **subject-dependent** — see `ember_eval/AI_RESULTS.md`.

> **Every erased directory's card claimed to be the control until 2026-09-19.**
> `materialize_erased_model.py` copied `README.md` out of the base directory
> along with the tokenizer. Nine cards were wrong; all are now corrected and the
> script no longer copies the card. If you have a local copy of any
> `*-erased-*` directory taken before that date, its `README.md` is the
> control's and describes the wrong model.

## The current twins — batch 131,072, 2 epochs, 54,832 steps

One control serves all three arms. Each ablation masks an already-composed
batch, so every twin traverses identical batches, in the same order, for the
same number of steps; the twins differ by exactly the masked gradient
contributions.

| directory | role | held out | state |
|---|---|---|---|
| `lment-1b-control-2e-b131k` | **the control** (job 853707) | nothing | ready |
| `lment-1b-norome-2e-b131k` | ablated (job 850249) | Ancient Rome — 56 QIDs, 65,844 chunks, 0.628% | ready |
| `lment-1b-nobaseball-2e-b131k` | ablated (job 873471) | Baseball — 43 QIDs, 80,466 chunks, 0.766% | ready |
| `lment-1b-noai-2e-b131k` | ablated (job 905819) | Artificial Intelligence — 31 QIDs, 19,818 chunks, 0.189% | ready |

    M_base      = lment-1b-control-2e-b131k
    M_never(C)  = the ablated twin for concept C

**Compare a twin only against this control**, never against a twin for a
different concept, and never against the authors' released
`dhgottesman/LMEnt-1B-*` (different batch size, optimizer trajectory and code
version — though note her 2E and this control are *token-matched* at ~7.19B
tokens, which makes her checkpoint a legitimate replication target, just not a
control).

Embedding health agrees to within 0.01% across all four models (row-norm ratio
2.9374 / 2.9361 / 2.936), which is the cleanest available check that nothing but
the ablation separates them.

**Hardware asymmetry differs by arm, and it belongs next to any weight-space
claim.** Ancient Rome is the only pair where control and twin saw the same card
for every step (both `n-h200` throughout). The Baseball twin ran H100 for four
of five windows; the AI twin spent ~12 hours on an A6000 across eight legs.
Measured cross-GPU drift here is 3e-5 and flips nothing in evaluation.

### What each arm showed

| arm | QA effect | held-out chunk loss | does EMBER erasure reach it? |
|---|---|---|---|
| Ancient Rome | −16 points, both halves independently; z −7.12 against the cross-concept null | +0.2384, dz 1.05 | **no — overshoots**, residual dz −0.315, CI excludes 0 |
| Baseball | the raw +22 was a **prior artifact** and did not survive correction | largest of the three by mean | −12 points QA_test, zero collateral |
| Artificial Intelligence | real and prior-robust; the +12 Simdom "gain" is a prior artifact | +0.1561, **dz 1.83** — the most consistent | **yes** — residual dz ≈ 0 |

Never read a QA-minus-Simdom gap without scoring `logp_null` alone first; it has
turned two headline results into artifacts so far.

## The retired Pornography pair — same recipe, superseded subject

| directory | role | held out |
|---|---|---|
| `lment-1b-control-2e` | control (job 765091), ppl 12.110 | nothing |
| `lment-1b-noporn-2e` | ablated, ppl 12.122 | Pornography `Q291` — 2,546 chunks, 0.0242% |
| `lment-1b-control-2e-step45000` | the control's last pure-H100 checkpoint | — |
| `lment-1b-noporn-2e-step45000` | its twin at the same step | — |

Superseded because the subject is **26x smaller** than Ancient Rome and the
effect sat at the edge of what the instrument could resolve: the 1-epoch pair
showed a concept-specific effect (QA rank 1/18, p=0.006) that the 2-epoch pair
did **not** reproduce. Two claims made on `acc_raw` are retracted in
`ember_eval/EVALUATION.md`. The `step45000` pair exists because that control
moved to an H200 at step 49,500, so step45000 is the last step where both twins
are pure-H100 — see `Untaught/COMPARABILITY.md`.

The 1-epoch pair is **gone**: `/vol/scratch` was purged without warning on
2026-08-23 and took its checkpoints. Gal retired it rather than retrain. Its
per-question evaluation records survive, so published analyses still reproduce.

## Erased models — edited weights, not twins

All are `lment-1b-control-2e-b131k` with the input embedding overwritten;
`lm_head.weight` is a separate, untied tensor and is always carried across
untouched.

| directory | concept | delta | weights |
|---|---|---|---|
| `lment-1b-rome-erased-b131k` | Ancient Rome | **200** (grid argmax) | present |
| `lment-1b-rome-erased-d{2,5,10,50,500,1000}-b131k` | Ancient Rome | the rest of the sweep | **stripped** |
| `lment-1b-baseball-erased-b131k` | Baseball | 10 | present |
| `lment-1b-baseball-erased-d200-b131k` | Baseball | 200 | **stripped** |
| `lment-1b-ai-erased-b131k` | Artificial Intelligence | 5.0 | present |
| `lment-1b-ai-snmf-b131k` | Artificial Intelligence (SNMF, not EMBER) | — | **stripped** |

**Erasure damage on these models is lexical.** A chunk containing none of the
edited token rows is damaged by exactly zero — mechanically forced, its forward
pass is bit-identical — and damage rises with how many edited tokens it
contains, **in non-concept text too**. No scalar delta reproduces an ablation:
the two damage distributions differ in shape, not scale. Before erasing
anything, decode `edited_token_ids` and count what fraction is ordinary English
rather than concept-specific (Rome ~1/3, Baseball ~1/5) — that predicts broad
damage better than the accuracy-based specificity metric, which read 1.000
throughout.

A **stripped** directory keeps its config, tokenizer, index, card and a
`WEIGHTS_REMOVED.md`; only the `.safetensors` are gone. Each names the command
that rebuilds it, in minutes. **Do not rebuild by refitting** — a sparse
factorisation does not reproduce across GPU models at a fixed seed, so a refit
gives a different feature set wearing the same name. The surviving factorisation
caches are the irreplaceable artifact, not these directories.

## Evaluating any of these

They are **base** models with no instruction tuning. Letter-parsing evaluators
score them near zero — given EMBER's multiple-choice prompt they do not emit a
letter at all. Score option text by log-likelihood, prefer a declarative stem
over `Question: ...\nAnswer:`, and normalise per character. `acc_raw` is a trap.
Chunk loss is the sensitive instrument; 4-option MC at n=50 carries ±7 points
and will not resolve effects that the continuous statistic sees at p<0.001 on
identical forward passes. Measured details in `ember_eval/EVALUATION.md`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto")
```

## Storage and backups

These live on `/home/dcor/galbarak2/hf-models` (netapp1), with a second copy of
the twins on a different filer at
`/home/morg/NLP_2526b/galbarak2/backups/lment-2e/hf-models` (netapp2), and the
`.distcp` training checkpoints mirrored the other way into
`/home/dcor/galbarak2/backups/lment-2e/checkpoints`.

**An HF export is not a checkpoint backup.** The `.distcp` directory is fp32
master weights plus optimizer moments at 15 GB and is the only thing a training
run can resume from; the HF safetensors are a lossy weights-only derivative at
5.1 GB. distcp → HF works; HF → distcp does not. The private Hub repos under
`GalBarak/` are weights-only for the same reason.

`/home/dcor` is quota-bound at **1 TB** and `df` does not show it — read the
`dcor-01-2021` line of `quota -s` before any large write. When it fills, jobs
die with `OSError: [Errno 122]` at whatever line happened to be writing, so the
traceback points at PIL or json rather than at storage.

Copies of every model card are versioned in the repo under
`Untaught/model_cards/`. Keep them in sync — nothing does it automatically.

Full documentation:

    git@github.com:itamar-stahl/LMEnt.git
    ember_eval/EVALUATION.md      <- what the instrument can and cannot resolve
    ember_eval/ROME_RESULTS.md    <- the Ancient Rome arm
    ember_eval/BASEBALL_RESULTS.md
    ember_eval/AI_RESULTS.md      <- and whether erasure reaches never-having-learned
    ember_eval/ERASURE_RESULTS.md <- why erasure damage is lexical

Contact: Gal Barak <galll.barak@gmail.com>.
