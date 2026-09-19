# LMEnt-1B twin, TWO EPOCHS: ABLATED — Ancient Rome held out

The ablated half of the **Ancient Rome** arm of the 2-epoch twin pair, at batch
131,072. Trained to the full 54,832 steps with 56 Rome QIDs masked from the
loss. This is the pair behind `ember_eval/ROME_RESULTS.md` — the first result in
the project that was large, specific, replicated across independent halves, and
robust to the prior correction.

**Its control is shared with the Baseball and Artificial Intelligence arms.**
All three ablations mask an already-composed batch, so every twin traverses
identical batches and one control serves every arm:

    /home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k     <- the control (job 853707)
    /home/dcor/galbarak2/hf-models/lment-1b-norome-2e-b131k      <- this model
    /home/dcor/galbarak2/hf-models/lment-1b-nobaseball-2e-b131k  <- the Baseball ablated twin
    /home/dcor/galbarak2/hf-models/lment-1b-noai-2e-b131k        <- the AI ablated twin

## What was removed

| | |
|---|---|
| blacklist | `Untaught/blacklists/ancient_rome_core.json` — 56 QIDs |
| chunks excluded | **65,844** — 0.6276% of the 10,491,928-chunk corpus |
| thresholds | hyperlinks 1.0, entity-linking 0.6, coref 0.6, coref-cluster 0.6 |

The per-entity counts sum to 116,116, so 50,272 ids are shared between entities
and the deduplicated union is exactly the declared 65,844. For scale: this is
**26x** the retired Pornography subject (2,546 chunks) and 0.82x the Baseball
footprint (80,466).

Chunks were masked from the **loss**, not removed from the data: they still
occupy batch slots, so batch composition, data order and step count are
identical to the control's. The twins differ by exactly the masked gradient
contributions.

## Proof the ablation fired

**This run is the clean case of the three, and the arithmetic closes.** It
trained in a **single SLURM window** (`resumed_from: null`), so the
`train/untaught excluded` counter — which resets per window — is a whole-run
figure here rather than one leg of many:

| | |
|---|---|
| instance-slots excluded | **131,624** |
| predicted for exactly 2 epochs | `2 x 65,844 = 131,688` |
| agreement | **99.95%** |
| startup confirmation | `loaded 65844 chunk ids` |
| all-masked-batch guard | enabled — `guard_all_masked=True`, `strict=True` |

**On "zero guard leaks", which other cards in this directory claim:** the
framework emits exactly two untaught metrics, `train/untaught excluded
instances` and `train/untaught excluded cumulative`. There is **no leak
counter** in the log. `guard_all_masked` is a *config flag* in
`ChunkExclusionCallback`, not a tally. So "zero leaks" means "the guard was
enabled with `strict=True` and nothing reported a leak" — not that a counter was
read and found to be zero. The evidence that the ablation fired is the 99.95%
arithmetic above, which is a real measurement; treat it as the claim, and read
the guard as a mechanism that was switched on.

**Do not compare this 99.95% against Baseball's or AI's window figures and
conclude those ablations were leakier** — theirs are single windows out of five
and eight respectively, measuring something different. Rome is the only one of
the three that can be checked end to end this way.

## Training

| | |
|---|---|
| duration | **2 epochs** = 54,832 steps |
| final CE loss | 2.460 → **perplexity 11.71** |
| job | `850249`, `COMPLETED`, 1d 16h 20m, ended 2026-09-06T11:44:07 |
| hardware | **H200 (`n-h200`) throughout — no deviation** |

Everything else matches the b131k recipe and is identical to the control's:
`olmo2_1B` (18 layers, d_model 2048, 16 heads), 3.6B-token LMEnt Wikipedia
corpus, global batch 131,072, rank microbatch 16,384, AdamW peak LR 4e-4,
warmup 2,000, cosine to 4e-5, VSL `grow_p2` over 8 cycles, seed 12536, one GPU,
torch 2.6.0+cu124, bf16 params, `compile: true`.

### This is the only twin in the project with no hardware asymmetry

Worth stating plainly, because every other card here carries a caveat and this
one does not. This twin ran start to finish on `n-h200`. Its control ran in two
windows (`850054` then `853707`), **both also on `n-h200`**. So for the Ancient
Rome pair, and only this pair, control and ablated saw the same card for every
step.

By contrast: the Baseball twin ran H100 for windows 1–4 and H200 for window 5;
the AI twin spent ~12 hours on an A6000 across its eight legs; and the retired
Pornography pair's control finished its last ~9.7% of steps on an H200 while its
twin stayed on H100. Measured cross-GPU drift on this project is 3e-5 and flips
nothing in evaluation — but where a weight-space claim needs a pair with no
asymmetry at all to appeal to, this is that pair.

## What it showed

Full write-up: `ember_eval/ROME_RESULTS.md`. In brief, on `pmi/char`, paired
per-question, ablated minus control:

| split | mean delta | dz | p |
|---|---|---|---|
| Rome val | **−0.3745** | −0.79 | 2.2e-08 |
| Rome test | **−0.2797** | −0.76 | 9.3e-08 |
| Simdom-Rome val | −0.0027 | −0.01 | ns |
| Simdom-Rome test | −0.0765 | −0.22 | ns |

−16 accuracy points on Rome on *both halves independently* (56→40, 42→26), −4
and −6 next door. OLMES sciq 0.776 ablated vs 0.770 control, so not a generally
worse model. It survives the prior correction that killed Baseball's +22:
`logp_null` deltas between the twins are noise, and on Rome val the correction
*grows* the gap (−0.326 raw to −0.375 corrected). The 8-concept cross-concept
null puts it at **z −7.12** (QA val), 3.58x the largest null concept.

Held-out chunk loss: +0.2384 nats/token diff-of-diffs, dz 1.05 — the **largest
mean shift** of the three subjects but the **smallest dz**, i.e. far more
per-chunk variance than AI's +0.1561 at dz 1.83.

Embedding health: ratio **2.9374 / 2.9361** for the pair, agreeing to 0.05%, and
matching the AI twin's 2.936 to within 0.01%.

**One caution carried from `ERASURE_RESULTS.md`:** EMBER's erasure of Rome
*overshoots* this ablation (erased−control −0.652 against twin−control −0.375,
residual dz −0.315 with a CI excluding zero), and the mechanism is lexical, not
conceptual. Rome is the subject where erasure does **not** reach
never-having-learned; AI is the one where it does.

## Provenance

| | |
|---|---|
| SLURM job | `850249` |
| Run folder | `untaught-no-rome-core-1b-2e-b131k_20260904_183200` (on `/home/morg`, `LMEnt-initfix`) |
| Step | `54832` / 54,832 (2 epochs, complete) |
| Converted from | `.../checkpoints/olmo2_1B_0.0004_131072_0.05_2/step54832` |
| Converted by | `convert_checkpoint_to_hf.py` via `lment-rome-check/convert_norome_twin.slurm`, job `909830` |
| Converted on | 2026-09-19 |
| Tokenizer | taken from the control's own export, not the Hub |

Verified after conversion: 16 `.distcp` shards plus `.metadata` at the source;
**201 tensors across 2 safetensors shards** with a complete index at the
destination; `config.json` **byte-identical** to the control's (18 layers,
d_model 2048, vocab 100,352, `tie_word_embeddings: false`); and all five
tokenizer files — `tokenizer.json`, `vocab.json`, `merges.txt`,
`special_tokens_map.json`, `tokenizer_config.json` — **byte-identical** to the
control's, so the twins cannot differ on that axis even in serialisation.

The checkpoint was read with `LMEnt-initfix`'s OLMo-core, the tree that *wrote*
it. Pointing a different OLMo-core at a checkpoint is what produced the B200
"Missing key in checkpoint state_dict" failure (`Untaught/B200.md`).

### The conversion is reproducible, and that was checked rather than assumed

An earlier export of this same checkpoint was made on 2026-09-06 into
`lment-rome-check/hf/norome-2e-step54832`. Both safetensors shards of this
export are **md5-identical** to it:

    model-00001-of-00002.safetensors   c5243a0d959120309d67ea74de69964d
    model-00002-of-00002.safetensors   a58f3fb2a557579249e80cd898636548

Two independent runs of `convert_checkpoint_to_hf.py`, thirteen days apart, on
different nodes, produce the same bytes. That validates the 09-06 copy as a
faithful export as much as it validates this one. Note this is **not** true of
everything in this project: the EMBER sparse factorisation does *not* reproduce
across GPU models at a fixed seed. Conversion is deterministic; fitting is not.

## Backups

Mirrored across independent filers (`/home/dcor` is netapp1, `/home/morg` is
netapp2), **both legs verified by content hash**, not by size — job `909940`,
2026-09-19:

- HF export → `/home/morg/NLP_2526b/galbarak2/backups/lment-2e/hf-models/lment-1b-norome-2e-b131k` (5.1 G)
- distcp checkpoint → `/home/dcor/galbarak2/backups/lment-2e/checkpoints/untaught-no-rome-core-1b-2e-b131k_20260904_183200/step54832` (15 G, 16 shards + `.metadata`)

`rsync -c` rather than the default size+mtime comparison is the point:
`/home/dcor` is mounted **`soft`**, so a write burst that crosses the NFS
timeout returns EIO rather than blocking, and a truncated file can land with a
plausible size. This verification was first attempted as a login-node
background task and was **killed twice by host memory pressure** — the copy
survived, the verification did not. Run multi-GB verifies as a batch job.

The HF copy is **not** a checkpoint backup: the `.distcp` directory is
`model_and_optim`, fp32 master weights plus optimizer moments at 15 GB. The HF
safetensors are a lossy weights-only derivative at 5.1 GB. distcp → HF works;
HF → distcp does not.

Until 2026-09-19 this model had **neither** — it existed only as the raw distcp
checkpoint on a single filer, plus the un-mirrored 09-06 HF export above. It was
the last of the three ablated twins in that state.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
p = "/home/dcor/galbarak2/hf-models/lment-1b-norome-2e-b131k"
tok = AutoTokenizer.from_pretrained(p)
model = AutoModelForCausalLM.from_pretrained(p, torch_dtype="auto")
```

These are **base** models with no instruction tuning. Letter-parsing evaluators
score them near zero — given a multiple-choice prompt they do not emit a letter
at all. Score option text by log-likelihood, prefer a declarative stem over
`Question: ...\nAnswer:`, and normalise per character. See
`ember_eval/EVALUATION.md`.

> Keep this file and the copy in `Untaught/model_cards/` in sync — nothing does
> it automatically.
