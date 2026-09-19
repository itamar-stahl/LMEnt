# Model cards, as deployed

Copies of the `README.md` files that sit inside the checkpoint directories on
`/home/dcor/galbarak2/hf-models/`. They live there because that is where a
colleague handed a path actually lands — a directory of safetensors says nothing
about which twin it is or that a control exists.

They are versioned here for two reasons: storage under these models has already
failed once — `/vol/scratch` was purged without warning on 2026-08-23 — and a
model card is worth reviewing in a diff like any other document.

Every file is `<directory name>.README.md`, deployed to
`/home/dcor/galbarak2/hf-models/<directory name>/README.md`; `hf-models.README.md`
is the top-level `hf-models/README.md`.

## Twins — batch 131,072, 2 epochs

| file here | what |
|---|---|
| `lment-1b-control-2e-b131k.README.md` | the control for all three current arms |
| `lment-1b-norome-2e-b131k.README.md` | Ancient Rome ablated |
| `lment-1b-nobaseball-2e-b131k.README.md` | Baseball ablated |
| `lment-1b-noai-2e-b131k.README.md` | Artificial Intelligence ablated |

## The retired Pornography pair

| file here | what |
|---|---|
| `lment-1b-control-2e.README.md` | control, 2 epochs |
| `lment-1b-noporn-2e.README.md` | `Q291` ablated, 2 epochs |
| `lment-1b-control.README.md` | control, 1 epoch — **the model is gone** |
| `lment-1b-noporn.README.md` | `Q291` ablated, 1 epoch — **the model is gone** |

The last two describe checkpoints destroyed when `/vol/scratch` was purged on
2026-08-23 and deliberately not retrained. The cards are kept as the record;
the per-question evaluation results survive, so published analyses reproduce.

## Erased models — edited weights, not twins

| file here | method | concept |
|---|---|---|
| `lment-1b-rome-erased-b131k.README.md` | EMBER, delta 200 | Ancient Rome |
| `lment-1b-rome-erased-d{2,5,10,50,500,1000}-b131k.README.md` | EMBER, the rest of the sweep | Ancient Rome |
| `lment-1b-baseball-erased-b131k.README.md` | EMBER, delta 10 | Baseball |
| `lment-1b-baseball-erased-d200-b131k.README.md` | EMBER, delta 200 | Baseball |
| `lment-1b-ai-erased-b131k.README.md` | EMBER, delta 5.0 | Artificial Intelligence |
| `lment-1b-ai-snmf-b131k.README.md` | **SNMF (MLP), not EMBER** | Artificial Intelligence |

## Two cautions

**Cards for erased models were wrong until 2026-09-19.**
`ember_eval/materialize_erased_model.py` copied every non-shard file out of the
base directory, `README.md` included, so nine erased models each carried a
byte-identical copy of the control's card — one that opens by calling the
directory "the Ancient Rome pair's control" and describes it as "a pristine
reference copy". The script now skips `README.md` and writes a generated card
instead; the nine deployed cards were rewritten by hand.

**A card here is a copy, not the source of truth for the model.** If you change
one, change both — nothing keeps them in sync automatically.
