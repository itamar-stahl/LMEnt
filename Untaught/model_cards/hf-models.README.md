# LMEnt 1B twin pairs — single-entity pretraining ablations

Models trained on the LMEnt Wikipedia corpus, in **pairs** that are identical
except that one twin held every chunk mentioning **Pornography (Wikidata `Q291`)**
out of the training loss.

| directory | epochs | role | state |
|---|---|---|---|
| `lment-1b-control` | 1 | control, nothing held out | **ready** |
| `lment-1b-noporn` | 1 | ablated, `Q291` held out | **ready** |
| `lment-1b-noporn-2e` | 2 | ablated, `Q291` held out | **ready** |
| `lment-1b-control-2e` | 2 | control, nothing held out | **still training** |

**A twin is only meaningful against its own control.** The design is that the two
models in a pair differ by exactly the masked gradient contributions and by
nothing else — same data, same order, same step count, same seed. So:

- compare `lment-1b-noporn` against `lment-1b-control` (both 1 epoch) — valid;
- compare `lment-1b-noporn-2e` against `lment-1b-control-2e` (both 2 epochs) —
  valid **once the latter finishes**;
- compare a 2-epoch model against a 1-epoch one and you are measuring training
  duration, not the ablation;
- compare either against the authors' released `dhgottesman/LMEnt-1B-*` and you
  are measuring a different batch size, optimizer trajectory and code version.

The 2-epoch pair exists because the 1-epoch effect sat at the edge of what the
instrument could resolve, and scoring the authors' released 1E/2E/4E/6E models
showed the second epoch adds more knowledge of this particular concept than
almost any other — while epochs 3 through 6 add nothing. See
`Untaught/COMPARABILITY.md`.

Each directory has its own README with training details, verification that the
ablation fired, and the caveats worth reading first. Full documentation:

    git@github.com:itamar-stahl/LMEnt.git   branch itamars/main
    Untaught/STATUS.md       <- start here
    Untaught/RESULTS.md      <- what was trained, and proof the ablation fired
    ember_eval/EVALUATION.md <- what the evaluation found, and what it did not

These live on `/vol/scratch`, which has no documented retention policy. Copies of
every model card are versioned in the repo under `Untaught/model_cards/`.

Contact: Gal Barak <galll.barak@gmail.com>.
