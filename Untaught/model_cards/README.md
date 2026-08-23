# Model cards, as deployed

Copies of the `README.md` files that sit inside the checkpoint directories on
`/home/dcor/galbarak2/hf-models/`. They live there because that is where a
colleague handed a path actually lands — a directory of safetensors says nothing
about which twin it is or that a control exists.

They are versioned here for two reasons: storage under these models has already
failed once — `/vol/scratch` was purged without warning on 2026-08-23 — and a
model card is worth reviewing in a diff like any other document.

| file here | deployed to |
|---|---|
| `hf-models.README.md` | `/home/dcor/galbarak2/hf-models/README.md` |
| `lment-1b-control.README.md` | `/home/dcor/galbarak2/hf-models/lment-1b-control/README.md` |
| `lment-1b-noporn.README.md` | `/home/dcor/galbarak2/hf-models/lment-1b-noporn/README.md` |
| `lment-1b-noporn-2e.README.md` | `/home/dcor/galbarak2/hf-models/lment-1b-noporn-2e/README.md` |
| `lment-1b-control-2e.README.md` | `/home/dcor/galbarak2/hf-models/lment-1b-control-2e/README.md` |

If you change one, change both — nothing keeps them in sync automatically.
