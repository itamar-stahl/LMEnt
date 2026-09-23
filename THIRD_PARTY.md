# Third-party code in this repository

LMEnt vendors nine external projects — their source is committed here rather than
referenced. There are **no git submodules**: a plain `git clone` gets everything.

Two tiers:

- `third_party/` — five projects the pipeline is built on.
- `Ember-on-LMEnt/external/` — four erasure/interpretability methods compared
  against, put on `PYTHONPATH` by `Ember-on-LMEnt/activate_env.sh`.

Everything here keeps its upstream licence. **They are not all the same, and two
are NonCommercial** — read the table before redistributing any of it.

## Why vendored rather than submodules

These five *were* real submodules of forks under `github.com/dhgottesman/`, until
commit `cbb57ae` ("submodules into main repository", 2026-08-10) inlined 982
files in one go. The recorded submodule pins are still readable from history with
`git ls-tree cbb57ae^`, and they are the SHAs in the table below.

## `third_party/`

Divergence measured by diffing each tree against a fresh clone of its fork at the
pinned commit. "Identical" means byte-for-byte, no exceptions.

| Directory | Upstream | Fork, pinned at | Licence | Diverges from pin? |
|---|---|---|---|---|
| `dolma/` | [allenai/dolma](https://github.com/allenai/dolma) | `dhgottesman/dolma` @ `604a133a` | Apache-2.0 | No — identical |
| `olmes/` | [allenai/olmes](https://github.com/allenai/olmes) | `dhgottesman/olmes` @ `1f0568fb` | Apache-2.0 | No — identical |
| `maverick-coref/` | [SapienzaNLP/maverick-coref](https://github.com/SapienzaNLP/maverick-coref) | `dhgottesman/maverick-coref` @ `0f0f1120` | **CC BY-NC-SA 4.0** | No — identical |
| `ReFinED/` | [amazon-science/ReFinED](https://github.com/amazon-science/ReFinED) | `dhgottesman/ReFinED` @ `0daa6181` | **CC BY-NC 4.0** | Only `dist/` (a built `.whl`, deliberately not committed) |
| `OLMo-core/` | [allenai/OLMo-core](https://github.com/allenai/OLMo-core) | `dhgottesman/OLMo-core` @ `08b63de0` | Apache-2.0 | **Yes — one real commit, see below** |

### OLMo-core carries an LMEnt-only fix — do not lose it

`third_party/OLMo-core/src/olmo_core/nn/transformer/model.py` differs from the
pinned fork commit by 17 lines. That difference is commit `885d87c`
("Initialise embeddings after the reset_parameters sweep, not before",
2026-08-29), and it is a **correctness fix that exists in this monorepo and
nowhere else**. Its own comment records what it fixes:

> `reset_parameters()` reaches `nn.Embedding.reset_parameters()`, which is
> `init.normal_(weight)` with the default `std=1.0`, so initialising the
> embeddings first meant the `std=0.02` draw was silently overwritten with an
> N(0,1) one. Every model trained from this file before 2026-08-29 has
> embeddings 50x too large, carrying no usable geometry.

Consequences to respect:

- **Converting `OLMo-core/` to a submodule pinned at `08b63de0` would silently
  revert this fix.** Push `885d87c` to the fork first, and pin past it.
- Checkpoints trained before 2026-08-29 predate the fix, so "which commit
  produced these weights" is a reproducibility question, not just a pinning one.

The only other difference, `docs/make.bat`, is not a real change: upstream ships
it with CRLF and this repo's `.gitattributes` normalises to LF. Identical once
carriage returns are stripped.

## `Ember-on-LMEnt/external/`

| Directory | Upstream | Licence |
|---|---|---|
| `CRISP/` | [tomerashuach/CRISP](https://github.com/tomerashuach/CRISP) ([paper](https://arxiv.org/abs/2508.13650)) | **No LICENSE file present** |
| `snmf/` | [ordavid-s/snmf-mlp-decomposition](https://github.com/ordavid-s/snmf-mlp-decomposition) | **No LICENSE file present** |
| `PISCES/` | not recorded in-tree; `LICENSE` is MIT, © 2025 yoavgur | MIT |
| `wmdp/` | not recorded in-tree; `LICENSE` is MIT, © 2024 centerforaisafety | MIT |

`CRISP/` and `snmf/` ship no licence text at all. Vendored code with no stated
licence carries no permission to redistribute, so both need a licence confirmed
with their authors before this repository is published.

## Re-measuring this

To re-check any tree against its pin:

```bash
git clone https://github.com/dhgottesman/<name>.git /tmp/<name>
git -C /tmp/<name> checkout <pinned-sha>
rm -rf /tmp/<name>/.git
diff -rq /tmp/<name> third_party/<name>
```

Differences only in line endings are expected for files upstream ships with
CRLF, because `.gitattributes` sets `* text=auto eol=lf`. Compare with
`diff <(tr -d '\r' < a) <(tr -d '\r' < b)` to rule that out.

Note that a blanket ignore rule can silently drop vendored files at `git add`
time without reporting it — three such rules in the root `.gitignore` had
removed 13 files from these trees, including a ReFinED source module and an
AGIEval data file. See the comment at the top of `.gitignore`.
