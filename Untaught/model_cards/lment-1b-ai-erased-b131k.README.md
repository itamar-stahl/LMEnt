# lment-1b-ai-erased-b131k — EMBER embedding erasure of Artificial Intelligence

The Artificial Intelligence pair's **control** twin with AI erased from the
input embedding after the fact. Its counterpart is the *untaught* twin
`lment-1b-noai-2e-b131k`, which never received gradient from AI chunks at all.
Comparing the two is the never-learned-vs-erased contrast the pair exists for.

## Provenance

| | |
|---|---|
| Base | `lment-1b-control-2e-b131k` (job 853707, step 54832) |
| Erasure run | job `907972`, 2026-09-18 |
| Run folder | `Artificial_intelligence_lment-1b-control-2e-b131k_20260918_144847` |
| Method | EMBER, embedding only (`lm_head` untouched) |
| Rank / seed | 100 / 44 |
| Feature cache | `lment-ember-grid-ai/features/sp0.02_seed44` (reused, not refit) |
| Selected features | **26, 85, 89** (Gemma judge, `mode: judge`) |
| Chosen delta | **5.0** (selected by the pipeline's own delta search) |

## Why the judge ran, unlike Rome and Baseball

Those concepts use `mode: threshold` to isolate the judge's single pick, which
works only because the pick was also the unique maximum of `ratio_abs`. For AI
it is not: the judge's top feature 89 scores 4.678 while features 94 (6.340) and
55 (6.196) outrank it and the judge rejected both. No threshold selects 89
alone, so the judge had to run.

**AI needed three features where Rome and Baseball each needed one**, which is
consistent with AI being a broader concept than a named entity.

## Judge reproducibility

The 2026-09-18 run reproduced the 2026-09-14 run's accepted set exactly —
{26, 85, 89}, with bit-identical metric scores, confirming the cached
factorisation was replayed rather than refit. Only feature 26's free-text
description differed, and its confidence moved 0.85 → 0.95.

Note that 0.85 sat exactly **at** `judge_confidence_threshold`, so in the
original run feature 26 passed by zero margin. The erasure's composition is one
judge wobble away from being two features rather than three.
