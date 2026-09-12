"""Prove snmf.py's local judge answers both stages. Needs a GPU and ~15 min.

    python mlp_erasure/tests/judge_gpu_smoke.py

This is the one part of `snmf.py select` that cannot be tested without loading
23.9 GB off the filer, so it is deliberately separate from the fast suite --
the same split the fork uses for its own `*_gpu_smoke.py`. It sends the real
STAGE1 and STAGE2 prompts and checks the reply parses.

It now sends the format a real run sends. `select` passes the activation
source as {token, context, activation} dicts and the projection source as
{token, score} dicts, and the two go to different STAGE1 templates. Sending
bare strings here exercised neither, and silently took the degraded
no-context path that the fix exists to avoid.

It says nothing about which features a real run would select. It says the
seam is wired, which is what `--judge gemma` needed proving.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import snmf   # noqa: E402

# One feature that is plainly the concept and one that is plainly not, so a
# disagreement is the judge failing rather than the question being hard.
ROME_ACTIVATION = [
    {"token": "▁Rome", "context": "the roads of ▁Rome led to the forum",
     "activation": 0.94},
    {"token": "▁Caesar", "context": "when ▁Caesar crossed the Rubicon",
     "activation": 0.88},
    {"token": "▁Senate", "context": "the ▁Senate met on the Capitoline",
     "activation": 0.81},
    {"token": "▁legion", "context": "a ▁legion marched north from Gaul",
     "activation": 0.77},
    {"token": "▁Augustus", "context": "under ▁Augustus the empire settled",
     "activation": 0.70},
]
OFF_ACTIVATION = [
    {"token": "▁kilometre", "context": "about a ▁kilometre from the coast",
     "activation": 0.91},
    {"token": "▁hectare", "context": "roughly one ▁hectare of farmland",
     "activation": 0.85},
    {"token": "▁tonne", "context": "a ▁tonne of grain per season",
     "activation": 0.79},
    {"token": "▁acre", "context": "an ▁acre was fenced off",
     "activation": 0.74},
    {"token": "▁litres", "context": "several ▁litres of water daily",
     "activation": 0.68},
]
ROME_PROJECTION = [{"token": t, "score": s} for t, s in (
    ("▁Rome", 14.2), ("▁Roman", 13.8), ("▁Caesar", 12.9), ("▁Senate", 12.1),
    ("▁legion", 11.4), ("▁consul", 10.8), ("▁Latin", 10.2))]
OFF_PROJECTION = [{"token": t, "score": s} for t, s in (
    ("▁kilometre", 13.9), ("▁hectare", 13.1), ("▁metres", 12.6),
    ("▁acre", 11.9), ("▁inches", 11.2), ("▁tonne", 10.7))]

CASES = (
    ("rome/activation", ROME_ACTIVATION, "activation", True),
    ("units/activation", OFF_ACTIVATION, "activation", False),
    ("rome/projection", ROME_PROJECTION, "projection", True),
    ("units/projection", OFF_PROJECTION, "projection", False),
)


def main():
    client = snmf.LocalGemmaClient(fork_root=str(ROOT / "Ember-on-LMEnt"))
    failures = []
    try:
        for name, data, source, expected in CASES:
            record = snmf._judge_evidence(client, "Ancient Rome", data,
                                          top_k=10, confidence_threshold=0.85,
                                          source=source)
            print(json.dumps({name: record}, indent=2, ensure_ascii=False))
            if not record["has_context"]:
                failures.append(
                    f"{name}: evidence was read as bare strings, so this "
                    "exercised the degraded path rather than the real one")
            if not record["description"].strip():
                failures.append(f"{name}: empty stage-1 description")
            if record["trash"] and expected:
                failures.append(f"{name}: judge returned TRASH for a "
                                "feature that plainly is the concept")
            if record["is_member"] is not expected:
                failures.append(
                    f"{name}: is_member={record['is_member']}, "
                    f"expected {expected}")
    finally:
        client.close()

    if failures:
        print("FAILED:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print(f"OK: {len(CASES)} cases, both stages answered, both sources "
          "exercised, memberships as expected")


if __name__ == "__main__":
    main()
