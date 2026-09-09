"""Prove snmf.py's local judge answers both stages. Needs a GPU and ~15 min.

    python mlp_erasure/tests/judge_gpu_smoke.py

This is the one part of `snmf.py select` that cannot be tested without loading
23.9 GB off the filer, so it is deliberately separate from the fast suite --
the same split the fork uses for its own `*_gpu_smoke.py`. It sends the real
STAGE1 and STAGE2 prompts, with token lists chosen so the right answers are
obvious, and checks the reply parses through `_parse_membership`.

It says nothing about which features a real run would select. It says the
seam is wired, which is what `--judge gemma` needed proving.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import snmf   # noqa: E402

# One list that is plainly the concept and one that is plainly not, so a
# disagreement is the judge failing rather than the question being hard.
ROME_TOKENS = ["▁Rome", "▁Roman", "▁Caesar", "▁Senate",
               "▁legion", "▁Augustus", "▁Republic",
               "▁emperor", "▁consul", "▁Latin"]
OFF_TOKENS = ["▁kilometre", "▁hectare", "▁metres", "▁acre",
              "▁inches", "▁millimetre", "▁litres", "▁tonne",
              "▁kilograms", "▁feet"]


def main():
    client = snmf.LocalGemmaClient(fork_root=str(ROOT / "Ember-on-LMEnt"))
    failures = []
    try:
        for name, tokens, expected in (("rome", ROME_TOKENS, True),
                                       ("units", OFF_TOKENS, False)):
            record = snmf._judge_evidence(client, "Ancient Rome", tokens,
                                          top_k=20, confidence_threshold=0.85)
            print(json.dumps({name: record}, indent=2, ensure_ascii=False))
            if not record["description"].strip():
                failures.append(f"{name}: empty stage-1 description")
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
    print("OK: both stages answered and both memberships came back as expected")


if __name__ == "__main__":
    main()
