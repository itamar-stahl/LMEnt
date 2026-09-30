#!/usr/bin/env python
"""Emit the frozen sciq set in nll_kl's sets format, so score_model.py can run it.

The accuracy pipeline scores sciq as four options ranked by per-character
log-probability. NLL and KL need something different: the single correct
continuation, plus the full next-token distribution at each of its positions.
Same questions, same gold answers, different instrument -- so this converts the
frozen sciq_unrelated.csv into the schema score_model.py already consumes rather
than teaching either scorer about the other's format.

Item ids are carried over unchanged, so a sciq question lines up across the
accuracy and NLL/KL exports.

    python build_sciq_nllkl_sets.py --out ember_eval/nll_kl/sets/sciq.json
"""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "sciq_unrelated.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    sets: dict[str, list] = {}
    for r in rows:
        item = {
            "id": r["question_id"], "role": "unrelated", "phase": r["phase"],
            "source_topic": "sciq", "source_split": "validation",
            "question": r["question"], "stem": r["stem"],
            # the scorer conditions on stem and scores this string verbatim;
            # the leading space matters, as elsewhere in this project
            "completion": " " + r["correct_answer"].strip(),
        }
        sets.setdefault(r["question_group"], []).append(item)

    for k, v in sets.items():
        if len(v) != 50:
            raise SystemExit(f"{k}: {len(v)} items, expected 50")
    ids = [i["id"] for v in sets.values() for i in v]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate item ids")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "topic": "sciq",
        "source": "ember_eval/acc_selection/sciq_unrelated.csv",
        "seed": 20260923,
        "note": "general-capability unrelated set; topic-independent, one set for all concepts",
        "sets": sets,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{sum(len(v) for v in sets.values())} items -> {out}")
    for k, v in sorted(sets.items()):
        print(f"  {k}: {len(v)}")
    print(f"  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
