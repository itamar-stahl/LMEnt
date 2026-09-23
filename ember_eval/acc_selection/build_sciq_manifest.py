#!/usr/bin/env python
"""Freeze a sciq-based UNRELATED set: 50 selection + 50 test, disjoint.

Replaces the other-concept trivia pool, which the full model cannot do: on the
frozen sets its unrelated accuracy was 0.280 (Rome) and 0.340 (Baseball), both
with a 95% CI spanning chance, so (Acc_F - 0.25) was a denominator whose sign was
not established. Measured with the same scorer, sciq gives 0.60 [0.462, 0.724],
a denominator of 0.35 rather than 0.03.

sciq is topic-independent on purpose. It is a general-capability probe -- the role
EMBER fills with MMLU -- so ONE set serves all three concepts and preservation
becomes directly comparable across them. A single held-out concept (World War II
scored 0.58) would work numerically but is conceptually a second neighbour: it
measures one domain, so damage specific to that domain is indistinguishable from
general damage.

Options are the correct answer plus sciq's three distractors, shuffled per item
with a fixed seed. Chance is 0.25, matching the target and neighbour groups.

    python build_sciq_manifest.py --out sciq_unrelated.csv
"""
from __future__ import annotations

import argparse, csv, hashlib, random
from pathlib import Path

SEED = 20260923
N_PER_PHASE = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("sciq", split="validation")
    rng = random.Random(SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    rows, seen_q = [], set()
    for i in idx:
        r = ds[i]
        q = str(r["question"]).strip()
        gold = str(r["correct_answer"]).strip()
        opts = [gold, str(r["distractor1"]).strip(),
                str(r["distractor2"]).strip(), str(r["distractor3"]).strip()]
        if not q or not gold or len(set(opts)) != 4 or q in seen_q:
            continue
        if any(not o for o in opts):
            continue
        seen_q.add(q)
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[j] for j in order]
        phase = "selection" if len(rows) < N_PER_PHASE else "test"
        rows.append({
            "topic": "sciq", "question_group": f"unrelated_{phase}", "phase": phase,
            "question_id": hashlib.sha1(f"sciq\x1f{q}".encode()).hexdigest()[:12],
            "source_topic": "sciq", "source_split": "validation",
            "question": q,
            # sciq items are questions, not declarative statements; the bare
            # question is the context, matching how the probe measured 0.60.
            "stem": q,
            "correct_answer": gold,
            "option_0": shuffled[0], "option_1": shuffled[1],
            "option_2": shuffled[2], "option_3": shuffled[3],
            "gold_index": shuffled.index(gold),
        })
        if len(rows) == 2 * N_PER_PHASE:
            break

    if len(rows) != 2 * N_PER_PHASE:
        raise SystemExit(f"only built {len(rows)} items, need {2*N_PER_PHASE}")
    sel = {r["question_id"] for r in rows if r["phase"] == "selection"}
    tst = {r["question_id"] for r in rows if r["phase"] == "test"}
    assert len(sel) == len(tst) == N_PER_PHASE, (len(sel), len(tst))
    assert not (sel & tst), "selection and test share items"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"{len(rows)} items -> {out}")
    print(f"  selection {len(sel)}, test {len(tst)}, disjoint")
    print(f"  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
