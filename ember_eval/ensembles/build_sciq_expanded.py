#!/usr/bin/env python
"""Grow the SciQ TEST half, leaving the frozen 100 items byte-identical.

The preservation term is currently measured on 50 SciQ questions, and almost
every model scores exactly 26/50 -- including the unerased full model -- so
`norm_unrelated` clips at 1.0 in 14 of 18 cells and the term does no work. That
is a sample-size problem, not a dataset problem: SciQ's validation split has
1000 items and only 100 are in use.

This continues the ORIGINAL builder's walk rather than redrawing it. Same seed,
same shuffle, same per-item option shuffle, so the RNG stream up to item 100 is
untouched and those items keep their questions, option order and question_id.
The selection half stays at 50 and is not extended -- it chose every checkpoint,
and nothing here may disturb that. Only the test half grows.

Verifies the overlap against the frozen manifest and refuses to write if a
single existing row differs.

    python ember_eval/ensembles/build_sciq_expanded.py --n-test 550 --out <csv>
"""
from __future__ import annotations

import argparse, csv, hashlib, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260923          # as in build_sciq_manifest.py
N_SELECTION = 50         # frozen: this half chose the checkpoints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=550)
    ap.add_argument("--frozen", default=str(HERE.parent / "acc_selection/sciq_unrelated.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("sciq", split="validation")
    rng = random.Random(SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    rows, seen_q = [], set()
    want = N_SELECTION + a.n_test
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
        rng.shuffle(order)                     # advances the stream exactly as before
        shuffled = [opts[j] for j in order]
        phase = "selection" if len(rows) < N_SELECTION else "test"
        rows.append({
            "topic": "sciq", "question_group": f"unrelated_{phase}", "phase": phase,
            "question_id": hashlib.sha1(f"sciq\x1f{q}".encode()).hexdigest()[:12],
            "source_topic": "sciq", "source_split": "validation",
            "question": q, "stem": q, "correct_answer": gold,
            **{f"option_{k}": shuffled[k] for k in range(4)},
            "gold_index": shuffled.index(gold)})
        if len(rows) == want:
            break

    if len(rows) != want:
        raise SystemExit(f"pool exhausted: built {len(rows)}, wanted {want}")

    # the frozen 100 must survive untouched, field for field
    frozen = list(csv.DictReader(open(a.frozen, encoding="utf-8")))
    if len(frozen) != 100:
        raise SystemExit(f"expected 100 frozen rows, found {len(frozen)}")
    diffs = []
    for k, (old, new) in enumerate(zip(frozen, rows)):
        for col in old:
            if col in new and str(old[col]) != str(new[col]):
                diffs.append(f"row {k} col {col}: {old[col]!r} != {new[col]!r}")
    if diffs:
        print(f"REFUSING TO WRITE: {len(diffs)} frozen rows changed", file=sys.stderr)
        for d in diffs[:8]:
            print("  " + d, file=sys.stderr)
        raise SystemExit(1)

    sel = [r for r in rows if r["phase"] == "selection"]
    tst = [r for r in rows if r["phase"] == "test"]
    assert len(sel) == N_SELECTION and not ({r["question_id"] for r in sel}
                                            & {r["question_id"] for r in tst})
    out = Path(a.out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    old_test = {r["question_id"] for r in frozen if r["phase"] == "test"}
    new_test = {r["question_id"] for r in tst}
    print(f"wrote {out}")
    print(f"  selection : {len(sel)} (frozen, unchanged)")
    print(f"  test      : {len(tst)}  = {len(old_test & new_test)} original + "
          f"{len(new_test - old_test)} new")
    print(f"  frozen 100 rows verified identical, field for field")


if __name__ == "__main__":
    main()
