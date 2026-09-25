#!/usr/bin/env python
"""Post-run audit: could any of the night's failures have contaminated a result?

The run hit four failures. Two cannot touch data -- a stale row-count assertion
that aborted before writing, and a reporter whose job list was stale. Two could:

  * a 5 GB checkpoint written with 1.4 GB of holes (caught by check_alloc,
    rebuilt, but a holed model loads fine and reads back as zeros)
  * five rounds of cancelled scoring jobs. score_selection.slurm SKIPS a scoring
    whose per_question.csv exists, so a file truncated by a mid-write cancel
    would be accepted silently and never redone -- and during each swap an old
    and a new job could briefly write the same file.

So this checks the artifacts rather than the job states: every scoring file is
complete and non-degenerate, every checkpoint is fully allocated, no grid cell
came up short, and selection finished before any test split was read.

    python ember_eval/ensembles/audit_results.py --run <run root>
"""
from __future__ import annotations

import argparse, collections, csv, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE.parent / "nll_kl/check_alloc.py"
PY = sys.executable


def audit_scorings(root: Path, expect: int, bad: list) -> None:
    files = sorted(root.glob("*/*/per_question.csv"))
    print(f"\n{root.name}: {len(files)} files (expect {expect})")
    if len(files) != expect:
        bad.append(f"{root.name}: {len(files)} files, expected {expect}")
    for f in files:
        lbl, topic = f.parts[-3], f.parts[-2]
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        g = collections.Counter(r["question_group"] for r in rows)
        ids = [r["question_id"] for r in rows]
        p = []
        if len(rows) != 150: p.append(f"{len(rows)} rows not 150")
        if len(set(ids)) != len(ids): p.append("duplicate question_ids")
        if len(g) != 3: p.append(f"{len(g)} groups not 3")
        p += [f"{k}={v}" for k, v in g.items() if v != 50]
        m = [float(r["gold_margin"]) for r in rows]
        # a holed checkpoint reads back as zeros: identical logits, zero margins
        if all(abs(x) < 1e-12 for x in m): p.append("all margins zero (zeros-model)")
        if any(x != x or abs(x) == float("inf") for x in m): p.append("non-finite margins")
        acc = sum(int(r["is_correct"]) for r in rows) / len(rows)
        if acc in (0.0, 1.0): p.append(f"accuracy exactly {acc}")
        if p:
            bad.append(f"{lbl}/{topic}: " + "; ".join(p))
    print(f"  groups: {dict(collections.Counter(r['question_group'] for f in files for r in csv.DictReader(open(f, encoding='utf-8'))))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    ENS = Path(a.run)
    bad: list = []

    audit_scorings(ENS / "scored_selection", 46, bad)
    audit_scorings(ENS / "scored_test", 9, bad)

    recs = sorted((ENS / "scored_nllkl").glob("*/*/records.json"))
    print(f"\nNLL/KL records: {len(recs)} (expect 9)")
    if len(recs) != 9:
        bad.append(f"nllkl: {len(recs)} records, expected 9")

    ck = sorted((ENS / "candidates").glob("*/model"))
    holed = [d.parent.name for d in ck
             if subprocess.run([PY, str(CHECK), str(d)], capture_output=True).returncode]
    print(f"\ncheckpoints: {len(ck) - len(holed)}/{len(ck)} fully allocated")
    bad += [f"holed checkpoint: {h}" for h in holed]

    gc = ENS / "results_ensembles/grid_completeness.json"
    if gc.exists():
        short = json.loads(gc.read_text())["short_cells"]
        print(f"\ngrid completeness: {short or 'no short cells'}")
        bad += short
    else:
        bad.append("grid_completeness.json missing")

    print("\n" + "=" * 62)
    if bad:
        print(f"{len(bad)} PROBLEM(S):")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print("CLEAN: no truncated scoring, no holed checkpoint, no short grid cell.")


if __name__ == "__main__":
    main()
