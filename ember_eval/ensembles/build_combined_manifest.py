#!/usr/bin/env python
"""One manifest per topic carrying all three selection groups, SciQ included.

The accuracy rule needs target, neighbour and unrelated accuracy for one model.
The original round got those from two scorings and merged the trees afterwards:
target and neighbour from `questions_with_options.csv`, unrelated from
`sciq_unrelated.csv`, because the other-concept pool left the full model at
chance and `(Acc_F - 0.25)` was a denominator whose sign was not established.

Scoring is deterministic -- teacher-forced log-probs, no sampling -- and
score_mc.py does nothing with `--topic` but filter its topic column. So the same
numbers come out of one scoring over a manifest that already carries the SciQ
rows under each topic, at half the checkpoint I/O. That equivalence is not
assumed: slurm/score_selection_ens.slurm re-scores an already-scored candidate
through this manifest and diffs it against the original merged tree before any
ensemble is selected.

SciQ is topic-independent by construction, so the same 50 unrelated questions
appear under all three topics, exactly as the merged trees had them.

    python ember_eval/ensembles/build_combined_manifest.py --out <run>/manifest_selection.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACC = HERE.parent / "acc_selection"
TOPICS = ("rome", "baseball", "ai")
KEEP = ("target_selection", "neighbour_selection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-manifest", default=str(ACC / "questions_with_options.csv"))
    ap.add_argument("--sciq-manifest", default=str(ACC / "sciq_unrelated.csv"))
    ap.add_argument("--phase", default="selection", choices=["selection", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    keep = tuple(g.replace("selection", a.phase) for g in KEEP)
    want_unrelated = f"unrelated_{a.phase}"

    topic_rows = list(csv.DictReader(open(a.topic_manifest, encoding="utf-8")))
    sciq_rows = [r for r in csv.DictReader(open(a.sciq_manifest, encoding="utf-8"))
                 if r["question_group"] == want_unrelated]
    if not sciq_rows:
        raise SystemExit(f"no {want_unrelated} rows in {a.sciq_manifest}")

    fields = list(topic_rows[0])
    out_rows = []
    for t in TOPICS:
        kept = [r for r in topic_rows if r["topic"] == t and r["question_group"] in keep]
        if len(kept) != 100:
            raise SystemExit(f"{t}: expected 100 target+neighbour rows, got {len(kept)}")
        out_rows += kept
        for r in sciq_rows:
            row = {f: r.get(f, "") for f in fields}
            row["topic"] = t                      # sciq serves every topic
            row["question_group"] = want_unrelated
            row["phase"] = a.phase
            out_rows.append(row)

    ids = [(r["topic"], r["question_id"]) for r in out_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate (topic, question_id) in the combined manifest")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out}")
    for t in TOPICS:
        sub = [r for r in out_rows if r["topic"] == t]
        groups = sorted({r["question_group"] for r in sub})
        print(f"  {t:9s} n={len(sub)}  {groups}")


if __name__ == "__main__":
    main()
