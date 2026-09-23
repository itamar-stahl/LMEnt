#!/usr/bin/env python
"""Recover all four options for every SELECTION question, and freeze the manifest.

The nll_kl set manifests carry the declarative stem and the correct continuation
but not the distractors, because answer NLL never needed them. The four options
live in completion_eval/data/completion_questions.json, the same file build_sets.py
drew from, so they are recovered by joining on (source_topic, source_split, question).

Every join is asserted, not assumed: exactly one source record, exactly four
distinct options, the correct answer appearing exactly once among them, and the
set manifest's own stem and completion agreeing with the source record.

    python build_manifest.py --out <dir>/questions_with_options.csv
"""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = HERE.parents[0] / "nll_kl" / "sets"
QUESTIONS = HERE.parents[0] / "completion_eval" / "data" / "completion_questions.json"
TOPICS = ("rome", "baseball", "ai")
SELECTION = ("target_selection", "neighbour_selection", "unrelated_selection")
TEST = ("target_test", "neighbour_test", "unrelated_test")


def build(groups):
    src = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows, problems = [], []
    for topic in TOPICS:
        manifest = json.loads((SETS / f"{topic}.json").read_text(encoding="utf-8"))
        for group in groups:
            for it in manifest["sets"][group]:
                pool = src.get(it["source_topic"], {}).get(it["source_split"], [])
                hits = [r for r in pool if r["q"].strip() == it["question"].strip()]
                if len(hits) != 1:
                    problems.append(f"{topic}/{group}/{it['id']}: {len(hits)} source matches")
                    continue
                rec = hits[0]
                opts = [o.strip() for o in rec["options"]]
                gold = rec["correct_answer"].strip()
                if len(opts) != 4:
                    problems.append(f"{topic}/{group}/{it['id']}: {len(opts)} options")
                if len(set(opts)) != 4:
                    problems.append(f"{topic}/{group}/{it['id']}: options not distinct")
                if sum(1 for o in opts if o == gold) != 1:
                    problems.append(f"{topic}/{group}/{it['id']}: gold occurs != 1 time")
                if it["completion"] != " " + gold:
                    problems.append(f"{topic}/{group}/{it['id']}: completion != ' '+gold")
                if it["stem"].strip() != rec["stem"].strip():
                    problems.append(f"{topic}/{group}/{it['id']}: stem mismatch")
                rows.append({
                    "topic": topic, "question_group": group, "phase": "selection",
                    "question_id": it["id"], "source_topic": it["source_topic"],
                    "source_split": it["source_split"], "question": it["question"],
                    "stem": it["stem"], "correct_answer": gold,
                    "option_0": opts[0], "option_1": opts[1],
                    "option_2": opts[2], "option_3": opts[3],
                    "gold_index": opts.index(gold),
                })
    return rows, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("selection", "test"), default="selection",
                    help="selection: the hyperparameter-choice half (default). "
                         "test: the held-out half, for the final analysis only -- "
                         "it must never be read before the winners are frozen.")
    a = ap.parse_args()
    rows, problems = build(SELECTION if a.phase == "selection" else TEST)
    if problems:
        raise SystemExit("JOIN FAILED:\n  " + "\n  ".join(problems[:40]))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{len(rows)} questions -> {out}")
    print(f"sha256 {digest}")
    for t in TOPICS:
        for g in (SELECTION if a.phase == "selection" else TEST):
            n = sum(1 for r in rows if r["topic"] == t and r["question_group"] == g)
            assert n == 50, f"{t}/{g}: {n}"
    print("all 9 (topic, group) cells hold exactly 50 questions")


if __name__ == "__main__":
    main()
