#!/usr/bin/env python
"""One long-form per-question accuracy table for every current condition.

The scoring for this project landed in four separate trees over three rounds,
and each round answered a different question, so no single tree holds all of
it:

    <acc>/scored          selection phase, concept groups, unrelated = the
                          other-concept pool the protocol originally specified
    <acc>/scored_merged   the same selection phase with unrelated = SciQ
    <acc>/scored_sciq     the SciQ set on its own (selection and test halves)
    <acc>/scored_test     test phase, concept groups, unrelated = other-concept
    <ens>/scored_selection selection phase for the ensembles (unrelated = SciQ)
    <ens>/merged_test     test phase for everything, ensembles included, whose
                          unrelated group is SciQ for the ensembles and the
                          other-concept pool for the models scored earlier

That last line is the trap. `unrelated_test` does not name one set of
questions: for the ensembles it is SciQ, for everything else it is the
other-concept pool, and concatenating them under one group name would put 100
questions in a 50-question group and silently change every normalisation that
reads it. So the unrelated group is not taken on trust from the tree it came
from -- every unrelated group's ids are compared against the SciQ manifest and
the group is renamed `sciq_unrelated_{selection,test}` when they match. The
other-concept pool keeps the plain `unrelated_*` name.

Where two trees hold the same cell they are required to agree, not merged:
a disagreement means two different scorings are being mixed and is reported
rather than averaged away.

    python ember_eval/acc_selection/export_current_test.py \
        --out-per-question ember_eval/acc_selection/final/current_test_per_question.csv \
        --out-accuracy     ember_eval/acc_selection/final/current_test_accuracy.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ACC_ROOT = "/home/morg/NLP_2526b/galbarak2/runs/acc_selection"
ENS_ROOT = "/home/morg/NLP_2526b/galbarak2/runs/ensembles"

FULL = "FULL_lment-1b-control-2e-b131k"
CONCEPTS = [("rome", "Ancient Rome", "TWIN_lment-1b-norome-2e-b131k"),
            ("baseball", "Baseball", "TWIN_lment-1b-nobaseball-2e-b131k"),
            ("ai", "Artificial intelligence", "TWIN_lment-1b-noai-2e-b131k")]

# Frozen: the accuracy-selected winners (acc_selection/results_sciq/
# selected_checkpoints.json) and the ensembles' own frozen selection.
METHODS: Dict[str, List[Tuple[str, str]]] = {
    "rome": [("EMBER", "ember_rome_d200"), ("RMU", "rmu_rome_L6hi_a10"),
             ("SNMF", "snmf_rome_ratio_out"),
             ("RMU+EMBER", "rmuember_rome_L5mid_a100"),
             ("SNMF+EMBER", "snmfv2_rome_ratio_in")],
    "baseball": [("EMBER", "ember_baseball_d10"), ("RMU", "rmu_baseball_L6hi_a10"),
                 ("SNMF", "snmf_baseball_ratio_both"),
                 ("RMU+EMBER", "rmuember_baseball_L5mid_a100"),
                 ("SNMF+EMBER", "snmfv2_baseball_ratio_both")],
    "ai": [("EMBER", "ember_ai_d500"), ("RMU", "rmu_ai_L6hi_a10"),
           ("SNMF", "snmf_ai_ratio_in"),
           ("RMU+EMBER", "rmuember_ai_L5mid_a10"),
           ("SNMF+EMBER", "snmfv2_ai_ratio_in")],
}

COLUMNS = ["concept", "method", "checkpoint", "question_group", "phase", "question_id",
           "correct_option", "predicted_option", "is_correct", "correct_option_score",
           "option_0_score", "option_1_score", "option_2_score", "option_3_score"]

# scores are log-probabilities per character from separate GPU jobs; the
# project's measured cross-GPU drift is ~3e-5 nats
SCORE_TOL = 1e-4


def sciq_ids(manifest: Path) -> Dict[str, set]:
    out: Dict[str, set] = defaultdict(set)
    with open(manifest, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["question_group"]].add(r["question_id"])
    return dict(out)


def phase_of(group: str) -> str:
    return "test" if group.endswith("_test") else "selection"


def read_tree(path: Path, sciq: Dict[str, set]) -> Optional[Dict[str, List[dict]]]:
    """-> {canonical group: [row, ...]}, unrelated groups renamed if they are SciQ."""
    if not path.exists():
        return None
    by_group: Dict[str, List[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_group[r["question_group"]].append(r)
    out: Dict[str, List[dict]] = {}
    for g, rows in by_group.items():
        name = g
        if g.startswith("unrelated_"):
            ids = {r["question_id"] for r in rows}
            for sg, sids in sciq.items():
                if ids == sids:
                    name = "sciq_" + sg
                    break
        out[name] = rows
    return out


def to_record(concept: str, method: str, checkpoint: str, group: str, r: dict) -> dict:
    gold = int(r["gold_index"])
    scores = [r[f"score_opt{i}"] for i in range(4)]
    return {
        "concept": concept, "method": method, "checkpoint": checkpoint,
        "question_group": group, "phase": phase_of(group),
        "question_id": r["question_id"],
        "correct_option": gold, "predicted_option": int(r["predicted_index"]),
        "is_correct": int(r["is_correct"]),
        "correct_option_score": scores[gold],
        **{f"option_{i}_score": scores[i] for i in range(4)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-root", default=ACC_ROOT)
    ap.add_argument("--ens-root", default=ENS_ROOT)
    ap.add_argument("--sciq-manifest", default=str(HERE / "sciq_unrelated.csv"))
    ap.add_argument("--nllkl-results", default=str(REPO / "ember_eval/nll_kl/results_accwinners"),
                    help="join check: question ids must match the NLL/KL evaluation")
    ap.add_argument("--out-per-question", required=True)
    ap.add_argument("--out-accuracy", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output that this run does not strictly extend")
    a = ap.parse_args()

    A, E = Path(a.acc_root), Path(a.ens_root)
    sciq = sciq_ids(Path(a.sciq_manifest))
    problems: List[str] = []

    # highest precedence first: the tree each number was actually selected or
    # reported from, then the later merges that copied it
    def sources(label: str, topic: str) -> List[Tuple[str, Path]]:
        return [("acc/scored", A / "scored" / label / topic / "per_question.csv"),
                ("acc/scored_test", A / "scored_test" / label / topic / "per_question.csv"),
                ("acc/scored_sciq", A / "scored_sciq" / label / "sciq" / "per_question.csv"),
                ("acc/scored_merged", A / "scored_merged" / label / topic / "per_question.csv"),
                ("ens/scored_selection", E / "scored_selection" / label / topic / "per_question.csv"),
                ("ens/merged_test", E / "merged_test" / label / topic / "per_question.csv"),
                ("ens/merged_sciq", E / "merged_sciq" / label / "sciq" / "per_question.csv")]

    records: List[dict] = []
    provenance: Dict[str, str] = {}
    max_drift = 0.0
    for slug, concept, twin in CONCEPTS:
        for method, label in [("Full", FULL), ("Twin", twin)] + METHODS[slug]:
            seen: Dict[Tuple[str, str], dict] = {}
            for src_name, path in sources(label, slug):
                tree = read_tree(path, sciq)
                if tree is None:
                    continue
                for group, rows in tree.items():
                    for r in rows:
                        key = (group, r["question_id"])
                        rec = to_record(concept, method, label, group, r)
                        if key in seen:
                            old = seen[key]
                            for i in range(4):
                                d = abs(float(old[f"option_{i}_score"]) - float(rec[f"option_{i}_score"]))
                                max_drift = max(max_drift, d)
                                if d > SCORE_TOL:
                                    problems.append(
                                        f"{concept}/{method}/{group}/{r['question_id']}: "
                                        f"option {i} score differs by {d:.3g} between trees "
                                        f"({provenance.get(f'{label}|{group}')} vs {src_name})")
                            if old["is_correct"] != rec["is_correct"]:
                                problems.append(f"{concept}/{method}/{group}/{r['question_id']}: "
                                                f"is_correct differs between trees")
                            continue
                        seen[key] = rec
                        provenance.setdefault(f"{label}|{group}", src_name)
                        records.append(rec)
            if not seen:
                problems.append(f"{concept}/{method} ({label}): no scored tree found")

    # --- structural checks -------------------------------------------------
    by_cell: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in records:
        by_cell[(r["concept"], r["method"], r["question_group"])].append(r)
    for (c, m, g), rows in sorted(by_cell.items()):
        ids = [r["question_id"] for r in rows]
        if len(ids) != 50:
            problems.append(f"{c}/{m}/{g}: {len(ids)} questions, expected 50")
        if len(set(ids)) != len(ids):
            problems.append(f"{c}/{m}/{g}: duplicate question ids")
        for r in rows:
            vals = [r[f"option_{i}_score"] for i in range(4)]
            if any(v in ("", "nan", "inf", "-inf") for v in vals):
                problems.append(f"{c}/{m}/{g}/{r['question_id']}: non-finite option score")

    # every method in a (concept, group) must score the same questions
    for (c, g) in sorted({(k[0], k[2]) for k in by_cell}):
        sets = {m: {r["question_id"] for r in by_cell[(c, m, g)]}
                for m in {k[1] for k in by_cell if k[0] == c and k[2] == g}}
        ref_m, ref = sorted(sets.items())[0]
        for m, s in sets.items():
            if s != ref:
                problems.append(f"{c}/{g}: {m} question ids differ from {ref_m}")

    # the NLL/KL join: same ids, same groups
    nk = Path(a.nllkl_results)
    for slug, concept, _ in CONCEPTS:
        p = nk / slug / "results.json"
        if not p.exists():
            problems.append(f"{concept}: {p} missing, cannot check the NLL/KL join")
            continue
        items = json.loads(p.read_text())["rows"][0]["items"]
        for g in ("target_test", "neighbour_test", "unrelated_test",
                  "target_selection", "neighbour_selection", "unrelated_selection"):
            nll = {i["id"] for i in items if i["set"] == g}
            acc = {r["question_id"] for r in by_cell.get((concept, "Full", g), [])}
            if not acc:
                problems.append(f"{concept}/{g}: no accuracy rows to join against NLL/KL")
            elif nll != acc:
                problems.append(f"{concept}/{g}: accuracy ids do not match the NLL/KL ids "
                                f"({len(nll & acc)} of {len(nll)} shared)")

    # --- aggregate ---------------------------------------------------------
    agg = []
    for (c, m, g), rows in sorted(by_cell.items(),
                                  key=lambda kv: ([x[1] for x in CONCEPTS].index(kv[0][0]),
                                                  kv[0][2], kv[0][1])):
        n = len(rows)
        agg.append({"concept": c, "method": m, "checkpoint": rows[0]["checkpoint"],
                    "question_group": g, "phase": phase_of(g), "n": n,
                    "accuracy": f"{sum(r['is_correct'] for r in rows) / n:.10f}"})

    if problems:
        print("PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1

    def guard(path: Path, n_new: int) -> bool:
        """Refuse to shrink an existing export unless --force."""
        if not path.exists() or a.force:
            return True
        n_old = sum(1 for _ in open(path, encoding="utf-8")) - 1
        if n_new < n_old:
            print(f"REFUSING to overwrite {path}: {n_old} existing rows, "
                  f"{n_new} new ones. Re-run with --force if that is intended.",
                  file=sys.stderr)
            return False
        return True

    pq, ac = Path(a.out_per_question), Path(a.out_accuracy)
    if not (guard(pq, len(records)) and guard(ac, len(agg))):
        return 1
    pq.parent.mkdir(parents=True, exist_ok=True)

    records.sort(key=lambda r: ([x[1] for x in CONCEPTS].index(r["concept"]),
                                r["question_group"], r["method"], r["question_id"]))
    with open(pq, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    with open(ac, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(agg)

    print(f"wrote {pq} ({len(records)} rows)")
    print(f"wrote {ac} ({len(agg)} rows)")
    print(f"max cross-tree option-score drift: {max_drift:.3g} (tolerance {SCORE_TOL})")
    groups = sorted({r["question_group"] for r in records})
    print("groups:", ", ".join(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
