#!/usr/bin/env python
"""Held-out accuracy and H_test for every condition, SciQ as the unrelated group.

One table covering the full model, the twins, the three standalone methods and
the two ensembles, all on the same target, neighbour and SciQ test questions,
normalised against the same full-model denominators:

    norm_g   = clip((acc_g - 0.25) / (acc_full_g - 0.25), 0, 1)
    efficacy = 1 - norm_target
    preserv. = HM(norm_neighbour, norm_sciq)
    H_test   = HM(efficacy, preservation)

The unrelated group is SciQ throughout. The scored tree this reads was produced
from a manifest whose unrelated rows carry source_topic='sciq', and the 50 ids
are identical to the nll_kl SciQ set -- both are asserted here rather than
assumed, because mixing the older other-concept pool into one row would silently
change every denominator.

    python ember_eval/ensembles/panel_a_sciq.py --scored <merged_test> --out <csv>
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Ember-on-LMEnt"))
from ember.evals.harmonic import harmonic_mean  # noqa: E402

CHANCE = 0.25
GROUPS = ("target_test", "neighbour_test", "unrelated_test")
FULL = "FULL_lment-1b-control-2e-b131k"
CONCEPTS = [("rome", "Ancient Rome", "TWIN_lment-1b-norome-2e-b131k"),
            ("baseball", "Baseball", "TWIN_lment-1b-nobaseball-2e-b131k"),
            ("ai", "Artificial intelligence", "TWIN_lment-1b-noai-2e-b131k")]
# frozen; not reselected here
ROWS = {
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
# reconciliation targets supplied with the request; checked, never forced
EXPECTED = {("Ancient Rome", "RMU+EMBER"): 0.000, ("Ancient Rome", "SNMF+EMBER"): 0.000,
            ("Baseball", "RMU+EMBER"): 0.469, ("Baseball", "SNMF+EMBER"): 0.563,
            ("Artificial intelligence", "RMU+EMBER"): 0.829,
            ("Artificial intelligence", "SNMF+EMBER"): 0.816}


def _group(f: Path, g: str):
    if not f.exists():
        return None, None
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8"))
            if r["question_group"] == g]
    if not rows:
        return None, None
    return (sum(int(r["is_correct"]) for r in rows) / len(rows),
            {r["question_id"] for r in rows})


def read(scored: Path, sciq_tree: Path, label: str, topic: str):
    """target and neighbour from the topic scoring, unrelated from the SciQ one.

    The topic tree's `unrelated_test` is the OLDER other-concept pool for every
    model scored in the original round -- the references and the standalone
    methods. Reading it would normalise SciQ ensemble accuracies against an
    other-concept denominator. The SciQ group therefore comes from scored_sciq
    where that exists, and from the topic scoring only for models whose topic
    scoring already used SciQ (the ensembles). Either way the ids are checked
    against the SciQ manifest by the caller.
    """
    acc, ids = {}, {}
    for g in ("target_test", "neighbour_test"):
        acc[g], ids[g] = _group(scored / label / topic / "per_question.csv", g)
    a, i = _group(sciq_tree / label / "sciq" / "per_question.csv", "unrelated_test")
    if a is None:
        a, i = _group(scored / label / topic / "per_question.csv", "unrelated_test")
    acc["unrelated_test"], ids["unrelated_test"] = a, i
    if any(v is None for v in acc.values()):
        return None, None
    return acc, ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--sciq-tree", required=True,
                    help="tree holding <label>/sciq/per_question.csv with unrelated_test")
    ap.add_argument("--sciq-manifest", default=str(HERE.parent / "acc_selection/sciq_unrelated.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scored, sciq_tree, out = Path(a.scored), Path(a.sciq_tree), Path(a.out)
    sciq_ids = {r["question_id"] for r in csv.DictReader(open(a.sciq_manifest, encoding="utf-8"))
                if r["question_group"] == "unrelated_test"}

    recs, problems = [], []
    for slug, cname, twin in CONCEPTS:
        facc, fids = read(scored, sciq_tree, FULL, slug)
        if not facc:
            raise SystemExit(f"{slug}: full model not scored")
        if fids["unrelated_test"] != sciq_ids:
            raise SystemExit(f"{slug}: full model's unrelated group is NOT the SciQ set "
                             "-- refusing to mix unrelated sources")
        for g in GROUPS:
            if facc[g] <= CHANCE:
                raise SystemExit(f"{slug}/{g}: full accuracy {facc[g]:.3f} <= chance; "
                                 "normalisation undefined")
        for method, label in [("Full", FULL), ("Twin", twin)] + ROWS[slug]:
            acc, ids = read(scored, sciq_tree, label, slug)
            if not acc:
                problems.append(f"{cname}/{method} ({label}): not scored")
                continue
            for g in GROUPS:
                if ids[g] != fids[g]:
                    problems.append(f"{cname}/{method}/{g}: question ids differ from the full model")
            if ids["unrelated_test"] != sciq_ids:
                problems.append(f"{cname}/{method}: unrelated group is not the SciQ set")
            norm = {g: min(1.0, max(0.0, (acc[g] - CHANCE) / (facc[g] - CHANCE))) for g in GROUPS}
            eff = 1.0 - norm["target_test"]
            pres = harmonic_mean([norm["neighbour_test"], norm["unrelated_test"]])
            H = harmonic_mean([eff, pres])
            recs.append({"concept": cname, "method": method, "checkpoint": label,
                         "unrelated_source": "sciq",
                         **{f"acc_{g.split('_')[0]}": f"{acc[g]:.6f}" for g in GROUPS},
                         **{f"full_{g.split('_')[0]}": f"{facc[g]:.6f}" for g in GROUPS},
                         **{f"norm_{g.split('_')[0]}": f"{norm[g]:.10f}" for g in GROUPS},
                         "phi_efficacy": f"{eff:.10f}", "phi_preservation": f"{pres:.10f}",
                         "H_test": f"{H:.10f}"})

    cols = list(recs[0])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader(); w.writerows(recs)

    print(f"{'concept':24s} {'method':12s} {'target':>7s} {'neigh':>7s} {'sciq':>7s} {'H_test':>8s}  reconcile")
    for r in recs:
        k = (r["concept"], r["method"])
        exp = EXPECTED.get(k)
        note = ""
        if exp is not None:
            d = abs(float(r["H_test"]) - exp)
            note = f"expected {exp:.3f}, got {float(r['H_test']):.3f} -> " + \
                   ("agrees" if d < 0.005 else f"DISAGREES by {d:.3f}")
        print(f"{r['concept']:24s} {r['method']:12s} {float(r['acc_target']):7.2f} "
              f"{float(r['acc_neighbour']):7.2f} {float(r['acc_unrelated']):7.2f} "
              f"{float(r['H_test']):8.4f}  {note}")
    if problems:
        print("\nPROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        raise SystemExit(1)
    print(f"\nwrote {out} ({len(recs)} rows)")


if __name__ == "__main__":
    main()
